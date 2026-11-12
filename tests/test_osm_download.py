"""Offline tests of the OpenStreetMap download: a fake Overpass server built from
the stored demo data answers the queries, so no internet is needed."""
import re
from xml.sax.saxutils import quoteattr

import geopandas as gpd
import pandas as pd
import pytest
import requests
from shapely.geometry import Polygon, box

from vertical25d import osm
from vertical25d.demo import DEMO_DIR


# ----------------------------------------------------------------------------- fake Overpass server
def _demo_elements():
    b = gpd.read_file(f"{DEMO_DIR}/buildings.gpkg").to_crs(4326)
    p = gpd.read_file(f"{DEMO_DIR}/pois.gpkg").to_crs(4326)
    nodes, ways = {}, []
    nid = 10**9
    for _, r in b.iterrows():
        g = r.geometry
        g = max(g.geoms, key=lambda q: q.area) if g.geom_type == "MultiPolygon" else g
        refs = []
        for x, y in g.exterior.coords[:-1]:
            nid += 1
            nodes[nid] = (x, y, {})
            refs.append(nid)
        refs.append(refs[0])
        tags = {"building": r.get("building") or "yes"}
        if pd.notna(r.get("building:levels", None)):
            tags["building:levels"] = str(r["building:levels"])
        ways.append((int(str(r.get("osmid", len(ways)))), refs, tags, g))
    pois = []
    for i, r in p.iterrows():
        nid += 1
        cat = r.get("category") or "restaurant"
        pois.append((nid, r.geometry.x, r.geometry.y, {"amenity": str(cat), "name": str(r.get("name") or "")}))
    return nodes, ways, pois


class FakeOverpass:
    """Answers Overpass QL queries with demo elements inside the query polygons."""

    def __init__(self, behaviour=None):
        self.nodes, self.ways, self.pois = _demo_elements()
        self.behaviour = behaviour or {}     # host -> list of actions: "down", "busy", "error", "ok"
        self.calls = []

    def _answer(self, query):
        polys = [Polygon([(float(c[i + 1]), float(c[i])) for i in range(0, len(c), 2)])
                 for c in (s.split() for s in re.findall(r'poly:"([^"]+)"', query))]
        area = polys[0]
        for q in polys[1:]:
            area = area.union(q)
        out = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="fake">']
        if '["building"]' in query:
            keep = [w for w in self.ways if w[3].intersects(area)]
            used = {n for w in keep for n in w[1]}
            for n in sorted(used):
                x, y, _ = self.nodes[n]
                out.append(f'<node id="{n}" lat="{y:.7f}" lon="{x:.7f}"/>')
            for wid, refs, tags, _ in keep:
                out.append(f'<way id="{wid}">' + "".join(f'<nd ref="{r}"/>' for r in refs)
                           + "".join(f"<tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items()) + "</way>")
        else:
            for nid, x, y, tags in self.pois:
                if area.contains(box(x, y, x, y).centroid):
                    out.append(f'<node id="{nid}" lat="{y:.7f}" lon="{x:.7f}">'
                               + "".join(f"<tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items())
                               + "</node>")
        out.append("</osm>")
        return "\n".join(out).encode()

    def post(self, url, data=None, **kw):
        host = url.split("/")[2]
        self.calls.append(host)
        plan = self.behaviour.get(host, [])
        action = plan.pop(0) if plan else "ok"
        if action == "down":
            raise requests.ConnectionError(f"cannot reach {host}")
        if action == "busy":
            return _Resp(429, b"busy")
        if action == "error":
            return _Resp(200, b'<?xml version="1.0"?><osm version="0.6">\n<remark> runtime error: Query timed out '
                              b'in "query" at line 3 after 91 seconds. </remark>\n</osm>')
        return _Resp(200, self._answer(data["data"]))


class _Resp:
    def __init__(self, code, body):
        self.status_code, self.reason, self.body = code, "OK" if code == 200 else "Too Many Requests", body

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def fake(monkeypatch, tmp_path):
    srv = FakeOverpass()
    monkeypatch.setattr(requests.Session, "post", lambda self, url, **kw: srv.post(url, **kw))
    monkeypatch.setattr(osm, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(osm, "BUSY_WAIT_S", 0)
    monkeypatch.setitem(osm._preferred, "url", None)
    monkeypatch.delenv("VERTICAL25D_OVERPASS_URL", raising=False)
    return srv


def _aoi():
    from vertical25d import read_aoi
    a = read_aoi(f"{DEMO_DIR}/aoi.kml")
    return a.geometry.iloc[0], a.estimate_utm_crs()


# ----------------------------------------------------------------------------- tests
def test_buildings_and_pois_download(fake):
    geom, utm = _aoi()
    b = osm.fetch_buildings(geom, utm, log=lambda m: None)
    p = osm.fetch_pois(geom, log=lambda m: None)
    assert len(b) > 2000 and (b.geom_type.isin(["Polygon", "MultiPolygon"])).all()
    assert len(p) > 200 and (p.geom_type == "Point").all()


def test_dead_and_busy_servers_are_skipped_quickly(fake):
    geom, utm = _aoi()
    fake.behaviour = {"overpass-api.de": ["down"], "overpass.private.coffee": ["busy", "ok"]}
    logs = []
    b = osm.fetch_buildings(geom, utm, log=logs.append)
    assert len(b) > 2000
    assert fake.calls[:3] == ["overpass-api.de", "overpass.private.coffee", "overpass.private.coffee"]
    assert any("busy" in m for m in logs)
    # the server that worked is asked first next time
    osm.fetch_pois(geom, log=lambda m: None)
    assert fake.calls[3] == "overpass.private.coffee"


def test_cache_avoids_second_download(fake):
    geom, utm = _aoi()
    osm.fetch_buildings(geom, utm, log=lambda m: None)
    n = len(fake.calls)
    osm.fetch_buildings(geom, utm, log=lambda m: None)
    assert len(fake.calls) == n


def test_large_area_is_tiled_without_duplicates(fake, monkeypatch):
    geom, utm = _aoi()
    whole = osm.fetch_buildings(geom, utm, log=lambda m: None)
    monkeypatch.setattr(osm, "TILE_KM2", 0.3)       # force several tiles over the 1.45 km² demo area
    monkeypatch.setattr(osm, "CACHE_DIR", osm.CACHE_DIR + "_tiles")
    logs = []
    tiled = osm.fetch_buildings(geom, utm, log=logs.append)
    n_parts = len(osm._tiles(geom, utm))
    assert n_parts > 3 and any(f"in {n_parts} parts" in m for m in logs)
    assert set(tiled["bid"]) == set(whole["bid"]) and tiled["bid"].is_unique


def test_query_timeout_on_every_server_splits_the_part(fake, monkeypatch):
    geom, utm = _aoi()
    monkeypatch.setattr(osm, "MIN_TILE_KM2", 0.05)
    fake.behaviour = {h: ["error"] for h in ["overpass-api.de", "overpass.private.coffee",
                                             "overpass.kumi.systems", "maps.mail.ru"]}
    logs = []
    b = osm.fetch_buildings(geom, utm, log=logs.append)
    assert any("splitting" in m for m in logs) and len(b) > 2000


def test_all_servers_down_gives_a_clear_error(fake):
    geom, utm = _aoi()
    fake.behaviour = {h: ["down"] * 5 for h in ["overpass-api.de", "overpass.private.coffee",
                                                "overpass.kumi.systems", "maps.mail.ru"]}
    with pytest.raises(RuntimeError, match="internet connection"):
        osm.fetch_buildings(geom, utm, log=lambda m: None)
    assert len(fake.calls) == 4                     # one quick try per server, no endless retries


def test_tiled_download_gives_identical_analysis(fake, monkeypatch, tmp_path):
    """Downloading a large area in parts must not change any result."""
    from vertical25d import Settings, run
    from vertical25d.demo import demo_inputs

    def go(tile):
        monkeypatch.setattr(osm, "TILE_KM2", tile)
        monkeypatch.setattr(osm, "CACHE_DIR", str(tmp_path / f"c{tile}"))
        inp = demo_inputs()
        inp.buildings, inp.pois = None, None
        r = run(inp, Settings(n_sim=30, seed=1), out_dir=str(tmp_path / f"o{tile}"), log=lambda m: None)
        return len(r.buildings), r.summary["pois"]["n_classified"], r.summary["indices"]["grid"]

    assert go(9.0) == go(0.25)
