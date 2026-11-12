"""Download OpenStreetMap building footprints and points of interest (POIs)
for a study area from the public Overpass servers.

The download is done here (not by ``osmnx.features_from_polygon``) so that it
never hangs silently: large areas are fetched in tiles with progress messages,
each server gets a short connect timeout and a bounded wait, busy servers are
retried once and then skipped, and finished tiles are cached on disk so a
re-run does not download them again. ``osmnx`` only turns the downloaded OSM
XML into geometries.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import time

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from .config import OVERPASS_URLS

POI_TAGS = {"amenity": True, "shop": True, "office": True, "tourism": True,
            "leisure": True, "healthcare": True, "craft": True}

POI_KEYS = ["amenity", "shop", "office", "tourism", "leisure", "healthcare", "craft", "sport",
            "level", "addr:floor", "addr:unit", "addr:full", "addr:housenumber", "addr:street",
            "brand", "cuisine", "name"]

# ----------------------------------------------------------------------------- Overpass download
QUERY_TIMEOUT_S = 90       # server-side limit for one query ([timeout:...]); bigger parts are split
CONNECT_TIMEOUT_S = 10     # give up on a server that does not answer within this
READ_TIMEOUT_S = QUERY_TIMEOUT_S + 30   # max silence while waiting for / streaming the answer
BUSY_WAIT_S = 10           # one wait-and-retry when a server says it is busy (HTTP 429 / 504)
TILE_KM2 = 9.0             # areas larger than this are downloaded in tiles
MIN_TILE_KM2 = 0.5         # do not split tiles below this when servers time out
USER_AGENT = "vertical25d (https://github.com/ravi-rayv/vertical-landuse-stratification)"
CACHE_DIR = os.path.join(tempfile.gettempdir(), "vertical25d_osm_cache")

_preferred = {"url": None}   # the server that last worked is tried first


class OverpassError(RuntimeError):
    """A server answered, but with an error (query timed out, out of memory...)."""


def _servers():
    urls = list(OVERPASS_URLS)
    extra = os.environ.get("VERTICAL25D_OVERPASS_URL")      # e.g. a private or regional server
    if extra:
        urls.insert(0, extra.rstrip("/"))
    if _preferred["url"] in urls:
        urls.remove(_preferred["url"]); urls.insert(0, _preferred["url"])
    return urls


def _poly_filters(geom, max_pts=800):
    """Overpass ``poly:"lat lon ..."`` filters (one per outer ring)."""
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    out = []
    for p in parts:
        tol = 0.00002
        while len(p.exterior.coords) > max_pts:
            p = p.simplify(tol); tol *= 2
        out.append("poly:\"" + " ".join(f"{y:.6f} {x:.6f}" for x, y in p.exterior.coords[:-1]) + "\"")
    return out


def _query(tags, geom):
    sel = []
    for f in _poly_filters(geom):
        for k, v in tags.items():
            vals = [None] if v is True else ([v] if isinstance(v, str) else list(v))
            for val in vals:
                t = f'["{k}"]' if val is None else f'["{k}"="{val}"]'
                sel.append(f"nwr{t}({f});")
    return (f"[out:xml][timeout:{QUERY_TIMEOUT_S}];\n(\n" + "\n".join(sel) + "\n);\n(._;>;);\nout body;")


def _check_xml(path):
    """Overpass reports query errors inside an otherwise normal answer."""
    with open(path, "rb") as f:
        head = f.read(4096)
        f.seek(max(0, os.path.getsize(path) - 4096))
        tail = f.read()
    if b"<osm" not in head:
        raise OverpassError(f"unexpected answer: {head[:200]!r}")
    m = re.search(rb"<remark>(.*?)</remark>", tail, re.S)
    if m and b"error" in m.group(1).lower():
        raise OverpassError(m.group(1).decode("utf-8", "replace").strip()[:300])


def _download(query, log=print, label=""):
    """Download one query, trying each server in turn. Returns the path of the OSM XML file."""
    import requests
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, hashlib.sha1(query.encode()).hexdigest() + ".osm")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        log(f"OSM{label}: using data downloaded earlier")
        return path
    errors, server_errors = [], 0
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    for url in _servers():
        host = url.split("/")[2]
        for attempt in (1, 2):
            t0 = time.time()
            try:
                log(f"OSM{label}: asking {host}" + (" again" if attempt == 2 else ""))
                with sess.post(url + "/interpreter", data={"data": query}, stream=True,
                               timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)) as r:
                    if r.status_code in (429, 504) and attempt == 1:
                        log(f"OSM{label}: {host} is busy (HTTP {r.status_code}); waiting {BUSY_WAIT_S} s")
                        time.sleep(BUSY_WAIT_S)
                        continue
                    if r.status_code != 200:
                        raise requests.HTTPError(f"HTTP {r.status_code} {r.reason}")
                    tmp, size, next_note = path + ".part", 0, 5_000_000
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 16):
                            f.write(chunk); size += len(chunk)
                            if size >= next_note:
                                log(f"OSM{label}: {size / 1e6:,.0f} MB received")
                                next_note += 5_000_000
                try:
                    _check_xml(tmp)
                except OverpassError:
                    server_errors += 1
                    raise
                os.replace(tmp, path)
                _preferred["url"] = url
                log(f"OSM{label}: {size / 1e6:,.1f} MB from {host} in {time.time() - t0:,.0f} s")
                return path
            except Exception as e:  # noqa: BLE001 - try the next server
                msg = f"{type(e).__name__}: {e}"[:200]
                errors.append(f"{host}: {msg}")
                log(f"OSM{label}: {host} failed ({msg}); trying the next server")
                break
    if server_errors and server_errors == len(errors):
        raise OverpassError("; ".join(errors))
    raise RuntimeError("Could not download OpenStreetMap data from any Overpass server. Check your internet "
                       "connection, or try again in a few minutes (the public servers are sometimes overloaded). "
                       "Details: " + " | ".join(errors))


def _tiles(geom, utm, tile_km2=None):
    """Split the study area into roughly square tiles of at most ``tile_km2`` (default ``TILE_KM2``)."""
    tile_km2 = tile_km2 or TILE_KM2
    g = gpd.GeoSeries([geom], crs=4326).to_crs(utm).iloc[0]
    if g.area / 1e6 <= tile_km2:
        return [geom]
    side = math.sqrt(tile_km2) * 1000
    minx, miny, maxx, maxy = g.bounds
    out = []
    for x in np.arange(minx, maxx, side):
        for y in np.arange(miny, maxy, side):
            t = g.intersection(box(x, y, x + side, y + side))
            if not t.is_empty and t.area > 1:
                out.append(t)
    return list(gpd.GeoSeries(out, crs=utm).to_crs(4326))


def _split4(geom, utm):
    g = gpd.GeoSeries([geom], crs=4326).to_crs(utm).iloc[0]
    minx, miny, maxx, maxy = g.bounds
    mx, my = (minx + maxx) / 2, (miny + maxy) / 2
    parts = [g.intersection(box(*b)) for b in [(minx, miny, mx, my), (mx, miny, maxx, my),
                                               (minx, my, mx, maxy), (mx, my, maxx, maxy)]]
    return list(gpd.GeoSeries([p for p in parts if not p.is_empty and p.area > 1], crs=utm).to_crs(4326))


def _parse(path, geom, tags):
    import osmnx as ox
    try:
        return ox.features_from_xml(path, polygon=geom, tags=tags)
    except Exception as e:  # noqa: BLE001
        if "No matching features" in str(e) or "InsufficientResponse" in type(e).__name__:
            return None
        raise


def _area_km2(geom, utm):
    return gpd.GeoSeries([geom], crs=4326).to_crs(utm).area.iloc[0] / 1e6


def _features(polygon, tags, log=print, timeout=None, what="features"):
    """All OSM features with ``tags`` in ``polygon`` (EPSG:4326), as an osmnx-style GeoDataFrame."""
    polygon = polygon.buffer(0)
    utm = gpd.GeoSeries([polygon], crs=4326).estimate_utm_crs()
    todo = _tiles(polygon, utm)
    n0 = len(todo)
    log(f"OSM: downloading {what}" + (f" in {n0} parts" if n0 > 1 else ""))
    parts, done = [], 0
    while todo:
        tile = todo.pop(0)
        done += 1
        label = f" [{what} {done}/{done + len(todo)}]" if n0 > 1 or done > 1 else f" [{what}]"
        try:
            path = _download(_query(tags, tile), log=log, label=label)
        except OverpassError as e:
            if _area_km2(tile, utm) / 4 < MIN_TILE_KM2:
                raise RuntimeError(f"OpenStreetMap servers could not answer even for a small part: {e}") from e
            log(f"OSM{label}: too much data for the servers ({e}); splitting this part into 4")
            todo = _split4(tile, utm) + todo
            done -= 1
            continue
        g = _parse(path, tile, tags)
        if g is not None and len(g):
            parts.append(g)
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    gdf = pd.concat(parts)
    gdf = gdf[~gdf.index.duplicated(keep="first")]          # features on tile edges appear twice
    gdf = gdf.sort_index()        # same order however the area was split, so results are reproducible
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=4326)


def _flatten_index(gdf):
    gdf = gdf.reset_index()
    return gdf.rename(columns={"id": "osmid", "element": "osm_type", "element_type": "osm_type"})


def fetch_buildings(aoi_geom, utm, log=print) -> gpd.GeoDataFrame:
    """OSM building polygons whose representative point lies in the study area."""
    gdf = _features(aoi_geom, {"building": True}, log=log, what="buildings")
    if gdf.empty:
        raise RuntimeError("No OSM buildings found in the study area")
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = _flatten_index(gdf)
    return clean_buildings(gdf, aoi_geom, utm)


def clean_buildings(gdf, aoi_geom, utm) -> gpd.GeoDataFrame:
    """Standardise columns: bid, osmid, osm_type, building, building_levels, name, geometry."""
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    gdf.columns = [c.replace(":", "_") for c in gdf.columns]
    inside = gdf.to_crs(utm).representative_point().to_crs(4326).within(aoi_geom).values
    gdf = gdf[inside]
    for c in ["osm_type", "osmid", "building", "building_levels", "name"]:
        if c not in gdf.columns:
            gdf[c] = np.nan
    if gdf["osmid"].isna().all():
        gdf["osmid"] = np.arange(len(gdf))
        gdf["osm_type"] = "x"
    if "bid" not in gdf.columns:
        gdf["bid"] = gdf["osm_type"].astype(str).str[0] + gdf["osmid"].astype(str)
    gdf["building_levels"] = pd.to_numeric(gdf["building_levels"], errors="coerce")
    gdf["name"] = gdf["name"].astype(object).fillna("").astype(str)
    gdf["building"] = gdf["building"].fillna("yes").astype(str)
    gdf = gdf[["bid", "osmid", "osm_type", "building", "building_levels", "name", "geometry"]]
    return gdf.drop_duplicates("bid").reset_index(drop=True)


def fetch_pois(aoi_geom, log=print) -> gpd.GeoDataFrame:
    """OSM POIs (nodes, and centroids of tagged ways) with the tags the method needs."""
    gdf = _features(aoi_geom, POI_TAGS, log=log, what="POIs")
    if gdf.empty:
        raise RuntimeError("No OSM POIs found in the study area")
    gdf = _flatten_index(gdf)
    return standardise_pois(gdf, aoi_geom)


KV = re.compile(r'"([^"]+)"=>"((?:[^"\\]|\\.)*)"')


def standardise_pois(gdf, aoi_geom) -> gpd.GeoDataFrame:
    """Turn any POI layer (OSM download, QGIS export with ``other_tags``,
    GeoJSON/GPKG) into points with columns amenity, shop, office, ...,
    addr_floor, level, name."""
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    if "other_tags" in gdf.columns:  # QGIS/ogr OSM export: hstore-like string
        tags = gdf["other_tags"].fillna("").map(lambda s: dict(KV.findall(str(s))))
        for k in POI_KEYS:
            if k.replace(":", "_") not in gdf.columns and k not in gdf.columns:
                gdf[k] = tags.map(lambda d, k=k: d.get(k))
    gdf.columns = [c.replace(":", "_") for c in gdf.columns]
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    for k in POI_KEYS:
        c = k.replace(":", "_")
        if c not in gdf.columns:
            gdf[c] = None
    # polygons (shops mapped as buildings, malls...) -> representative point
    poly = ~gdf.geom_type.isin(["Point"])
    if poly.any():
        utm = gdf.estimate_utm_crs()
        gdf.loc[poly, "geometry"] = gdf.loc[poly].to_crs(utm).representative_point().to_crs(4326).values
    gdf = gdf[gdf.geom_type == "Point"]
    gdf = gdf[gdf.within(aoi_geom)]
    if "osmid" in gdf.columns:
        otype = gdf["osm_type"] if "osm_type" in gdf.columns else pd.Series("n", index=gdf.index)
        gdf["osm_id"] = otype.astype(object).fillna("n").astype(str).str[0] + gdf["osmid"].astype(str)
    elif "osm_id" not in gdf.columns:
        gdf["osm_id"] = [f"p{i}" for i in range(len(gdf))]
    for c in gdf.columns:
        if c != "geometry" and gdf[c].dtype == object:
            gdf[c] = gdf[c].map(lambda v: v if (v is None or isinstance(v, str)) else (None if pd.isna(v) else str(v)))
    return gdf.reset_index(drop=True)
