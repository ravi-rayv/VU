"""Offline tests of the batched Earth Engine extraction used for large areas."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from vertical25d import eebatch
from vertical25d.demo import DEMO_DIR


class _Info:
    def __init__(self, feats):
        self.feats = feats

    def getInfo(self):  # noqa: N802 - mimics ee
        return {"type": "FeatureCollection", "features": self.feats}


class FakeEE:
    """Answers with h = number in the bid; refuses batches larger than ``max_batch``."""

    def __init__(self, max_batch=10**9, rate_errors=0, dead_bids=()):
        self.max_batch, self.rate_errors, self.dead = max_batch, rate_errors, set(dead_bids)
        self.calls, self.sizes = 0, []

    def __call__(self, fc):
        self.calls += 1
        feats = fc["features"]
        self.sizes.append(len(feats))
        if self.rate_errors > 0:
            self.rate_errors -= 1
            raise RuntimeError("Too many concurrent aggregations.")
        if len(feats) > self.max_batch:
            raise RuntimeError("Computation timed out.")
        if any(f["properties"]["bid"] in self.dead for f in feats):
            raise RuntimeError("Internal error")
        return _Info([{"properties": {"bid": f["properties"]["bid"], "median": 3.0, "p90": 4.0}} for f in feats])


@pytest.fixture
def bld(monkeypatch, tmp_path):
    monkeypatch.setattr(eebatch, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(eebatch.time, "sleep", lambda s: None)
    return gpd.read_file(f"{DEMO_DIR}/buildings.gpkg").to_crs(4326)[["bid", "geometry"]]


COLS = ["bid", "median", "p90"]


def test_batches_are_spatially_compact(bld):
    chunks = eebatch.make_chunks(bld, 200)
    utm = bld.estimate_utm_crs()
    spans = [np.hypot(*np.ptp(np.c_[g.x, g.y], axis=0))
             for g in (c.geometry.to_crs(utm).centroid for c in chunks)]
    whole = np.hypot(*np.ptp(np.c_[bld.to_crs(utm).centroid.x, bld.to_crs(utm).centroid.y], axis=0))
    assert sum(len(c) for c in chunks) == len(bld)
    assert np.median(spans) < 0.6 * whole


def test_failing_batches_are_split_not_dropped(bld):
    ee = FakeEE(max_batch=300)
    logs = []
    df = eebatch.run_batches(ee, eebatch.make_chunks(bld, 1500), COLS, workers=2, log=logs.append, label="heights")
    assert set(df["bid"]) == set(bld["bid"])
    assert max(s for s in ee.sizes if s <= 300) <= 300 and any("splitting" in m for m in logs)


def test_rate_limit_is_retried_without_splitting(bld):
    ee = FakeEE(rate_errors=3)
    df = eebatch.run_batches(ee, eebatch.make_chunks(bld, 5000), COLS, workers=1, log=lambda m: None)
    assert len(df) == len(bld) and set(ee.sizes) == {len(bld)}


def test_only_bad_buildings_are_lost(bld):
    dead = [bld["bid"].iloc[17]]
    logs = []
    df = eebatch.run_batches(FakeEE(dead_bids=dead), eebatch.make_chunks(bld, 1000), COLS, workers=2,
                             log=logs.append, label="heights")
    missing = set(bld["bid"]) - set(df["bid"])
    assert dead[0] in missing and len(missing) <= eebatch.MIN_SPLIT
    assert any("got no heights" in m for m in logs)


def test_interrupted_run_resumes_from_saved_batches(bld):
    chunks = eebatch.make_chunks(bld, 300)

    class Dies(FakeEE):
        def __call__(self, fc):
            if self.calls >= 3:
                raise KeyboardInterrupt
            return super().__call__(fc)

    with pytest.raises(KeyboardInterrupt):
        eebatch.run_batches(Dies(), chunks, COLS, workers=1, log=lambda m: None, cache_tag="t")
    ee2 = FakeEE()
    logs = []
    df = eebatch.run_batches(ee2, chunks, COLS, workers=1, log=logs.append, cache_tag="t")
    assert len(df) == len(bld) and ee2.calls == len(chunks) - 3
    assert any("reused" in m for m in logs)
