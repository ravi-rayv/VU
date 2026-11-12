"""Robust batched Earth Engine extraction for large study areas.

Large areas are processed as many small requests and then combined:

* buildings are ordered by neighbourhood (1 km strips, snake order) before
  being cut into batches, so each request covers a small patch of the city
  instead of buildings scattered over the whole area (scattered batches make
  Earth Engine compute the mosaic for the entire area and time out);
* a batch that keeps failing (time-out, memory, payload) is split in half and
  retried, down to a few dozen buildings, instead of being dropped;
* "too many requests" errors are retried with back-off, not split;
* every finished batch is saved on disk, so if a run stops (lost connection,
  closed laptop) the next run with the same buildings resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(tempfile.gettempdir(), "vertical25d_ee_cache")
MIN_SPLIT = 50            # do not split batches below this many buildings
RATE_WORDS = ("too many concurrent", "too many requests", "quota", "rate limit", "429")


def spatial_order(gdf, cell_m: float = 1000.0):
    """Order buildings in 1 km strips (snake order) so consecutive rows are neighbours."""
    if len(gdf) == 0:
        return gdf
    utm = gdf.estimate_utm_crs()
    c = gdf.geometry.to_crs(utm).representative_point()
    cx = np.floor(c.x.to_numpy() / cell_m).astype(np.int64)
    cy = np.floor(c.y.to_numpy() / cell_m).astype(np.int64)
    snake = np.where(cy % 2 == 0, cx, -cx)
    return gdf.iloc[np.lexsort((c.x.to_numpy(), snake, cy))]


def make_chunks(gdf, chunk_size: int):
    """Neighbourhood-compact batches of at most ``chunk_size`` buildings (``bid`` + ``geometry``)."""
    send = spatial_order(gdf[["bid", "geometry"]].copy())
    send["geometry"] = send.geometry.simplify(0.000005)   # ~0.5 m, shrinks the request
    return [send.iloc[i:i + chunk_size] for i in range(0, len(send), chunk_size)]


def _key(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:20]


def run_batches(request, chunks, cols, *, workers=6, log=print, progress=None, label="values",
                cache_tag=None, tries=3):
    """Run ``request(fc_geojson_dict) -> ee object`` for every batch and combine the results.

    ``request`` must return something whose ``getInfo()`` is a FeatureCollection dict whose
    feature properties include ``cols``. Returns one DataFrame with columns ``cols``.
    """
    import json

    total = sum(len(c) for c in chunks)
    if total == 0:
        return pd.DataFrame(columns=cols)
    cdir = os.path.join(CACHE_DIR, _key(cache_tag)) if cache_tag else None
    if cdir:
        os.makedirs(cdir, exist_ok=True)
    lock = threading.Lock()
    state = {"done": 0, "failed": [], "cached": 0}

    def path_for(ch):
        return os.path.join(cdir, _key(*ch["bid"].tolist()) + ".csv") if cdir else None

    def once(ch):
        fc = json.loads(ch.to_json(drop_id=True))
        info = request(fc).getInfo()
        return pd.DataFrame([f["properties"] for f in info["features"]], columns=cols)

    def solve(ch):
        """Return results for ``ch``; split it if Earth Engine cannot handle it whole."""
        p = path_for(ch)
        if p and os.path.exists(p):
            with lock:
                state["cached"] += len(ch)
            return pd.read_csv(p, dtype={"bid": str})
        err = ""
        for attempt in range(tries + 2):
            try:
                df = once(ch)
                if p:
                    df.to_csv(p, index=False)
                return df
            except Exception as e:  # noqa: BLE001
                err = str(e)
                rate = any(w in err.lower() for w in RATE_WORDS)
                if not rate and attempt + 1 >= tries:
                    break                       # a real failure: try smaller pieces
                time.sleep(min(60, 5 * 2 ** attempt))
        if len(ch) > MIN_SPLIT:
            half = len(ch) // 2
            log(f"Earth Engine: a batch of {len(ch):,} {label} failed ({err[:90]}); splitting it in two")
            parts = [d for d in (solve(ch.iloc[:half]), solve(ch.iloc[half:])) if len(d)]
            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
        with lock:
            state["failed"].extend(ch["bid"].tolist())
        log(f"Earth Engine: {len(ch)} buildings could not be read ({err[:120]})")
        return pd.DataFrame(columns=cols)

    log(f"Earth Engine: {label} for {total:,} buildings in {len(chunks)} batches")
    out = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(solve, ch) for ch in chunks]
        for n, fut in enumerate(as_completed(futs), 1):
            out.append(fut.result())
            if progress:
                progress(n / len(chunks))
            if n % max(1, len(chunks) // 10) == 0 and len(chunks) >= 10:
                log(f"Earth Engine: {label}: {n}/{len(chunks)} batches done")
    if state["cached"]:
        log(f"Earth Engine: {state['cached']:,} buildings reused from an earlier, interrupted run")
    if state["failed"]:
        log(f"Earth Engine: {len(state['failed']):,} of {total:,} buildings got no {label} "
            "(re-run to retry only those)")
    out = [d for d in out if len(d)]
    if not out:
        return pd.DataFrame(columns=cols)
    res = pd.concat(out, ignore_index=True)
    res["bid"] = res["bid"].astype(str)
    return res.drop_duplicates("bid")
