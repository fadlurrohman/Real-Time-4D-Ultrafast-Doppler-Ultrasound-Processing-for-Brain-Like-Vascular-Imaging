"""
The table this project exists to produce.

Quality, latency, and whether the deadline is met -- for the same code on a
2D+time plane and a 3D+time volume.

    python -m src.benchmark                   # 2D+t
    python -m src.benchmark --volume          # 3D+t (4D)
    python -m src.benchmark --volume --roofline
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from . import filters as F
from . import metrics as M
from . import profiling as P
from . import streaming as S
from .simulate import plane_config, simulate_block, volume_config


def timed(fn, repeats: int = 3):
    """Warm up once, then median wall time in milliseconds.

    Each call allocates a fresh output block, so on a memory-constrained machine
    the timings drift upward as the benchmark proceeds unless the garbage is
    collected between methods. Measured consequence of not doing this: a
    subsampled filter that takes 124 ms in isolation reported 635 ms when run
    eighth in a sequence -- a 5x error caused entirely by allocator pressure,
    and exactly the sort of artefact that makes a benchmark table wrong without
    looking wrong.
    """
    gc.collect()
    out = fn()
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return out, float(np.median(times))


def run(args):
    cfg = (
        volume_config(args.side, args.side, args.side, args.nt, contrast=args.contrast)
        if args.volume
        else plane_config(96, 128, args.nt, contrast=args.contrast)
    )
    d = simulate_block(cfg, seed=args.seed)
    iq, blood, tissue, mask = d["iq"], d["blood"], d["tissue"], d["mask"]

    k = F.adaptive_cutoff(iq, cfg.prf)
    deadline_ms = 1000.0 * cfg.nt / cfg.prf

    print(
        f"block {iq.shape}  ({cfg.ndim}D+t)  "
        f"{cfg.block_bytes / 1e6:.0f} MB  Casorati {cfg.n_voxels}x{cfg.nt}  k={k}"
    )
    print(f"acquisition deadline: {deadline_ms:.0f} ms per block at PRF {cfg.prf:.0f} Hz\n")

    reference = F.svd_filter(iq, k)

    methods = [
        ("unfiltered", lambda: iq),
        ("highpass 40 Hz", lambda: F.highpass_filter(iq, cfg.prf, 40.0)),
        ("exact SVD", lambda: F.svd_filter(iq, k)),
        ("subsample 20%", lambda: F.subsampled_svd_filter(iq, k, fraction=0.20)),
        ("subsample 5%", lambda: F.subsampled_svd_filter(iq, k, fraction=0.05)),
        ("randomized sketch", lambda: F.randomized_svd_filter(iq, k, sketch=32)),
        ("streaming (deployable)", lambda: S.streaming_filter(iq, k, 0.05, args.chunk)),
    ]
    if not args.skip_rpca:
        methods.append(("RPCA ALM 40 it", lambda: F.rpca(iq, n_iter=40)[0]))
    methods.append(("oracle", lambda: blood))

    header = f"{'method':<24}{'CNR dB':>9}{'PD corr':>9}{'rel.err':>9}{'ms':>9}{'deadline':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, fn in methods:
        est, ms = timed(fn, args.repeats)
        summary = M.summarise(est, blood, tissue, mask, contrast=args.contrast)
        rel = M.relative_error(est, reference) if name != "unfiltered" else float("nan")
        verdict = "-" if name in ("unfiltered", "oracle") else ("ok" if ms <= deadline_ms else "MISS")
        print(
            f"{name:<24}{summary.get('cnr_db', float('nan')):>9.2f}"
            f"{summary.get('pd_corr', float('nan')):>9.3f}{rel:>9.4f}{ms:>9.1f}{verdict:>10}"
        )
        rows.append({"method": name, "ms": ms, "rel_err": rel, **summary})

    streaming_ms = next(r["ms"] for r in rows if r["method"] == "streaming (deployable)")
    budget = S.RealTimeBudget(cfg.nt, cfg.prf, streaming_ms)
    print(f"\ndeployable path: {budget.describe()}")
    print(f"  sustainable rate {budget.sustainable_rate_hz:.1f} blocks/s")

    per_unit = streaming_ms / (cfg.n_voxels * cfg.nt)
    side = S.largest_volume_meeting_deadline(per_unit, cfg.nt, cfg.prf)
    print(
        f"  extrapolated ceiling on this machine: {side}^3 volume at {cfg.nt} frames "
        f"(1 CPU core)"
    )

    if args.roofline:
        print()
        P.report(cfg.n_voxels, cfg.nt, k, subsample_rows=int(0.05 * cfg.n_voxels))

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume", action="store_true", help="3D+t instead of 2D+t")
    p.add_argument("--side", type=int, default=40, help="cubic volume side")
    p.add_argument("--nt", type=int, default=128)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--contrast", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--roofline", action="store_true")
    p.add_argument("--skip-rpca", action="store_true", help="RPCA is slow on 4D blocks")
    run(p.parse_args())


if __name__ == "__main__":
    main()
