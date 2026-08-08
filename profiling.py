"""
Where the time actually goes.

Reporting "N milliseconds per block" is not a latency analysis. It does not say
whether the number will hold at four times the volume, whether the bottleneck is
arithmetic or memory traffic, or how far the implementation sits from what the
hardware could do. This module answers those three questions.

Cost model
----------
For a Casorati matrix of m voxels by n frames, clutter rank k, and a voxel
subsampling fraction that leaves m' rows:

    Gram matrix     X'^H X'     ->  m' n^2   complex MACs
    eigendecomposition          ->  O(n^3)   negligible while n << m
    clutter projection          ->  2 m n k  complex MACs
                                    (X V_c is m n k, times V_c^H is m n k)

    memory traffic              ->  read m n, write m n, plus the m' n read
                                    for the Gram pass

A complex multiply-accumulate is 8 real flops. The arithmetic intensity -- flops
per byte moved -- decides whether the kernel is compute-bound or memory-bound,
and that in turn decides whether a faster algorithm or a better memory layout is
the thing to work on.

The measured conclusion for 4D blocks, spelled out in the README: once the
projection is written against the small clutter subspace rather than its
complement, the filter becomes **memory-bound**. Every voxel must be read and
written regardless of how clever the subspace estimate is, and that sets a hard
floor no amount of algorithmic work gets under.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

COMPLEX_MAC_FLOPS = 8.0  # 4 real multiplies + 4 real adds
BYTES_COMPLEX64 = 8


# --------------------------------------------------------------------------- #
# analytic cost
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    """Analytic flop and byte counts for one filtered block."""

    n_voxels: int
    n_frames: int
    cutoff: int
    subsample_rows: int | None = None

    @property
    def gram_rows(self) -> int:
        return self.n_voxels if self.subsample_rows is None else self.subsample_rows

    @property
    def gram_flops(self) -> float:
        return COMPLEX_MAC_FLOPS * self.gram_rows * self.n_frames**2

    @property
    def eigh_flops(self) -> float:
        # constant is loose; the point is that it vanishes while n << m
        return COMPLEX_MAC_FLOPS * 10.0 * self.n_frames**3

    @property
    def projection_flops(self) -> float:
        return COMPLEX_MAC_FLOPS * 2.0 * self.n_voxels * self.n_frames * self.cutoff

    @property
    def total_flops(self) -> float:
        return self.gram_flops + self.eigh_flops + self.projection_flops

    @property
    def bytes_moved(self) -> float:
        """Lower bound: read the block, write the result, read the Gram subset."""
        block = self.n_voxels * self.n_frames * BYTES_COMPLEX64
        subset = self.gram_rows * self.n_frames * BYTES_COMPLEX64
        return 2.0 * block + subset

    @property
    def arithmetic_intensity(self) -> float:
        """Flops per byte moved. Low means memory-bound."""
        return self.total_flops / self.bytes_moved

    def breakdown(self) -> dict:
        total = self.total_flops
        return {
            "gram_%": 100.0 * self.gram_flops / total,
            "eigh_%": 100.0 * self.eigh_flops / total,
            "projection_%": 100.0 * self.projection_flops / total,
            "gflops": total / 1e9,
            "mbytes": self.bytes_moved / 1e6,
            "intensity_flops_per_byte": self.arithmetic_intensity,
        }

    def roofline_ms(self, gflops_per_s: float, gbytes_per_s: float) -> dict:
        """Time predicted by each ceiling, and which one binds."""
        compute_ms = 1e3 * self.total_flops / (gflops_per_s * 1e9)
        memory_ms = 1e3 * self.bytes_moved / (gbytes_per_s * 1e9)
        return {
            "compute_bound_ms": compute_ms,
            "memory_bound_ms": memory_ms,
            "predicted_ms": max(compute_ms, memory_ms),
            "bound_by": "memory" if memory_ms > compute_ms else "compute",
        }


# --------------------------------------------------------------------------- #
# machine calibration
# --------------------------------------------------------------------------- #
def measure_gflops(size: int = 512, repeats: int = 3) -> float:
    """Sustained complex GEMM throughput, measured rather than assumed.

    Vendor peak numbers are useless for predicting this code, which runs through
    whatever BLAS numpy is linked against. Measure the machine you are on.
    """
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))).astype(
        np.complex64
    )
    b = a.copy()
    a @ b  # warm up

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        a @ b
        times.append(time.perf_counter() - t0)

    flops = COMPLEX_MAC_FLOPS * size**3
    return flops / (np.median(times) * 1e9)


def measure_bandwidth(mbytes: int = 256, repeats: int = 3) -> float:
    """Sustained memory bandwidth for a streaming copy, in GB/s."""
    n = int(mbytes * 1e6 / BYTES_COMPLEX64)
    a = np.ones(n, dtype=np.complex64)
    b = np.empty_like(a)
    np.copyto(b, a)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.copyto(b, a)
        times.append(time.perf_counter() - t0)

    moved = 2.0 * n * BYTES_COMPLEX64  # read + write
    return moved / (np.median(times) * 1e9)


def machine_profile() -> dict:
    """Calibrate the two roofline ceilings for this machine."""
    return {"gflops_per_s": measure_gflops(), "gbytes_per_s": measure_bandwidth()}


# --------------------------------------------------------------------------- #
# empirical scaling
# --------------------------------------------------------------------------- #
def time_filter(fn, block, repeats: int = 3) -> float:
    """Warm up once, then return the median wall time in milliseconds."""
    fn(block)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(block)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit ``y = c * x^p`` in log space. Returns ``(p, c)``.

    The exponent is the useful number: it says how the measurement will behave
    at sizes you have not run. If the fitted exponent disagrees with the cost
    model, one of the two is wrong and it is worth finding out which.
    """
    mask = (x > 0) & (y > 0)
    p, log_c = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(p), float(np.exp(log_c))


def scaling_experiment(
    filter_fn,
    make_block,
    voxel_counts: list[int],
    n_frames: int = 128,
    repeats: int = 2,
) -> dict:
    """Measure latency against voxel count and fit the exponent.

    ``make_block(n_voxels, n_frames)`` returns a synthetic array of that shape.
    Using random data rather than the full simulator keeps the experiment cheap;
    the runtime of a matrix product does not depend on its contents.
    """
    rows = []
    for m in voxel_counts:
        block = make_block(m, n_frames)
        ms = time_filter(filter_fn, block, repeats)
        rows.append({"n_voxels": m, "ms": ms, "mbytes": m * n_frames * BYTES_COMPLEX64 / 1e6})

    x = np.array([r["n_voxels"] for r in rows], dtype=float)
    y = np.array([r["ms"] for r in rows], dtype=float)
    exponent, coefficient = fit_power_law(x, y)

    return {"rows": rows, "exponent": exponent, "coefficient": coefficient}


def random_block(n_voxels: int, n_frames: int, seed: int = 0) -> np.ndarray:
    """A Casorati-shaped block of random complex data, for timing only."""
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_voxels, n_frames)) + 1j * rng.standard_normal((n_voxels, n_frames))
    ).astype(np.complex64)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def report(cfg_voxels: int, cfg_frames: int, cutoff: int, subsample_rows=None) -> None:
    machine = machine_profile()
    model = CostModel(cfg_voxels, cfg_frames, cutoff, subsample_rows)
    breakdown = model.breakdown()
    roofline = model.roofline_ms(machine["gflops_per_s"], machine["gbytes_per_s"])

    print(f"machine: {machine['gflops_per_s']:.1f} GFLOP/s complex GEMM, "
          f"{machine['gbytes_per_s']:.1f} GB/s streaming copy")
    print(f"block:   {cfg_voxels} voxels x {cfg_frames} frames, k={cutoff}"
          + (f", Gram on {subsample_rows} rows" if subsample_rows else ""))
    print(f"  work        {breakdown['gflops']:.2f} GFLOP, {breakdown['mbytes']:.0f} MB moved")
    print(f"  split       Gram {breakdown['gram_%']:.0f}%  "
          f"eigh {breakdown['eigh_%']:.1f}%  projection {breakdown['projection_%']:.0f}%")
    print(f"  intensity   {breakdown['intensity_flops_per_byte']:.2f} flops/byte")
    print(f"  roofline    compute {roofline['compute_bound_ms']:.1f} ms, "
          f"memory {roofline['memory_bound_ms']:.1f} ms -> bound by {roofline['bound_by']}")
