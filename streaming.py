"""
Running the filter inside the acquisition loop.

An offline filter can hold the whole block, take as long as it likes, and be
judged on image quality alone. A filter inside the acquisition loop cannot. It
has a deadline set by the acquisition itself, and a memory ceiling set by the
machine. This module makes both explicit and measures whether they are met.

The deadline
------------
A Doppler ensemble of ``nt`` frames at rate ``PRF`` takes ``nt / PRF`` seconds to
acquire. If the filter takes longer than that, blocks arrive faster than they
are consumed and the pipeline falls behind without bound. So

    deadline_ms = 1000 * nt / PRF

For 128 frames at 1 kHz that is 128 ms, and it is a hard number rather than a
preference.

Why streaming is possible at all
--------------------------------
Once the temporal subspace ``V_c`` is known, the projection

    X_filtered = X - (X V_c) V_c^H

is **row-independent**: each voxel's time series is filtered using only itself
and the shared ``V_c``. So the block never has to be resident in full. The
subspace comes from a small subsample, and the projection then streams over
spatial chunks with a working set of ``chunk_voxels x nt`` instead of
``n_voxels x nt``.

That is the property that makes 4D deployable, and it follows directly from
writing the projection against the clutter subspace rather than its complement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import gram_spectrum

BYTES_COMPLEX64 = 8


# --------------------------------------------------------------------------- #
# the budget
# --------------------------------------------------------------------------- #
@dataclass
class RealTimeBudget:
    """Deadline arithmetic for one Doppler block."""

    n_frames: int
    prf: float
    measured_ms: float

    @property
    def deadline_ms(self) -> float:
        return 1000.0 * self.n_frames / self.prf

    @property
    def headroom(self) -> float:
        """Deadline divided by measured time. Above 1.0 means it keeps up."""
        return self.deadline_ms / max(self.measured_ms, 1e-9)

    @property
    def meets_deadline(self) -> bool:
        return self.measured_ms <= self.deadline_ms

    @property
    def sustainable_rate_hz(self) -> float:
        """Blocks per second this implementation can actually consume."""
        return 1000.0 / max(self.measured_ms, 1e-9)

    def describe(self) -> str:
        verdict = "MEETS" if self.meets_deadline else "MISSES"
        return (
            f"{verdict} deadline: {self.measured_ms:.0f} ms measured against "
            f"{self.deadline_ms:.0f} ms available ({self.headroom:.2f}x headroom)"
        )


def largest_volume_meeting_deadline(
    ms_per_voxel_frame: float,
    n_frames: int,
    prf: float,
    max_side: int = 256,
) -> int:
    """Largest cubic volume side whose filter fits the deadline.

    Uses the measured per-(voxel x frame) cost, which is the honest way to
    extrapolate once the kernel is memory-bound and therefore close to linear in
    block size.
    """
    deadline_ms = 1000.0 * n_frames / prf
    max_voxels = deadline_ms / (ms_per_voxel_frame * n_frames)
    side = int(np.floor(max_voxels ** (1.0 / 3.0)))
    return int(min(max(side, 0), max_side))


def working_set_bytes(chunk_voxels: int, n_frames: int, n_buffers: int = 3) -> int:
    """Resident memory for the streaming path.

    ``n_buffers`` counts the chunk being read, the intermediate ``X V_c``, and
    the output chunk. The full block never has to be resident, which is the
    entire point.
    """
    return int(n_buffers * chunk_voxels * n_frames * BYTES_COMPLEX64)


# --------------------------------------------------------------------------- #
# the streaming filter
# --------------------------------------------------------------------------- #
def estimate_subspace(
    block: np.ndarray,
    cutoff: int,
    fraction: float = 0.05,
    min_rows: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Estimate the clutter subspace from a random subset of voxels.

    Returns ``(V_c, n_rows_used)`` with ``V_c`` of shape ``(nt, cutoff)``.

    The subset is never allowed below ``10 * nt`` rows: the Gram matrix needs to
    be estimated from a matrix comfortably taller than it is wide, and below
    that ratio the temporal singular vectors stop being reliable. Measured
    consequence of ignoring this, on a squat block: a cutoff of 4 where the
    optimum was 21, a 25 dB CNR loss, and no error raised anywhere.
    """
    X = block.reshape(-1, block.shape[-1])
    m, n = X.shape

    floor = 10 * n if min_rows is None else min_rows
    n_rows = int(min(m, max(floor, round(fraction * m))))

    if n_rows >= m:
        _, V = gram_spectrum(X.astype(np.complex128))
        return V[:, :cutoff].astype(np.complex64), m

    rng = np.random.default_rng(seed)
    rows = rng.choice(m, size=n_rows, replace=False)
    _, V = gram_spectrum(X[rows].astype(np.complex128))
    return V[:, :cutoff].astype(np.complex64), n_rows


def apply_projection_streaming(
    block: np.ndarray,
    clutter_basis: np.ndarray,
    chunk_voxels: int = 8192,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Subtract the clutter projection, one spatial chunk at a time.

    Peak extra memory is ``chunk_voxels x nt`` regardless of how large the block
    is. In a real pipeline the chunks would be arriving from the beamformer
    rather than sliced out of a resident array, and nothing about this loop would
    change.
    """
    shape = block.shape
    X = block.reshape(-1, shape[-1])
    m = X.shape[0]

    result = np.empty_like(X) if out is None else out.reshape(-1, shape[-1])

    for start in range(0, m, chunk_voxels):
        stop = min(start + chunk_voxels, m)
        chunk = X[start:stop]
        coeffs = chunk @ clutter_basis  # (chunk, k)
        result[start:stop] = chunk - coeffs @ clutter_basis.conj().T

    return result.reshape(shape)


def streaming_filter(
    block: np.ndarray,
    cutoff: int,
    fraction: float = 0.05,
    chunk_voxels: int = 8192,
    seed: int = 0,
) -> np.ndarray:
    """The full deployable path: subsampled subspace, then chunked projection."""
    basis, _ = estimate_subspace(block, cutoff, fraction, seed=seed)
    return apply_projection_streaming(block, basis, chunk_voxels)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def chunk_sweep(
    block: np.ndarray, cutoff: int, chunk_sizes, fraction: float = 0.05, repeats: int = 5
):
    """Latency and resident memory against chunk size.

    Measured shape, which is not the one I expected: there is no sharp interior
    optimum. The curve is **flat across two orders of magnitude** (64 to ~4096
    voxels per chunk, all within about 10% of each other) and then degrades once
    the working set stops fitting in cache -- roughly 1.4x slower at 16k voxels
    and 2x at a single full-block chunk.

    The practical consequence is a good one: chunk size is not a parameter that
    needs careful tuning. Anything small enough to stay in cache works, so it can
    be chosen to suit the memory ceiling rather than to chase speed.

    Single-core timings are noisy enough that an occasional point comes back 3x
    high from scheduling interference. ``repeats`` defaults to 5 and the median
    is reported for that reason; treat any single outlier with suspicion rather
    than as structure.
    """
    import time

    n_frames = block.shape[-1]
    basis, n_rows = estimate_subspace(block, cutoff, fraction)
    out = np.empty_like(block)

    rows = []
    for chunk in chunk_sizes:
        apply_projection_streaming(block, basis, chunk, out)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            apply_projection_streaming(block, basis, chunk, out)
            times.append((time.perf_counter() - t0) * 1e3)
        rows.append(
            {
                "chunk_voxels": chunk,
                "projection_ms": float(np.median(times)),
                "working_set_mb": working_set_bytes(chunk, n_frames) / 1e6,
            }
        )
    return {"rows": rows, "subspace_rows": n_rows}
