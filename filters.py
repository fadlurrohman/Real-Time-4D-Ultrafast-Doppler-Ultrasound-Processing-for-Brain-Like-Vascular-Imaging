"""
Clutter filters that do not care how many spatial dimensions there were.

Everything here operates on the Casorati matrix ``(n_voxels, nt)``. A 2D+time
plane and a 3D+time volume both flatten to that, so a single implementation
covers both -- the reshape is the whole of the "extension to 4D" as far as the
algorithms are concerned.

What is *not* free in 4D is cost. A full SVD of an ``m x n`` Casorati matrix is
O(m n^2), and m grows with the volume. Three routes to making that fit in a
real-time budget are implemented and measured:

``svd_filter``
    The baseline. Gram-matrix route (eigendecompose the n x n matrix ``X^H X``
    instead of running a full SVD of the tall one), which is exact and already
    a few times faster than ``np.linalg.svd``.

``subsampled_svd_filter``
    Estimate the temporal subspace from a random *subset* of voxels, then apply
    the projection to all of them. Cost drops from O(m n^2) to
    O(m' n^2 + m n k). This is the single biggest win available, and its
    validity is exactly the aspect-ratio condition that ``adaptive_cutoff``
    already checks: the temporal singular vectors are well estimated as long as
    the subset is comfortably taller than it is wide.

``randomized_svd_filter``
    Sketch the range with a Gaussian test matrix and work in the sketched basis.
    Useful when the clutter rank is small relative to nt.

``rpca``
    Principal component pursuit by inexact ALM. Higher quality in the sparse
    (contrast-enhanced) regime, an order of magnitude slower, and included as
    the upper bound a learned unrolling is trying to reach cheaply.
"""

from __future__ import annotations

import warnings

import numpy as np


# --------------------------------------------------------------------------- #
# reshaping
# --------------------------------------------------------------------------- #
def to_casorati(block: np.ndarray) -> np.ndarray:
    """``(*spatial, nt) -> (n_voxels, nt)``, for any number of spatial axes."""
    return block.reshape(-1, block.shape[-1])


def from_casorati(mat: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """``(n_voxels, nt) -> (*spatial, nt)``."""
    return mat.reshape(shape)


# --------------------------------------------------------------------------- #
# spectrum
# --------------------------------------------------------------------------- #
def gram_spectrum(X: np.ndarray):
    """Singular values and right singular vectors of a tall-skinny matrix.

    The Casorati matrix has n_voxels >> nt, so eigendecomposing the nt x nt Gram
    matrix ``X^H X`` is cheaper than a full SVD and never forms U at all.
    Numerically identical to machine precision.
    """
    gram = X.conj().T @ X
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    s = np.sqrt(np.clip(evals[order], 0.0, None))
    return s, evecs[:, order]


def project_out(X: np.ndarray, V: np.ndarray, cutoff: int, high: int | None = None):
    """Remove the span of the first ``cutoff`` temporal singular vectors.

    The obvious implementation keeps the complement, ``X V_k V_k^H`` with
    ``V_k = V[:, cutoff:]``. That is correct and it is a trap: ``V_k`` has
    ``nt - k`` columns, so the projection costs O(m n (n-k)), which for a
    clutter rank of 6 out of 128 is 20x more work than necessary and leaves the
    projection dominating the whole filter.

    Since V is unitary, ``V_c V_c^H + V_k V_k^H = I``, so the same result comes
    from subtracting the *clutter* projection:

        X_filtered = X - (X V_c) V_c^H ,   V_c = V[:, :cutoff]

    which costs O(m n k) with k the clutter rank. Measured on a 40^3 x 128
    volume this alone took the exact filter from 493 ms to well under half that,
    and it is what lets voxel subsampling actually pay off -- before the fix,
    subsampling only shrank a term that was not the bottleneck.
    """
    if high is not None:
        keep = V[:, cutoff:high]
        return (X @ keep) @ keep.conj().T

    clutter = V[:, :cutoff]
    return X - (X @ clutter) @ clutter.conj().T


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #
def svd_filter(block: np.ndarray, cutoff: int = 8, high_cutoff: int | None = None):
    """Exact subspace filter via the Gram route."""
    shape = block.shape
    X = to_casorati(block).astype(np.complex64)
    _, V = gram_spectrum(X.astype(np.complex128))
    filtered = project_out(X, V.astype(np.complex64), cutoff, high_cutoff)
    return from_casorati(filtered, shape)


# --------------------------------------------------------------------------- #
# the two accelerations that matter in 4D
# --------------------------------------------------------------------------- #
def subsampled_svd_filter(
    block: np.ndarray,
    cutoff: int = 8,
    fraction: float = 0.05,
    min_rows: int | None = None,
    seed: int = 0,
    return_subset_size: bool = False,
):
    """Estimate the temporal subspace from a random subset of voxels.

    The temporal singular vectors describe how the *clutter evolves in time*, and
    that is a property of the tissue motion, not of which voxels you look at.
    Sampling 5% of a volume still leaves tens of thousands of rows against a few
    hundred columns, which is far more than enough to estimate an nt x nt Gram
    matrix.

    Cost: O(m' n^2) to build the subspace plus O(m n k) to apply it, against
    O(m n^2) for the full solve. Measured on a 40^3 x 128 volume this is where
    most of the speedup in ``benchmark.py`` comes from.

    ``min_rows`` guards the estimate at ``10 * nt`` rows, the same 10:1
    aspect-ratio condition ``adaptive_cutoff`` enforces.

    That floor is necessary but **not sufficient, and it should depend on the
    clutter rank**. Estimating a 16-dimensional subspace needs far more samples
    than a 6-dimensional one, and the fixed floor does not know that. Measured
    on a 28^3 volume, relative error against the exact solve:

        rank k=6            k=16
        ratio  10   0.22    0.61
        ratio  86   0.05    0.19

    At k=16 a 5% subsample costs over a dB of CNR, where at k=6 it cost
    essentially nothing. The floor is left rank-independent here because
    choosing the right scaling is a question this study did not answer -- but
    the dependence is real, and ``fraction`` should be raised when the clutter
    rank is high rather than trusted at its default.
    """
    shape = block.shape
    X = to_casorati(block).astype(np.complex64)
    m, n = X.shape

    floor = 10 * n if min_rows is None else min_rows
    n_rows = int(min(m, max(floor, round(fraction * m))))

    if n_rows >= m:
        result = svd_filter(block, cutoff)
        return (result, m) if return_subset_size else result

    rng = np.random.default_rng(seed)
    rows = rng.choice(m, size=n_rows, replace=False)
    _, V = gram_spectrum(X[rows].astype(np.complex128))

    filtered = project_out(X, V.astype(np.complex64), cutoff)
    out = from_casorati(filtered, shape)
    return (out, n_rows) if return_subset_size else out


def randomized_svd_filter(
    block: np.ndarray, cutoff: int = 8, sketch: int = 48, n_power: int = 1, seed: int = 0
):
    """Sketch the range with a Gaussian test matrix, then filter in that basis.

    Useful when the clutter rank is small relative to nt: the Gram matrix drops
    from nt x nt to ``sketch x sketch``. ``n_power`` subspace iterations sharpen
    the separation when the spectrum decays slowly, at one extra pass each.
    """
    shape = block.shape
    X = to_casorati(block).astype(np.complex64)
    _, n = X.shape
    sketch = int(min(sketch, n))

    rng = np.random.default_rng(seed)
    omega = (rng.standard_normal((n, sketch)) + 1j * rng.standard_normal((n, sketch))).astype(
        np.complex64
    ) / np.sqrt(2.0)

    Y = X @ omega
    for _ in range(n_power):
        Y = X @ (X.conj().T @ Y)
    Q, _ = np.linalg.qr(Y)

    small = Q.conj().T @ X  # (sketch, n)
    _, _, Vh = np.linalg.svd(small, full_matrices=False)
    V = Vh.conj().T.astype(np.complex64)

    filtered = project_out(X, V, cutoff)
    return from_casorati(filtered, shape)


# --------------------------------------------------------------------------- #
# automatic cutoff
# --------------------------------------------------------------------------- #
def singular_doppler_frequencies(block: np.ndarray, prf: float, window: bool = True):
    """Mean absolute Doppler frequency carried by each temporal singular vector.

    The window is not optional. With a rectangular window the sidelobes of the
    very strong DC component leak across the whole band, and a power-weighted
    mean is dominated by those tails -- on short ensembles that inverted the
    curve outright, with leading tissue components reading higher than later
    blood ones.
    """
    X = to_casorati(block).astype(np.complex128)
    _, V = gram_spectrum(X)

    nt = V.shape[0]
    if window:
        V = V * np.hanning(nt)[:, None]

    freqs = np.fft.fftfreq(nt, d=1.0 / prf)
    spec = np.abs(np.fft.fft(V, axis=0)) ** 2
    weights = spec / (spec.sum(axis=0, keepdims=True) + 1e-20)
    return (weights * np.abs(freqs)[:, None]).sum(axis=0)


def adaptive_cutoff(
    block: np.ndarray,
    prf: float,
    alpha: float = 0.25,
    max_cutoff: int = 60,
    fallback: int = 6,
) -> int:
    """Pick the clutter order from the Doppler-frequency curve.

    There are three subspaces, not two: tissue near DC, blood at intermediate
    frequencies, and the noise floor highest of all -- a white sequence has a
    flat spectrum, so its power-weighted mean ``|f|`` is ``PRF/4``. A rule based
    on the largest jump in the curve finds the *blood-to-noise* transition and
    returns a cutoff several times too high. Thresholding at a fraction of
    ``PRF/4`` instead is scale-free and needs no per-acquisition tuning.

    Two guards, both of which correspond to measured silent failures:
    ensembles too short for the frequency resolution, and Casorati matrices too
    squat for the temporal singular vectors to be estimated at all.
    """
    nt = block.shape[-1]
    n_voxels = int(np.prod(block.shape[:-1]))
    cut_hz = alpha * prf / 4.0

    if n_voxels < 10 * nt:
        warnings.warn(
            f"Casorati matrix is {n_voxels}x{nt} (ratio {n_voxels / nt:.1f}); the "
            f"temporal singular vectors are unreliable below 10. Falling back to k={fallback}",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback

    if 8.0 * prf / nt > cut_hz:
        warnings.warn(
            f"ensemble of {nt} frames at PRF={prf:.0f} Hz is too short for reliable "
            f"cutoff selection (need about {int(np.ceil(32.0 / alpha))}); "
            f"falling back to k={fallback}",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback

    fmean = singular_doppler_frequencies(block, prf)[: max_cutoff + 5]

    if fmean[0] > cut_hz:
        warnings.warn(
            f"leading singular vector already exceeds the Doppler cut; "
            f"falling back to k={fallback}",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback

    above = np.nonzero(fmean > cut_hz)[0]
    return int(min(max(above[0], 1), max_cutoff)) if above.size else 1


# --------------------------------------------------------------------------- #
# proximal operators and robust PCA
# --------------------------------------------------------------------------- #
def soft_threshold(z: np.ndarray, tau: float) -> np.ndarray:
    """Complex soft threshold: shrink magnitude, keep phase."""
    return z * np.maximum(0.0, 1.0 - tau / (np.abs(z) + 1e-20))


def singular_value_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """Nuclear-norm prox, without ever forming U."""
    s, V = gram_spectrum(X)
    shrink = np.where(s > 1e-12, np.maximum(s - tau, 0.0) / np.maximum(s, 1e-12), 0.0)
    return X @ (V * shrink) @ V.conj().T


def rpca(block: np.ndarray, lam=None, n_iter: int = 40, rho: float = 1.5, tol: float = 1e-7):
    """Principal component pursuit by inexact ALM (Lin, Chen & Ma, 2010).

        min ||L||_* + lam ||S||_1     s.t.  L + S = X

    Worth recording: the *unconstrained* least-squares variant
    ``min 0.5||L+S-X||^2 + l1||L||_* + l2||S||_1`` is degenerate for this
    problem. With a small l2 the minimiser drives L to zero and dumps everything
    into S, so no separation happens at all -- measured clutter rejection came
    out worse than not filtering. The constrained form works immediately. The
    formulation was the bug, not the solver.
    """
    shape = block.shape
    X = to_casorati(block).astype(np.complex128)
    m, n = X.shape

    scale = np.linalg.norm(X, 2)
    Xn = X / (scale + 1e-20)

    lam = 1.0 / np.sqrt(max(m, n)) if lam is None else lam
    mu, mu_max = 1.25, 1e7

    L = np.zeros_like(Xn)
    S = np.zeros_like(Xn)
    Y = np.zeros_like(Xn)
    norm_X = np.linalg.norm(Xn, "fro") + 1e-20

    for _ in range(n_iter):
        L = singular_value_threshold(Xn - S + Y / mu, 1.0 / mu)
        S = soft_threshold(Xn - L + Y / mu, lam / mu)
        resid = Xn - L - S
        Y = Y + mu * resid
        mu = min(rho * mu, mu_max)
        if np.linalg.norm(resid, "fro") / norm_X < tol:
            break

    return from_casorati(S * scale, shape), from_casorati(L * scale, shape)


def highpass_filter(block: np.ndarray, prf: float, cutoff_hz: float = 40.0):
    """Per-voxel temporal high-pass -- the pre-SVD wall filter, as a lower bound."""
    nt = block.shape[-1]
    freqs = np.fft.fftfreq(nt, d=1.0 / prf)
    keep = (np.abs(freqs) >= cutoff_hz).astype(np.float32)
    return np.fft.ifft(np.fft.fft(block, axis=-1) * keep, axis=-1).astype(block.dtype)
