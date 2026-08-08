"""
Image-quality metrics, written once for any number of spatial axes.

Two families, deliberately kept apart.

*In silico* -- ground truth exists because the simulator renders blood and
tissue separately, so fidelity is measurable: scale-invariant NMSE, correlation
of the power Doppler maps, clutter rejection.

*In vivo* -- no ground truth, so quality must be read off the image itself:
contrast-to-noise between vessel and background. This is the one that transfers
to real recordings, and the one to report when benchmarking on PALA.
"""

from __future__ import annotations

import numpy as np


def power_doppler(block: np.ndarray) -> np.ndarray:
    """Mean power over slow time. ``(*spatial, nt) -> (*spatial)``."""
    return np.mean(np.abs(block) ** 2, axis=-1)


def to_db(img: np.ndarray, dynamic_range: float = 40.0) -> np.ndarray:
    """Log-compress and clip to a fixed range below the peak."""
    img = np.asarray(img, dtype=np.float64)
    db = 10.0 * np.log10(img / (img.max() + 1e-20) + 1e-20)
    return np.clip(db, -dynamic_range, 0.0)


def cnr_db(pd_map: np.ndarray, vessel_mask: np.ndarray, background_mask=None) -> float:
    """Vessel-to-background contrast-to-noise ratio, in dB.

    Read before ranking methods by this number: CNR measures background
    suppression, not fidelity, and it is **not bounded above by the ground
    truth**. A filter that projects onto a low-dimensional subspace also shrinks
    the variance inside the vessel region, which raises CNR even where the
    estimate is less faithful. In the microbubble regime that puts SVD above the
    true blood signal itself. For sparse blood, rank by :func:`pd_correlation`.
    """
    vessel = vessel_mask > 0.5
    background = vessel_mask < 0.05 if background_mask is None else background_mask > 0.5

    if vessel.sum() < 10 or background.sum() < 10:
        return float("nan")

    mu_v, mu_b = pd_map[vessel].mean(), pd_map[background].mean()
    denom = np.sqrt(0.5 * (pd_map[vessel].var() + pd_map[background].var())) + 1e-20
    return float(20.0 * np.log10(abs(mu_v - mu_b) / denom + 1e-20))


def nmse_db(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Complex NMSE in dB, after fitting the optimal global complex gain.

    A clutter filter is only defined up to gain, so penalising an amplitude
    mismatch would be measuring the wrong thing.
    """
    e = estimate.ravel()
    t = truth.ravel()
    alpha = np.vdot(e, t) / (np.vdot(e, e) + 1e-20)
    resid = alpha * e - t
    return float(
        10.0 * np.log10((np.abs(resid) ** 2).sum() / ((np.abs(t) ** 2).sum() + 1e-20) + 1e-30)
    )


def pd_correlation(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Pearson correlation of the two power Doppler maps in dB scale.

    dB rather than linear: vessel visibility is a log-domain judgement, and the
    linear correlation is dominated by the few brightest voxels.
    """
    a = to_db(power_doppler(estimate)).ravel()
    b = to_db(power_doppler(truth)).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


def clutter_rejection_db(estimate, tissue, blood) -> float:
    """Energy kept along the blood direction against energy kept along tissue."""

    def energy_along(x, ref):
        r = ref.ravel()
        coef = np.vdot(r, x.ravel()) / (np.vdot(r, r) + 1e-20)
        return float(np.abs(coef) ** 2 * (np.abs(r) ** 2).sum())

    return float(
        10.0
        * np.log10(
            (energy_along(estimate, blood) + 1e-20) / (energy_along(estimate, tissue) + 1e-20)
        )
    )


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Relative Frobenius error against another *estimate*, not against truth.

    Used to price the approximate filters: how much accuracy does subsampling
    give up relative to the exact solve, independently of how good the exact
    solve was.
    """
    return float(
        np.linalg.norm((estimate - reference).ravel())
        / (np.linalg.norm(reference.ravel()) + 1e-20)
    )


def summarise(estimate, truth_blood=None, truth_tissue=None, mask=None, contrast=False) -> dict:
    out = {}
    if mask is not None:
        out["cnr_db"] = cnr_db(power_doppler(estimate), mask)
    if truth_blood is not None:
        out["nmse_db"] = nmse_db(estimate, truth_blood)
        out["pd_corr"] = pd_correlation(estimate, truth_blood)
        if truth_tissue is not None:
            out["clutter_rejection_db"] = clutter_rejection_db(estimate, truth_tissue, truth_blood)
    out["primary"] = "pd_corr" if contrast else "cnr_db"
    return out
