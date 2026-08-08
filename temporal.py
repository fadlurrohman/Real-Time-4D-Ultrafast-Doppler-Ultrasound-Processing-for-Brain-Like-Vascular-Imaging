"""
How long must you wait before a small vessel becomes visible?

The clutter filter turns one Doppler ensemble into one power Doppler volume. A
single volume is noisy, and the smallest vessels sit inside that noise. The
standard answer is to accumulate: average several consecutive blocks before
deciding what is vessel and what is not.

That turns a purely spatial question into a temporal one, and it has a cost that
matters intraoperatively. Each block takes ``nt / PRF`` seconds to acquire, so
accumulating ``n`` blocks means the surgeon waits ``n * nt / PRF`` seconds for
the map. **Time-to-detection**, not latency-per-block, is the number that decides
whether a method is usable in theatre.

The experiment here measures that directly: segmentation recall as a function of
how many blocks were accumulated, split by vessel calibre. The expected and
measured result is that thick and thin vessels have very different answers, so a
single "how many blocks do we need" number is the wrong question.

Three things this module adds that the static pipeline could not express:

*Temporal analysis* -- recall against accumulation depth, per radius band.
*Temporal weighting* -- a learned alternative to uniform averaging.
*Real-time budget in the time domain* -- seconds to a usable map, not ms per block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .filters import svd_filter
from .metrics import power_doppler
from .simulate import SimConfig, simulate_block, volume_config


# --------------------------------------------------------------------------- #
# a sequence of blocks over the same vasculature
# --------------------------------------------------------------------------- #
def simulate_sequence(cfg: SimConfig, n_blocks: int, seed: int = 0) -> dict:
    """Consecutive Doppler ensembles imaging one fixed vascular tree.

    The vessel geometry is generated once and reused, so the tree is identical
    across blocks while speckle, tissue motion and noise are redrawn. That is
    what makes averaging meaningful: the signal is common and the noise is not.

    Returns ``power`` of shape ``(n_blocks, *spatial)`` plus the ground-truth
    lumen mask and a per-voxel radius map.
    """
    base = simulate_block(cfg, seed=seed)
    vessels = base["vessels"]

    fixed = SimConfig(**{**cfg.__dict__, "vessels": vessels})
    rng = np.random.default_rng(seed + 1)

    volumes = []
    for _ in range(n_blocks):
        d = simulate_block(fixed, seed=int(rng.integers(0, 2**31 - 1)))
        volumes.append(power_doppler(svd_filter(d["iq"], cutoff=16)))

    return {
        "power": np.stack(volumes),
        "mask": base["mask"],
        "radius": radius_map(fixed, vessels),
        "cfg": fixed,
    }


def radius_map(cfg: SimConfig, vessels) -> np.ndarray:
    """Per-voxel radius of the vessel it belongs to, for stratified scoring.

    Without this every recall number is a single average dominated by the
    trunks, which is exactly the failure the segmentation project documents.
    """
    from .simulate import vessel_masks

    masks = vessel_masks(cfg, vessels)
    radii = np.array([v.radius for v in vessels], dtype=np.float32)
    out = np.zeros(cfg.spatial_shape, dtype=np.float32)
    for m, r in zip(masks, radii):
        np.maximum(out, np.where(m > 0.5, r, 0.0), out=out)
    return out


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #
def segment(accumulated: np.ndarray, foreground_fraction: float) -> np.ndarray:
    """Threshold so the predicted foreground matches the true foreground size.

    Deliberately the simplest possible segmenter: the question is what
    *accumulation* buys, and a learned segmenter would confound that with
    whatever the network happened to learn.

    The threshold is calibrated to the true vessel fraction rather than fixed at
    a round number, and getting this wrong cost an afternoon. A fixed 96th
    percentile predicts 4% of voxels; the phantom's vessels occupy 9%. Recall was
    therefore capped at 0.45 **by construction**, no accumulation depth could
    ever pass it, and thick vessels scored *below* medium ones because the few
    predicted voxels were spread across a larger true region. Every number in the
    first run of this experiment was an artefact of that mismatch rather than a
    statement about accumulation.

    Matching the predicted count to the true count makes recall equal precision,
    so the only thing that varies with depth is whether averaging put the right
    voxels at the top of the ranking -- which is exactly the question.
    """
    cut = 100.0 * (1.0 - foreground_fraction)
    return accumulated > np.percentile(accumulated, cut)


def recall_by_radius(pred: np.ndarray, mask: np.ndarray, radius: np.ndarray, bands) -> dict:
    """Recall within each vessel-calibre band."""
    truth = mask > 0.5
    out = {}
    for lo, hi in bands:
        band = truth & (radius > lo) & (radius <= hi)
        n = int(band.sum())
        key = f"r_{lo:g}_{hi:g}" if np.isfinite(hi) else f"r_{lo:g}_inf"
        out[key] = float((pred & band).sum() / n) if n else float("nan")
    return out


DEFAULT_BANDS = ((0.0, 1.5), (1.5, 2.5), (2.5, np.inf))


# --------------------------------------------------------------------------- #
# the accumulation experiment
# --------------------------------------------------------------------------- #
@dataclass
class TemporalBudget:
    """Seconds of acquisition behind an accumulated map."""

    n_blocks: int
    n_frames: int
    prf: float

    @property
    def seconds(self) -> float:
        return self.n_blocks * self.n_frames / self.prf


def accumulation_experiment(sequence: dict, depths=(1, 2, 4, 8, 16), bands=DEFAULT_BANDS):
    """Recall against accumulation depth, per calibre, with the time cost attached."""
    power = sequence["power"]
    mask, radius, cfg = sequence["mask"], sequence["radius"], sequence["cfg"]

    fg = float((mask > 0.5).mean())

    rows = []
    for n in depths:
        if n > len(power):
            continue
        accumulated = power[:n].mean(axis=0)
        pred = segment(accumulated, fg)
        budget = TemporalBudget(n, cfg.nt, cfg.prf)
        rows.append(
            {
                "n_blocks": n,
                "seconds": budget.seconds,
                **recall_by_radius(pred, mask, radius, bands),
            }
        )
    return rows


def blocks_to_reach(rows, band: str, target: float = 0.8) -> int | None:
    """Smallest accumulation depth reaching ``target`` recall in a band."""
    for row in rows:
        value = row.get(band, float("nan"))
        if np.isfinite(value) and value >= target:
            return int(row["n_blocks"])
    return None


# --------------------------------------------------------------------------- #
# learned temporal weighting
# --------------------------------------------------------------------------- #
def weighted_accumulate(power: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean over blocks. Weights are softmaxed so they stay positive."""
    w = np.exp(weights - weights.max())
    w = w / w.sum()
    return np.tensordot(w, power, axes=(0, 0))


def train_temporal_weights(
    sequences: list[dict], n_blocks: int = 8, max_evals: int = 300, verbose: bool = True
) -> dict:
    """Learn how to weight blocks instead of averaging them uniformly.

    Uniform averaging is optimal only when every block carries the same noise
    level. Real acquisitions drift -- probe pressure, tissue motion, contrast
    washout -- so the optimal weighting is not flat. This is the smallest honest
    version of "temporal deep learning": ``n_blocks`` free parameters, trained by
    direct search, initialised at the uniform weighting so it can only improve on
    it or fail to.

    The objective is thin-vessel recall, because that is the band accumulation is
    supposed to rescue and the band a global score hides.

    Measured outcome: **it does not generalise, and it cannot.**
    Fitting gains 4.5 points of thin-vessel recall on the sequences it was fitted
    to (0.682 -> 0.727) and exactly 0.000 on three held-out sequences. The reason
    is structural rather than a training failure: this simulator draws every
    block from the same distribution, with identical signal and independent
    identically-distributed noise, so the blocks are *exchangeable* and uniform
    weighting is optimal in expectation. There is nothing for the weights to
    learn, and what they fitted was the noise realisation of two specific
    sequences.

    Learned temporal weighting only earns its place when blocks are **not**
    exchangeable -- probe drift, changing tissue motion, contrast washout,
    progressive decorrelation. Adding that non-stationarity to the simulator is
    the experiment this result points at, and it has not been done here.
    """

    def loss(weights):
        total = 0.0
        for seq in sequences:
            accumulated = weighted_accumulate(seq["power"][:n_blocks], weights)
            pred = segment(accumulated, float((seq["mask"] > 0.5).mean()))
            r = recall_by_radius(pred, seq["mask"], seq["radius"], DEFAULT_BANDS)
            total += 1.0 - np.nan_to_num(r["r_0_1.5"])
        return total / len(sequences)

    x0 = np.zeros(n_blocks)
    start = loss(x0)
    if verbose:
        print(f"  uniform weighting: thin-vessel recall {1 - start:.4f}", flush=True)

    result = minimize(
        loss, x0, method="Powell", options={"maxfev": max_evals, "xtol": 1e-2, "disp": False}
    )

    if verbose:
        print(f"  learned weighting: thin-vessel recall {1 - result.fun:.4f}", flush=True)

    return {
        "weights": np.exp(result.x - result.x.max()) / np.exp(result.x - result.x.max()).sum(),
        "uniform_recall": float(1 - start),
        "learned_recall": float(1 - result.fun),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def main():
    # SNR matters more than anything else here. At snr_db = 0 a single block is
    # already perfect and accumulation has nothing to improve; the question only
    # becomes non-trivial once the blood signal sits below the noise, which for
    # small vessels in functional ultrasound is the normal situation.
    cfg = volume_config(24, 24, 24, 64, snr_db=-15.0)
    print(f"simulating a sequence over one fixed tree, {cfg.spatial_shape} x {cfg.nt}, "
          f"SNR {cfg.snr_db:.0f} dB ...")
    seq = simulate_sequence(cfg, n_blocks=16, seed=0)

    rows = accumulation_experiment(seq)
    header = f"{'blocks':>7}{'seconds':>9}{'thin':>9}{'medium':>9}{'thick':>9}"
    print(f"\n{header}\n{'-' * len(header)}")
    for r in rows:
        print(
            f"{r['n_blocks']:>7}{r['seconds']:>9.2f}"
            f"{r['r_0_1.5']:>9.3f}{r['r_1.5_2.5']:>9.3f}{r['r_2.5_inf']:>9.3f}"
        )

    print("\ntime to 80% recall:")
    for band, label in (("r_2.5_inf", "thick"), ("r_1.5_2.5", "medium"), ("r_0_1.5", "thin")):
        n = blocks_to_reach(rows, band)
        if n is None:
            print(f"  {label:<8} not reached within {rows[-1]['n_blocks']} blocks")
        else:
            print(f"  {label:<8} {n} block(s), {n * cfg.nt / cfg.prf:.2f} s of acquisition")


if __name__ == "__main__":
    main()
