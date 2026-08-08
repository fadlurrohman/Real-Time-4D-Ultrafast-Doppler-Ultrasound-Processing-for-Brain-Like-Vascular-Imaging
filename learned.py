"""
A learned clutter filter that is actually trained.

``models.py`` holds a PyTorch unrolling that has never been run. That is a real
weakness: an untrained model is a claim, not a result. This module closes it
with a version small enough to train here and now, in NumPy, with no GPU.

The trick is to cut the parameter count until derivative-free optimisation is
viable. The PyTorch version learns per-layer thresholds, mixing coefficients and
a convolutional gate -- tens of thousands of parameters, which needs autodiff and
a GPU. Strip the gate and keep four scalars per layer, and a five-layer filter
has twenty parameters. Powell's method handles that comfortably, and twenty
well-chosen scalars is a real model: it is exactly the schedule of thresholds
and step sizes that the classical solver fixes by hand.

Each layer is one robust-PCA iteration with the constants made free:

    L <- SVT ( a_x * X + a_s * S ,  exp(log_tau_low)    )
    S <- soft( b_x * X + b_l * L ,  exp(log_tau_sparse) )

Initialised at the classical iteration (a_x = b_x = 1, a_s = b_l = -1) with a
geometrically decaying threshold schedule, so training starts from the known-good
solver and can only improve on it -- or fail to, which is also a result worth
reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .filters import gram_spectrum, soft_threshold, to_casorati


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
N_PARAMS_PER_LAYER = 4  # log_tau_low, log_tau_sparse, a_s, b_l


def initial_params(n_layers: int = 5) -> np.ndarray:
    """Start at the classical solver with a decaying threshold schedule.

    ``a_x`` and ``b_x`` are pinned at 1.0 rather than learned: the data term has
    to enter each layer at unit scale or the whole recursion is free to rescale
    itself, which makes the thresholds unidentifiable and the optimiser wander.
    """
    params = []
    for i in range(n_layers):
        params.extend(
            [
                np.log(0.8 * 0.55**i),  # log_tau_low, decaying
                np.log(2e-3 * 0.6**i),  # log_tau_sparse, decaying
                -1.0,  # a_s
                -1.0,  # b_l
            ]
        )
    return np.asarray(params, dtype=np.float64)


def _svt(X: np.ndarray, tau: float) -> np.ndarray:
    """Nuclear-norm prox via the Gram route, without forming U."""
    s, V = gram_spectrum(X)
    shrink = np.where(s > 1e-12, np.maximum(s - tau, 0.0) / np.maximum(s, 1e-12), 0.0)
    return X @ (V * shrink) @ V.conj().T


def forward(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Run the unrolled filter on a Casorati matrix. Returns the blood estimate.

    ``X`` is normalised by its spectral norm on entry and rescaled on exit, so
    the learned thresholds are dimensionless and transfer between acquisitions
    with different gain.
    """
    scale = np.linalg.norm(X, 2) + 1e-20
    Xn = X / scale

    L = np.zeros_like(Xn)
    S = np.zeros_like(Xn)

    for i in range(len(params) // N_PARAMS_PER_LAYER):
        log_tl, log_ts, a_s, b_l = params[i * N_PARAMS_PER_LAYER : (i + 1) * N_PARAMS_PER_LAYER]
        L = _svt(Xn + a_s * S, float(np.exp(log_tl)))
        S = soft_threshold(Xn + b_l * L, float(np.exp(log_ts)))

    return S * scale


def filter_block(block: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Apply the learned filter to a ``(*spatial, nt)`` block."""
    return forward(to_casorati(block).astype(np.complex128), params).reshape(block.shape)


# --------------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------------- #
def scale_invariant_nmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Complex NMSE after fitting the optimal global gain.

    A clutter filter is defined only up to a complex scale factor, so a metric
    that penalises amplitude mismatch would be training the model to reproduce
    an arbitrary gain rather than the right subspace.
    """
    e = estimate.ravel()
    t = truth.ravel()
    alpha = np.vdot(e, t) / (np.vdot(e, e) + 1e-20)
    return float(
        (np.abs(alpha * e - t) ** 2).sum() / ((np.abs(t) ** 2).sum() + 1e-20)
    )


def power_doppler_term(estimate: np.ndarray, truth: np.ndarray, shape) -> float:
    """L1 between log-compressed power Doppler maps.

    The complex term alone is dominated by the brightest vessels. Visibility is a
    log-domain judgement, so the objective needs a log-domain component or the
    model optimises for the trunks and abandons the periphery.
    """
    pe = np.mean(np.abs(estimate.reshape(shape)) ** 2, axis=-1)
    pt = np.mean(np.abs(truth.reshape(shape)) ** 2, axis=-1)
    le = np.log10(pe / (pe.max() + 1e-20) + 1e-6)
    lt = np.log10(pt / (pt.max() + 1e-20) + 1e-6)
    return float(np.abs(le - lt).mean())


@dataclass
class TrainingSet:
    """Casorati matrices with their exact blood ground truth."""

    inputs: list[np.ndarray]
    targets: list[np.ndarray]
    shapes: list[tuple]

    def __len__(self) -> int:
        return len(self.inputs)


def build_training_set(configs, seed: int = 0) -> TrainingSet:
    """Simulate blocks and flatten them, keeping the ground-truth blood signal."""
    from .simulate import simulate_block

    inputs, targets, shapes = [], [], []
    rng = np.random.default_rng(seed)
    for cfg in configs:
        d = simulate_block(cfg, seed=int(rng.integers(0, 2**31 - 1)))
        inputs.append(to_casorati(d["iq"]).astype(np.complex128))
        targets.append(to_casorati(d["blood"]).astype(np.complex128))
        shapes.append(d["iq"].shape)
    return TrainingSet(inputs, targets, shapes)


def objective(params: np.ndarray, data: TrainingSet, w_pd: float = 0.5) -> float:
    """Mean loss over the training blocks."""
    total = 0.0
    for X, target, shape in zip(data.inputs, data.targets, data.shapes):
        estimate = forward(X, params)
        total += scale_invariant_nmse(estimate, target) + w_pd * power_doppler_term(
            estimate, target, shape
        )
    return total / max(len(data), 1)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(
    data: TrainingSet,
    n_layers: int = 5,
    max_iter: int = 40,
    max_evals: int = 800,
    w_pd: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Fit the layer parameters by Powell's method.

    Derivative-free rather than gradient-based, for two reasons. The parameter
    count is small enough that it works, and the forward pass contains an
    eigendecomposition whose gradient is ill-conditioned wherever eigenvalues
    are close together -- which for a clutter spectrum is most of the time.
    Avoiding the gradient avoids the instability rather than patching it.
    """
    x0 = initial_params(n_layers)
    history = []

    def wrapped(p):
        value = objective(p, data, w_pd)
        history.append(value)
        return value

    start = wrapped(x0)
    if verbose:
        print(f"  initial loss (classical schedule): {start:.5f}", flush=True)

    result = minimize(
        wrapped,
        x0,
        method="Powell",
        options={
            "maxiter": max_iter,
            "maxfev": max_evals,
            "xtol": 1e-3,
            "ftol": 1e-4,
            "disp": False,
        },
    )

    if verbose:
        print(f"  final loss after training:         {result.fun:.5f}", flush=True)
        print(f"  improvement: {100 * (start - result.fun) / start:+.1f}%", flush=True)

    return {
        "params": result.x,
        "initial_loss": float(start),
        "final_loss": float(result.fun),
        "n_evals": len(history),
        "history": history,
        "n_layers": n_layers,
    }


def evaluate(params: np.ndarray, data: TrainingSet, w_pd: float = 0.5) -> dict:
    """Loss and its two components on a held-out set."""
    nmse, pd = [], []
    for X, target, shape in zip(data.inputs, data.targets, data.shapes):
        estimate = forward(X, params)
        nmse.append(scale_invariant_nmse(estimate, target))
        pd.append(power_doppler_term(estimate, target, shape))
    return {
        "nmse": float(np.mean(nmse)),
        "pd_term": float(np.mean(pd)),
        "loss": float(np.mean(nmse) + w_pd * np.mean(pd)),
    }


def save(path: str, result: dict) -> None:
    np.savez(path, params=result["params"], n_layers=result["n_layers"])


def load(path: str) -> np.ndarray:
    return np.load(path)["params"]
