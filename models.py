"""
Learned clutter filter: a deep unrolling of the robust-PCA iteration.

Following CORONA (Solomon et al., IEEE TMI 2020) and the more recent unsupervised
unfolded rPCA work: take the iterative solver in ``filters.rpca``, truncate it to
a handful of iterations, and let the constants become learnable. Every layer
still performs the same two proximal steps, so the model stays interpretable and
its failure modes are the familiar ones.

One classical iteration is

    L <- SVT ( X - S + Y/mu ,  1/mu  )
    S <- soft( X - L + Y/mu ,  lam/mu )

The learned layer keeps that structure and relaxes three things: the thresholds
become learnable per-layer scalars, the mixing of X, L and S becomes a learnable
linear combination, and an optional gate lets the sparse threshold vary across
space -- the spatially adaptive behaviour a global SVD cutoff cannot express.

Dimensionality
--------------
Input is Casorati-shaped ``(batch, n_voxels, n_frames)``, complex64, so the model
is indifferent to whether those voxels came from a plane or a volume. The spatial
gate needs a shape to convolve over, so it is given one; pass ``spatial_shape``
at construction and it builds a 2D or 3D convolution accordingly.

Why the inference path matters as much as the architecture
----------------------------------------------------------
``filters.py`` and ``streaming.py`` establish that once the clutter subspace is
known the projection is row-independent and can be chunked. The same is true
here for everything except the SVT, which needs the whole Gram matrix. So a
deployed version would estimate thresholds and bases on a voxel subsample and
stream the shrinkage -- ``inference_chunked`` sketches that path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# proximal operators
# --------------------------------------------------------------------------- #
def soft_threshold(z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Complex soft threshold: shrink magnitude, keep phase."""
    scale = torch.clamp(1.0 - tau / (z.abs() + 1e-9), min=0.0)
    return z * scale.to(z.dtype)


def singular_value_threshold(X: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Batched nuclear-norm prox on tall-skinny complex matrices.

    Same Gram route as the NumPy reference: eigendecompose the ``n_frames x
    n_frames`` matrix ``X^H X`` rather than running a full SVD of the tall one,
    then apply the shrinkage through the right singular vectors only. A small
    jitter keeps repeated eigenvalues from producing NaNs in the backward pass.
    """
    gram = X.mH @ X
    eye = torch.eye(gram.shape[-1], device=X.device, dtype=X.dtype)
    jitter = gram.diagonal(dim1=-2, dim2=-1).abs().mean(-1)[:, None, None]
    evals, evecs = torch.linalg.eigh(gram + 1e-7 * jitter * eye)

    s = torch.sqrt(torch.clamp(evals.real, min=0.0) + 1e-12)
    shrink = (torch.clamp(s - tau, min=0.0) / (s + 1e-9)).to(X.dtype)
    return X @ (evecs * shrink.unsqueeze(-2)) @ evecs.mH


# --------------------------------------------------------------------------- #
# one unrolled layer
# --------------------------------------------------------------------------- #
class UnrolledLayer(nn.Module):
    """One learnable robust-PCA iteration."""

    def __init__(self, spatial_shape: tuple[int, ...] | None = None, kernel: int = 5):
        super().__init__()
        # thresholds in log space so they stay positive
        self.log_tau_low = nn.Parameter(torch.tensor(-1.0))
        self.log_tau_sparse = nn.Parameter(torch.tensor(-3.0))

        # mixing coefficients, initialised at the classical iteration
        self.a_x, self.a_s, self.a_l = (
            nn.Parameter(torch.tensor(1.0)),
            nn.Parameter(torch.tensor(-1.0)),
            nn.Parameter(torch.tensor(0.0)),
        )
        self.b_x, self.b_l, self.b_s = (
            nn.Parameter(torch.tensor(1.0)),
            nn.Parameter(torch.tensor(-1.0)),
            nn.Parameter(torch.tensor(0.0)),
        )

        self.spatial_shape = spatial_shape
        if spatial_shape is not None:
            conv = nn.Conv2d if len(spatial_shape) == 2 else nn.Conv3d
            pad = kernel // 2
            self.gate = nn.Sequential(
                conv(1, 8, kernel, padding=pad),
                nn.ReLU(inplace=True),
                conv(8, 1, kernel, padding=pad),
                nn.Softplus(),
            )

    def forward(self, X, L, S):
        L = singular_value_threshold(
            self.a_x * X + self.a_s * S + self.a_l * L, torch.exp(self.log_tau_low)
        )
        arg = self.b_x * X + self.b_l * L + self.b_s * S
        tau = torch.exp(self.log_tau_sparse)

        if self.spatial_shape is not None:
            energy = arg.abs().pow(2).mean(dim=-1)
            energy = energy.reshape(-1, 1, *self.spatial_shape)
            energy = energy / (
                energy.amax(dim=tuple(range(2, 2 + len(self.spatial_shape))), keepdim=True) + 1e-9
            )
            tau = tau * self.gate(energy).reshape(arg.shape[0], -1, 1)

        return L, soft_threshold(arg, tau)


# --------------------------------------------------------------------------- #
# full model
# --------------------------------------------------------------------------- #
class UnrolledRPCA(nn.Module):
    """Truncated, learnable robust PCA for clutter filtering.

    ``n_layers`` of 5-8 is the useful range: beyond that the latency advantage
    over the streaming SVD path in ``streaming.py`` disappears, and that path
    already meets the deadline.
    """

    def __init__(
        self,
        n_layers: int = 6,
        spatial_shape: tuple[int, ...] | None = None,
        normalise: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(UnrolledLayer(spatial_shape) for _ in range(n_layers))
        self.normalise = normalise
        self.spatial_shape = spatial_shape

    def forward(self, casorati: torch.Tensor) -> dict:
        """``casorati`` is (B, n_voxels, n_frames), complex64."""
        X = casorati
        if self.normalise:
            scale = torch.linalg.matrix_norm(X, ord=2).reshape(-1, 1, 1) + 1e-9
            X = X / scale.to(X.dtype)

        L = torch.zeros_like(X)
        S = torch.zeros_like(X)
        for layer in self.layers:
            L, S = layer(X, L, S)

        if self.normalise:
            S, L = S * scale.to(S.dtype), L * scale.to(L.dtype)
        return {"blood": S, "clutter": L}

    @torch.no_grad()
    def inference_chunked(self, casorati: torch.Tensor, chunk_voxels: int = 4096) -> torch.Tensor:
        """Deployment sketch: estimate on a subsample, shrink in chunks.

        The SVT needs the whole Gram matrix, so the low-rank branch is computed
        once on a voxel subsample -- the same 10:1 aspect-ratio argument as in
        ``streaming.estimate_subspace`` applies. The sparse branch is
        element-wise and streams.

        Not benchmarked. The classical streaming path already meets the
        deadline, so this exists to show the learned filter is not structurally
        incompatible with it, not as a measured result.
        """
        n_voxels, n_frames = casorati.shape[-2:]
        rows = min(n_voxels, max(10 * n_frames, n_voxels // 20))
        idx = torch.randperm(n_voxels, device=casorati.device)[:rows]

        subset = self.forward(casorati[..., idx, :])
        basis = torch.linalg.svd(subset["clutter"][0], full_matrices=False)[2].mH

        out = torch.empty_like(casorati)
        k = min(8, basis.shape[-1])
        clutter_basis = basis[:, :k]
        for start in range(0, n_voxels, chunk_voxels):
            stop = min(start + chunk_voxels, n_voxels)
            chunk = casorati[..., start:stop, :]
            out[..., start:stop, :] = chunk - (chunk @ clutter_basis) @ clutter_basis.mH
        return out


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def complex_nmse_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-invariant complex NMSE.

    A clutter filter is defined only up to a global complex gain, so the optimal
    scale is fitted in closed form before measuring the residual. Without this
    the network wastes capacity matching an arbitrary amplitude.
    """
    e = estimate.reshape(estimate.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    alpha = (e.conj() * t).sum(-1, keepdim=True) / ((e.conj() * e).sum(-1, keepdim=True) + 1e-9)
    resid = alpha * e - t
    return (resid.abs().pow(2).sum(-1) / (t.abs().pow(2).sum(-1) + 1e-9)).mean()


def power_doppler_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on log-compressed power Doppler.

    The complex loss alone over-weights the brightest vessels. Vessel visibility
    is a log-domain judgement, so the objective should carry a log-domain term.
    """
    pe = estimate.abs().pow(2).mean(-1)
    pt = target.abs().pow(2).mean(-1)
    le = torch.log10(pe / (pe.amax(-1, keepdim=True) + 1e-12) + 1e-6)
    lt = torch.log10(pt / (pt.amax(-1, keepdim=True) + 1e-12) + 1e-6)
    return (le - lt).abs().mean()


def combined_loss(out: dict, target_blood: torch.Tensor, w_pd: float = 0.5) -> torch.Tensor:
    return complex_nmse_loss(out["blood"], target_blood) + w_pd * power_doppler_loss(
        out["blood"], target_blood
    )
