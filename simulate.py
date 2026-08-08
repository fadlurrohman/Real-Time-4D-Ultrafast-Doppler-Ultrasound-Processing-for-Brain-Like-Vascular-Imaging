"""
Ultrafast Doppler simulator, 2D+time and 3D+time.

The signal model is the same in both cases:

    IQ(x, t) = tissue(x, t) + blood(x, t) + noise(x, t)

where ``x`` ranges over a 2D plane or a 3D volume. Tissue is a static complex
speckle field, phase-modulated by a spatially smooth displacement built from
cardiac pulsatility, its second harmonic, respiration and a slow drift. Blood
flows along a branching vessel tree, carrying the Doppler shift that corresponds
to its axial velocity. Ground truth for both components is exact, because they
are rendered separately and then added.

Why the dimensionality is a parameter rather than a rewrite
-----------------------------------------------------------
The Casorati matrix -- space down the rows, slow time across the columns -- does
not care how many spatial axes there were. A ``(nz, nx, nt)`` plane and a
``(nz, ny, nx, nt)`` volume both flatten to ``(n_voxels, nt)``, and every filter
in ``filters.py`` operates on that. So the *algorithms* extend to 4D for free.

What does not extend for free is the resource picture. Going from a 96x128 plane
to a 96^3 volume at the same ensemble length multiplies the block by about 75x,
and it still has to be processed before the next one arrives. That is the actual
4D problem, and it is what ``profiling.py`` and ``streaming.py`` measure.

Doppler relation used throughout:

    f_d = 2 * v_axial * f0 / c        [Hz]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

C_SOUND = 1540.0  # m/s


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class Vessel:
    """A straight vessel segment.

    ``start`` and ``end`` are in voxels and must have the same length as the
    spatial shape (2 for a plane, 3 for a volume). ``v_axial`` is the velocity
    component along the beam axis in m/s -- the only component Doppler sees.
    """

    start: tuple[float, ...]
    end: tuple[float, ...]
    radius: float = 2.5
    v_axial: float = 4e-3


@dataclass
class SimConfig:
    """Acquisition, medium and geometry parameters.

    ``spatial_shape`` decides everything: give it two numbers for a plane, three
    for a volume. Defaults are loosely modelled on a small-animal ultrafast brain
    acquisition -- a ~15 MHz probe at a 1 kHz compounded frame rate, the same
    regime as the public PALA rat-brain data.
    """

    spatial_shape: tuple[int, ...] = (96, 128)
    nt: int = 200

    f0: float = 15.625e6  # Hz
    prf: float = 1000.0  # Hz, compounded volume/frame rate

    # tissue
    tissue_db: float = 35.0  # clutter amplitude above blood, dB
    tissue_speckle_sigma: float = 1.6  # voxels
    tissue_disp_um: float = 18.0  # peak displacement (brain pulsatility)
    tissue_motion_hz: float = 4.0
    tissue_resp_hz: float = 0.9
    tissue_drift_um: float = 6.0
    tissue_decorr_frames: float = 60.0
    tissue_decorr_weight: float = 0.12
    tissue_gain_sigma: float = 20.0

    # blood
    contrast: bool = False  # False = diffuse (fUS), True = microbubbles (ULM)
    blood_speckle_sigma: float = 1.2
    blood_decorr_frames: float = 3.0
    blood_speckle_cores: int = 4  # shared speckle realisations, see simulate_block
    mb_count: float = 50.0
    mb_db: float = -8.0
    mb_psf_sigma: float = 1.1
    mb_speed_vox: float = 0.9

    # vessel tree
    n_vessels: int = 0  # 0 = grow a branching tree instead of a fixed list
    tree_root_radius: float = 4.0
    tree_min_radius: float = 0.9
    tree_max_generations: int = 5
    tree_root_length: float = 0.22  # root segment as a fraction of the depth axis

    snr_db: float = 10.0
    vessels: Sequence[Vessel] = field(default_factory=list)

    @property
    def ndim(self) -> int:
        return len(self.spatial_shape)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.spatial_shape))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.spatial_shape) + (self.nt,)

    @property
    def block_bytes(self) -> int:
        """Size of one complex64 Doppler block, in bytes."""
        return self.n_voxels * self.nt * 8

    def doppler_hz(self, v_axial: float) -> float:
        return 2.0 * v_axial * self.f0 / C_SOUND

    def check_aliasing(self, vessels: Sequence[Vessel]) -> None:
        nyquist = self.prf / 2.0
        for v in vessels:
            fd = abs(self.doppler_hz(v.v_axial))
            if fd >= nyquist:
                raise ValueError(
                    f"v={v.v_axial * 1e3:.1f} mm/s gives f_d={fd:.0f} Hz "
                    f">= Nyquist {nyquist:.0f} Hz (aliased)"
                )


# --------------------------------------------------------------------------- #
# vessel geometry
# --------------------------------------------------------------------------- #
def grow_tree(cfg: SimConfig, rng: np.random.Generator) -> list[Vessel]:
    """Grow a branching vessel tree of the right dimensionality.

    Radii follow Murray's law at each bifurcation (``r_parent^3 = sum r_child^3``),
    which puts most of the vessel *length* at the smallest calibres -- the
    regime where clutter filtering actually decides whether a vessel is visible.
    """
    ndim = cfg.ndim
    shape = np.asarray(cfg.spatial_shape, dtype=float)

    start = shape / 2.0
    start[0] = 2.0
    direction = np.zeros(ndim)
    direction[0] = 1.0

    vessels: list[Vessel] = []
    stack = [(start, direction, cfg.tree_root_radius, float(shape[0]) * cfg.tree_root_length, 0)]

    while stack:
        pos, heading, radius, length, generation = stack.pop()
        if radius < cfg.tree_min_radius or generation > cfg.tree_max_generations:
            continue

        heading = heading / (np.linalg.norm(heading) + 1e-12)

        # Truncate the segment where it leaves the volume, rather than clamping
        # its endpoint. Clamping silently rotates the vessel: once the tree
        # reached the far wall, clipped endpoints pinned the depth coordinate
        # and the rendered direction came out perpendicular to the beam, which
        # drove the reported blind-zone fraction to 77% and the median Doppler
        # shift to 0 Hz. Solving for the exit parameter along the ray keeps the
        # direction exactly and only shortens the vessel.
        with np.errstate(divide="ignore", invalid="ignore"):
            to_high = (shape - 1.0 - pos) / heading
            to_low = (0.0 - pos) / heading
        limits = np.where(heading > 0, to_high, np.where(heading < 0, to_low, np.inf))
        t_max = float(np.min(limits[np.isfinite(limits)], initial=np.inf))
        length = min(length, max(t_max, 0.0))
        if length < 2.0:
            continue

        end = pos + heading * length

        # Doppler measures only the component along the beam, so the axial
        # velocity is the flow speed *projected* onto the beam axis (axis 0).
        # Assigning it at random regardless of orientation -- which is what this
        # did originally -- is harmless in a plane, where vessel directions can
        # be chosen to all carry signal, and wrong in a volume: a vessel running
        # perpendicular to the beam genuinely has v_axial ~ 0 and is invisible
        # to any clutter filter, however good. Simulating it as visible would
        # flatter every method equally and hide a real limit of the modality.
        #
        # The projection has to use the *rendered* direction, not the heading it
        # was grown along: clipping the endpoint to the volume bounds changes
        # the direction, and taking the cosine from the pre-clip heading left
        # the stored geometry and the stored velocity disagreeing. A test that
        # recomputes the angle from ``end - start`` catches it immediately.
        rendered = end - pos
        rendered_norm = np.linalg.norm(rendered)
        cos_theta = float(rendered[0] / rendered_norm) if rendered_norm > 1e-9 else 0.0
        speed = rng.uniform(2.0e-3, 1.0e-2) * rng.choice([-1.0, 1.0])
        v_axial = speed * cos_theta
        vessels.append(
            Vessel(tuple(pos), tuple(end), radius=float(radius), v_axial=float(v_axial))
        )

        fractions = rng.dirichlet(np.full(2, 6.0))
        for frac in fractions:
            child_radius = radius * frac ** (1.0 / 3.0)
            if child_radius < cfg.tree_min_radius:
                continue
            perp = rng.standard_normal(ndim)
            perp -= perp @ heading * heading
            norm = np.linalg.norm(perp)
            if norm < 1e-6:
                continue
            perp /= norm
            angle = rng.uniform(0.35, 0.8)
            child_dir = np.cos(angle) * heading + np.sin(angle) * perp
            stack.append(
                (end.copy(), child_dir, child_radius, length * 0.8, generation + 1)
            )

    return vessels


def vessel_masks(cfg: SimConfig, vessels: Sequence[Vessel]) -> np.ndarray:
    """Soft lumen mask per vessel, shape ``(n_vessels, *spatial_shape)``.

    Distance to a segment is computed by projecting each voxel onto it, which is
    written once for any number of spatial dimensions rather than twice.
    """
    grids = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in cfg.spatial_shape], indexing="ij")
    coords = np.stack(grids, axis=0)  # (ndim, *spatial)

    masks = np.zeros((len(vessels),) + tuple(cfg.spatial_shape), dtype=np.float32)
    for i, v in enumerate(vessels):
        a = np.asarray(v.start, dtype=np.float32).reshape((-1,) + (1,) * cfg.ndim)
        b = np.asarray(v.end, dtype=np.float32).reshape((-1,) + (1,) * cfg.ndim)
        ab = b - a
        seg_len2 = float((ab**2).sum()) + 1e-9

        t = ((coords - a) * ab).sum(axis=0) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        closest = a + ab * t[None]
        dist = np.sqrt(((coords - closest) ** 2).sum(axis=0))
        masks[i] = 0.5 * (1.0 - np.tanh((dist - v.radius) / 0.8))

    return masks


# --------------------------------------------------------------------------- #
# field generation
# --------------------------------------------------------------------------- #
def _complex_noise(shape, rng) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def _smooth_spatial(a: np.ndarray, sigma: float, ndim: int) -> np.ndarray:
    axes = tuple(range(ndim))
    return gaussian_filter(a.real, sigma=sigma, axes=axes, mode="wrap") + 1j * gaussian_filter(
        a.imag, sigma=sigma, axes=axes, mode="wrap"
    )


def _speckle(shape, spatial_sigma, temporal_sigma, ndim, rng) -> np.ndarray:
    """Complex speckle with a given grain size and decorrelation time."""
    w = _complex_noise(shape, rng)
    w = _smooth_spatial(w, spatial_sigma, ndim)
    if temporal_sigma > 0:
        w = gaussian_filter1d(w.real, temporal_sigma, axis=-1, mode="wrap") + 1j * gaussian_filter1d(
            w.imag, temporal_sigma, axis=-1, mode="wrap"
        )
    return w / (np.sqrt(np.mean(np.abs(w) ** 2)) + 1e-12)


def _microbubbles(cfg: SimConfig, vessels, rng) -> np.ndarray:
    """Sparse point scatterers flowing along the vessels (the ULM regime)."""
    field = np.zeros(cfg.shape, dtype=np.complex64)
    shape = np.asarray(cfg.spatial_shape)

    for _ in range(int(cfg.mb_count)):
        v = vessels[int(rng.integers(len(vessels)))]
        a = np.asarray(v.start)
        b = np.asarray(v.end)
        along = b - a
        seg_len = np.linalg.norm(along) + 1e-9
        unit = along / seg_len

        s0 = rng.uniform(-0.3, 1.0) * seg_len
        birth = int(rng.integers(-cfg.nt // 3, cfg.nt))
        life = int(rng.integers(cfg.nt // 6, cfg.nt))
        speed = cfg.mb_speed_vox * rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
        amp = rng.uniform(0.4, 1.0) * np.exp(1j * rng.uniform(0, 2 * np.pi))

        for k in range(max(birth, 0), min(birth + life, cfg.nt)):
            s = s0 + speed * (k - birth)
            if s < 0 or s > seg_len:
                continue
            p = np.round(a + unit * s).astype(int)
            if np.any(p < 0) or np.any(p >= shape):
                continue
            field[tuple(p) + (k,)] += amp

    return _smooth_spatial(field, cfg.mb_psf_sigma, cfg.ndim)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def simulate_block(cfg: SimConfig | None = None, seed: int | None = 0) -> dict:
    """Simulate one Doppler ensemble of shape ``(*spatial_shape, nt)``.

    Returns ``iq``, ``blood``, ``tissue`` (all complex64), ``mask``, ``cfg``.
    Memory: three complex64 arrays of the block size, so a 96^3 x 200 volume
    needs roughly 4 GB. Check ``cfg.block_bytes`` before asking for one.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    ndim = cfg.ndim
    shape = cfg.shape

    vessels = list(cfg.vessels) if cfg.vessels else grow_tree(cfg, rng)
    if not vessels:
        raise RuntimeError("no vessels generated; check the tree parameters")
    cfg.check_aliasing(vessels)

    t = np.arange(cfg.nt) / cfg.prf

    # ---- tissue -----------------------------------------------------------
    static = _speckle(tuple(cfg.spatial_shape) + (1,), cfg.tissue_speckle_sigma, 0.0, ndim, rng)

    waveforms = [
        np.sin(2 * np.pi * cfg.tissue_motion_hz * t),
        np.sin(2 * np.pi * 2 * cfg.tissue_motion_hz * t + 0.7),
        np.sin(2 * np.pi * cfg.tissue_resp_hz * t + 1.3),
        (cfg.tissue_drift_um / max(cfg.tissue_disp_um, 1e-9)) * np.linspace(0.0, 1.0, cfg.nt),
    ]
    weights = [1.0, 0.35, 0.55, 1.0]

    disp = np.zeros(shape, dtype=np.float32)
    for waveform, weight in zip(waveforms, weights):
        gain = gaussian_filter(
            rng.standard_normal(cfg.spatial_shape).astype(np.float32),
            sigma=cfg.tissue_gain_sigma,
            mode="wrap",
        )
        gain = 1.0 + 0.6 * gain / (np.std(gain) + 1e-12)
        disp += weight * gain[..., None] * waveform.astype(np.float32)
    disp *= cfg.tissue_disp_um * 1e-6

    phase = (4 * np.pi * cfg.f0 / C_SOUND) * disp
    decorr = _speckle(shape, cfg.tissue_speckle_sigma, cfg.tissue_decorr_frames, ndim, rng)
    tissue = static * np.exp(1j * phase) + cfg.tissue_decorr_weight * decorr
    tissue = tissue / np.sqrt(np.mean(np.abs(tissue) ** 2))
    tissue = (tissue * 10.0 ** (cfg.tissue_db / 20.0)).astype(np.complex64)

    # ---- blood ------------------------------------------------------------
    masks = vessel_masks(cfg, vessels)

    if cfg.contrast:
        blood = _microbubbles(cfg, vessels, rng)
        peak = np.max(np.abs(blood)) + 1e-20
        blood = blood / peak * np.max(np.abs(tissue)) * 10.0 ** (cfg.mb_db / 20.0)
    else:
        # One speckle realisation per vessel is the obvious implementation and it
        # is unusably slow: a tree has tens of branches and each field costs a
        # full-block Gaussian filter, which took 114 s for a single 96x128x200
        # plane. Blood speckle in different vessels only needs to be mutually
        # incoherent, not independently drawn, so a small pool of cores cycled
        # across branches gives the same statistics at a fraction of the cost.
        n_cores = min(cfg.blood_speckle_cores, len(vessels))
        cores = [
            _speckle(shape, cfg.blood_speckle_sigma, cfg.blood_decorr_frames, ndim, rng)
            for _ in range(n_cores)
        ]
        blood = np.zeros(shape, dtype=np.complex64)
        for i, (v, m) in enumerate(zip(vessels, masks)):
            fd = cfg.doppler_hz(v.v_axial)
            modulation = np.exp(1j * (2 * np.pi * fd * t + rng.uniform(0, 2 * np.pi)))
            blood += (m[..., None] * cores[i % n_cores] * modulation).astype(np.complex64)
        blood /= np.sqrt(np.mean(np.abs(blood) ** 2) + 1e-20)

    # ---- noise ------------------------------------------------------------
    noise = (10.0 ** (-cfg.snr_db / 20.0) * _complex_noise(shape, rng)).astype(np.complex64)
    iq = (tissue + blood + noise).astype(np.complex64)

    if cfg.contrast:
        occupancy = gaussian_filter(np.mean(np.abs(blood) ** 2, axis=-1), sigma=1.0)
        mask = (occupancy > 0.2 * occupancy.max()).astype(np.float32)
    else:
        mask = np.clip(masks.sum(axis=0), 0.0, 1.0)

    doppler = np.array([abs(cfg.doppler_hz(v.v_axial)) for v in vessels])

    return {
        "iq": iq,
        "blood": blood.astype(np.complex64),
        "tissue": tissue,
        "mask": mask,
        "vessels": vessels,
        "cfg": cfg,
        "stats": {
            "n_vessels": len(vessels),
            "median_doppler_hz": float(np.median(doppler)),
            # Fraction of vessels running close enough to perpendicular that no
            # clutter filter can recover them. Reported because a metric that
            # ignores the blind zone flatters every method equally.
            "blind_zone_fraction": float((doppler < 20.0).mean()),
            "block_mbytes": cfg.block_bytes / 1e6,
        },
    }


def plane_config(nz=96, nx=128, nt=200, **kwargs) -> SimConfig:
    """Convenience: a 2D+time configuration."""
    return SimConfig(spatial_shape=(nz, nx), nt=nt, **kwargs)


def volume_config(nz=48, ny=48, nx=48, nt=128, **kwargs) -> SimConfig:
    """Convenience: a 3D+time (4D) configuration.

    Defaults are deliberately modest. A 48^3 x 128 block is 113 MB as complex64
    and the simulator holds three of them, so this already needs about 350 MB.
    Scale up only after checking ``cfg.block_bytes``.
    """
    return SimConfig(spatial_shape=(nz, ny, nx), nt=nt, **kwargs)
