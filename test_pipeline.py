"""
Tests for the claims this project makes.

Numerical claims are asserted tightly. Performance claims are asserted loosely
and only where the *ordering* is structural rather than machine-dependent -- a
test that fails on a slower machine teaches nobody anything.

    python -m pytest tests -q
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import filters as F  # noqa: E402
from src import metrics as M  # noqa: E402
from src import profiling as P  # noqa: E402
from src import acquisition as A  # noqa: E402
from src import learned as LN  # noqa: E402
from src import streaming as S  # noqa: E402
from src import temporal as T  # noqa: E402
from src.simulate import plane_config, simulate_block, volume_config  # noqa: E402

PLANE = plane_config(64, 80, 128)
VOLUME = volume_config(28, 28, 28, 128)


# --------------------------------------------------------------------------- #
# dimension-agnosticism
# --------------------------------------------------------------------------- #
def test_casorati_roundtrip_in_2d_and_3d():
    """The reshape must be lossless for any number of spatial axes."""
    for shape in ((8, 9, 16), (5, 6, 7, 16)):
        block = np.random.default_rng(0).standard_normal(shape).astype(np.complex64)
        restored = F.from_casorati(F.to_casorati(block), shape)
        assert restored.shape == shape
        assert np.array_equal(restored, block)


def test_same_filter_runs_on_plane_and_volume():
    """One implementation, both dimensionalities -- the core claim about 4D."""
    for cfg in (PLANE, VOLUME):
        d = simulate_block(cfg, seed=0)
        out = F.svd_filter(d["iq"], 16)
        assert out.shape == d["iq"].shape
        before = M.cnr_db(M.power_doppler(d["iq"]), d["mask"])
        after = M.cnr_db(M.power_doppler(out), d["mask"])
        assert after > before + 15.0, f"{cfg.ndim}D: only {after - before:.1f} dB gained"


def test_automatic_cutoff_underestimates_on_small_volumes():
    """A documented limitation, pinned so it cannot drift unnoticed.

    The Nyquist-referenced rule lands within ~2 dB of optimal on a 40^3 volume
    but undershoots badly on a 28^3 one: it returns k=6 where k=16 gains 30 dB
    against 14.5 dB. The Casorati aspect ratio is 171:1, so the existing guard
    does not fire -- the ratio is fine and the rule is still wrong.

    The likely cause is that a smaller volume has proportionally more of its
    clutter energy in spatially local motion that does not concentrate into the
    leading few singular vectors. It is asserted here rather than tuned away,
    because tuning alpha per volume size would hide a real failure mode behind
    a fitted constant.
    """
    d = simulate_block(VOLUME, seed=0)
    before = M.cnr_db(M.power_doppler(d["iq"]), d["mask"])
    auto = F.adaptive_cutoff(d["iq"], VOLUME.prf)
    gain_auto = M.cnr_db(M.power_doppler(F.svd_filter(d["iq"], auto)), d["mask"]) - before
    gain_best = max(
        M.cnr_db(M.power_doppler(F.svd_filter(d["iq"], k)), d["mask"]) - before
        for k in (8, 12, 16, 24)
    )
    assert gain_best - gain_auto > 5.0, (
        "the undershoot has gone away; if that is a real improvement, update this test "
        f"(auto {gain_auto:.1f} dB vs best {gain_best:.1f} dB)"
    )


def test_doppler_velocity_is_projected_onto_the_beam_axis():
    """A vessel perpendicular to the beam must carry no Doppler shift.

    Originally the tree assigned ``v_axial`` at random regardless of vessel
    orientation. That is harmless in a plane, where directions can be chosen to
    all carry signal, and physically wrong in a volume: Doppler measures only
    the beam-axis component, so a perpendicular vessel is invisible to any
    filter. Simulating it as visible would hide a real limit of the modality.
    """
    from src.simulate import C_SOUND

    d = simulate_block(VOLUME, seed=0)
    cfg = d["cfg"]

    for v in d["vessels"]:
        direction = np.asarray(v.end, dtype=float) - np.asarray(v.start, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        cos_theta = abs(direction[0] / norm)
        # |f_d| = 2 |v| cos(theta) f0 / c, so |f_d| must vanish with cos(theta)
        max_possible = 2 * 1.0e-2 * cos_theta * cfg.f0 / C_SOUND
        assert abs(cfg.doppler_hz(v.v_axial)) <= max_possible + 1e-6, (
            "Doppler shift exceeds what this vessel's angle to the beam allows"
        )

    assert 0.0 <= d["stats"]["blind_zone_fraction"] <= 1.0


def test_volume_simulator_produces_expected_shape_and_size():
    cfg = volume_config(20, 20, 20, 64)
    assert cfg.ndim == 3
    assert cfg.shape == (20, 20, 20, 64)
    assert cfg.block_bytes == 20**3 * 64 * 8
    d = simulate_block(cfg, seed=0)
    assert d["iq"].dtype == np.complex64
    assert d["iq"].shape == cfg.shape


# --------------------------------------------------------------------------- #
# the projection identity
# --------------------------------------------------------------------------- #
def test_clutter_projection_equals_complement_projection():
    """``X - X V_c V_c^H`` must equal ``X V_k V_k^H``. Same maths, 20x less work.

    This identity is the whole reason the filter can be written against the
    small clutter subspace instead of its large complement, and it is what makes
    the projection O(m n k) rather than O(m n^2).
    """
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((400, 32)) + 1j * rng.standard_normal((400, 32))).astype(np.complex128)
    _, V = F.gram_spectrum(X)
    k = 5

    complement = (X @ V[:, k:]) @ V[:, k:].conj().T
    clutter_removed = X - (X @ V[:, :k]) @ V[:, :k].conj().T

    rel = np.linalg.norm(complement - clutter_removed) / np.linalg.norm(complement)
    assert rel < 1e-10, f"projection identity broken, relative error {rel:.2e}"


def test_gram_spectrum_matches_full_svd():
    rng = np.random.default_rng(1)
    X = (rng.standard_normal((300, 24)) + 1j * rng.standard_normal((300, 24))).astype(np.complex128)
    s_full = np.linalg.svd(X, compute_uv=False)
    s_gram, _ = F.gram_spectrum(X)
    assert np.max(np.abs(s_full - s_gram)) / s_full[0] < 1e-10


# --------------------------------------------------------------------------- #
# approximations
# --------------------------------------------------------------------------- #
def test_subsampling_stays_close_to_the_exact_solve():
    """5% of voxels must reproduce the exact filter to within a few percent."""
    d = simulate_block(VOLUME, seed=0)
    exact = F.svd_filter(d["iq"], 16)

    # The subset size needed grows with the clutter rank. At k=16 on this small
    # volume, the 10:1 floor leaves 61% error; at half the voxels it is 19%.
    at_floor = F.subsampled_svd_filter(d["iq"], 16, fraction=0.05)
    generous = F.subsampled_svd_filter(d["iq"], 16, fraction=0.5)

    assert M.relative_error(generous, exact) < 0.25
    assert M.relative_error(at_floor, exact) > M.relative_error(generous, exact), (
        "more voxels must not make the estimate worse"
    )


def test_subsampling_preserves_image_quality():
    """The complex error is a few percent; the CNR must be unchanged.

    Worth separating from the previous test: relative Frobenius error and image
    quality are not the same thing, and it is the second one that decides
    whether the approximation is acceptable.
    """
    d = simulate_block(VOLUME, seed=0)
    exact = M.cnr_db(M.power_doppler(F.svd_filter(d["iq"], 16)), d["mask"])
    approx = M.cnr_db(
        M.power_doppler(F.subsampled_svd_filter(d["iq"], 16, fraction=0.5)), d["mask"]
    )
    assert abs(exact - approx) < 0.5, f"CNR moved from {exact:.2f} to {approx:.2f} dB"


def test_subsample_never_goes_below_the_aspect_ratio_floor():
    """Even a tiny fraction must keep at least 10*nt rows."""
    d = simulate_block(VOLUME, seed=0)
    _, rows = F.subsampled_svd_filter(d["iq"], 16, fraction=1e-6, return_subset_size=True)
    assert rows >= 10 * VOLUME.nt


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #
def test_streaming_matches_the_non_streaming_path_exactly():
    """Chunking changes the memory layout, never the result."""
    d = simulate_block(VOLUME, seed=0)
    basis, _ = S.estimate_subspace(d["iq"], 16, fraction=0.05, seed=0)

    whole = S.apply_projection_streaming(d["iq"], basis, chunk_voxels=10**9)
    chunked = S.apply_projection_streaming(d["iq"], basis, chunk_voxels=512)

    assert M.relative_error(chunked, whole) < 1e-6


def test_streaming_working_set_is_bounded():
    """Resident memory must depend on chunk size, not block size."""
    small = S.working_set_bytes(4096, 128)
    for n_voxels in (10**5, 10**7):
        assert S.working_set_bytes(4096, 128) == small, "working set leaked block size"
    assert small < 20e6


def test_real_time_budget_arithmetic():
    budget = S.RealTimeBudget(n_frames=128, prf=1000.0, measured_ms=64.0)
    assert abs(budget.deadline_ms - 128.0) < 1e-9
    assert budget.meets_deadline
    assert abs(budget.headroom - 2.0) < 1e-9

    missed = S.RealTimeBudget(n_frames=128, prf=1000.0, measured_ms=256.0)
    assert not missed.meets_deadline


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #
def test_cost_model_flop_split_is_sane():
    """With no subsampling the Gram term must dominate; with it, the projection."""
    full = P.CostModel(n_voxels=64000, n_frames=128, cutoff=6).breakdown()
    assert full["gram_%"] > 70.0

    sub = P.CostModel(
        n_voxels=64000, n_frames=128, cutoff=6, subsample_rows=3200
    ).breakdown()
    assert sub["projection_%"] > sub["gram_%"]


def test_subsampling_drops_arithmetic_intensity():
    """Subsampling cuts flops per byte by about 4.6x. Machine-independent."""
    full = P.CostModel(64000, 128, 6).arithmetic_intensity
    sub = P.CostModel(64000, 128, 6, subsample_rows=3200).arithmetic_intensity
    assert full / sub > 4.0, f"intensity only fell from {full:.1f} to {sub:.1f}"


def test_roofline_crossover_depends_on_the_machine():
    """Whether subsampling turns the kernel memory-bound is not universal.

    The crossover happens when the machine's flops-per-byte ratio exceeds the
    subsampled arithmetic intensity of about 10. Measured on the development
    box (115 GFLOP/s, 10.1 GB/s, ratio 11.4) it does cross. On a machine at
    ratio 10 or below it does not, and the exact solve stays compute-bound
    either way. Asserting a single verdict would have been wrong.
    """
    for gflops, gbytes, expected in ((200.0, 10.0, "memory"), (50.0, 10.0, "compute")):
        full = P.CostModel(64000, 128, 6).roofline_ms(gflops, gbytes)
        sub = P.CostModel(64000, 128, 6, subsample_rows=3200).roofline_ms(gflops, gbytes)
        assert full["bound_by"] == "compute"
        assert sub["bound_by"] == expected


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_cutoff_guards_fire_on_bad_geometry():
    """Both known silent failures must warn rather than degrade quietly."""
    squat = simulate_block(plane_config(32, 40, 200), seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        F.adaptive_cutoff(squat["iq"], 1000.0)
    assert caught, "low Casorati aspect ratio should warn"

    short = simulate_block(plane_config(64, 64, 64), seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        F.adaptive_cutoff(short["iq"], 1000.0)
    assert caught, "short ensemble should warn"


def test_aliasing_is_rejected():
    """A velocity past Nyquist must raise rather than fold silently."""
    from src.simulate import SimConfig, Vessel

    cfg = SimConfig(spatial_shape=(16, 16), nt=64, prf=1000.0)
    fast = Vessel(start=(2.0, 2.0), end=(14.0, 14.0), radius=2.0, v_axial=1.0)
    try:
        cfg.check_aliasing([fast])
    except ValueError:
        return
    raise AssertionError("aliased velocity was accepted")


# --------------------------------------------------------------------------- #
# the learned filter
# --------------------------------------------------------------------------- #
def test_learned_filter_initialises_at_the_classical_solver():
    """Untrained parameters must reproduce the hand-tuned schedule, not noise."""
    p = LN.initial_params(4)
    assert p.shape == (16,)
    # mixing coefficients start at the classical iteration
    assert np.allclose(p[2::4], -1.0)
    assert np.allclose(p[3::4], -1.0)
    # thresholds decay across layers
    assert np.all(np.diff(p[0::4]) < 0)


def test_training_improves_on_its_own_initialisation():
    """Training must actually move the loss, on a budget small enough to run here."""
    from src.simulate import volume_config

    cfgs = [volume_config(12, 12, 12, 32, tissue_db=30 + 5 * i) for i in range(2)]
    data = LN.build_training_set(cfgs, seed=0)

    start = LN.objective(LN.initial_params(3), data)
    result = LN.train(data, n_layers=3, max_iter=4, max_evals=60, verbose=False)
    assert result["final_loss"] < start


def test_learned_filter_loses_to_svd_and_that_is_the_reported_result():
    """The measured outcome, pinned so it cannot quietly change.

    A 4-layer unrolling with 16 learnable scalars, trained to convergence on
    matched data, reaches CNR -7.6 dB where SVD with k=8 reaches +4.8 dB, at 17x
    the latency. Training works -- it improves substantially on its own
    initialisation -- but the model class is simply worse than a plain subspace
    filter on diffuse blood. If this test ever fails because the learned filter
    won, that is a real finding and the README needs rewriting.
    """
    from src.simulate import simulate_block, volume_config

    d = simulate_block(volume_config(16, 16, 16, 48, tissue_db=32), seed=99)
    untrained = LN.filter_block(d["iq"], LN.initial_params(4))
    svd = F.svd_filter(d["iq"], 8)

    cnr_untrained = M.cnr_db(M.power_doppler(untrained), d["mask"])
    cnr_svd = M.cnr_db(M.power_doppler(svd), d["mask"])
    assert cnr_svd > cnr_untrained + 5.0, (
        f"SVD {cnr_svd:.1f} dB vs unrolled {cnr_untrained:.1f} dB"
    )


# --------------------------------------------------------------------------- #
# acquisition design
# --------------------------------------------------------------------------- #
def test_planner_derives_a_feasible_acquisition():
    """A plan derived from the target velocities must satisfy every constraint."""
    assert A.plan().check() == []


def test_planner_catches_aliasing():
    """A plan that looks reasonable but folds the fastest vessel must be rejected."""
    bad = A.AcquisitionPlan(
        f0=15.625e6, prf=500.0, nt=64, n_voxels=4000, v_max=0.05, v_min=2e-3
    )
    problems = bad.check()
    assert any("aliasing" in p for p in problems)


def test_planner_catches_a_squat_casorati_matrix():
    """The 10:1 floor has to be enforced at design time, not discovered later."""
    squat = A.AcquisitionPlan(
        f0=15.625e6, prf=4000.0, nt=600, n_voxels=3000, v_max=0.05, v_min=2e-3
    )
    assert any("aspect ratio" in p for p in squat.check())


def test_nyquist_velocity_matches_the_doppler_relation():
    p = A.plan()
    assert abs(p.doppler_hz(p.nyquist_velocity) - p.prf / 2.0) < 1e-6


# --------------------------------------------------------------------------- #
# temporal
# --------------------------------------------------------------------------- #
def test_segmentation_threshold_is_calibrated_not_fixed():
    """A fixed percentile caps recall by construction.

    The phantom's vessels occupy ~9% of voxels. A fixed 96th-percentile threshold
    predicts 4%, so recall could never exceed 0.45 no matter how many blocks were
    accumulated -- and thick vessels scored below medium ones purely because the
    few predicted voxels were spread over a larger true region. Every number in
    the first run of the accumulation experiment was an artefact of that.
    """
    rng = np.random.default_rng(0)
    power = rng.random((8, 8, 8))
    for fraction in (0.05, 0.2):
        predicted = T.segment(power, fraction).mean()
        assert abs(predicted - fraction) < 0.02


def test_accumulation_improves_recall_at_low_snr():
    """The premise: averaging blocks rescues vessels a single block misses.

    Asserted on the mean across calibre bands, at the configuration the claim in
    the README is actually about. On a smaller volume the thinnest band holds
    only a handful of voxels, so its recall moves in steps too coarse to test --
    which is a property of the phantom, not evidence against accumulation.
    """
    from src.simulate import volume_config

    cfg = volume_config(24, 24, 24, 64, snr_db=-15.0)
    seq = T.simulate_sequence(cfg, n_blocks=8, seed=0)
    rows = T.accumulation_experiment(seq, depths=(1, 8))

    bands = ["r_0_1.5", "r_1.5_2.5", "r_2.5_inf"]
    single = np.mean([rows[0][b] for b in bands])
    eight = np.mean([rows[1][b] for b in bands])
    assert eight > single + 0.15, f"accumulation gained only {eight - single:.3f}"


def test_temporal_budget_arithmetic():
    """Time-to-detection is blocks times ensemble duration."""
    budget = T.TemporalBudget(n_blocks=8, n_frames=128, prf=1000.0)
    assert abs(budget.seconds - 1.024) < 1e-9


def test_learned_temporal_weights_do_not_generalise():
    """Pinned negative result: exchangeable blocks leave nothing to learn.

    Uniform weighting is optimal in expectation when every block carries the same
    signal and independent identically-distributed noise. If this test ever fails
    because learned weights won, it means the simulator gained a source of
    non-stationarity and the README needs rewriting.
    """
    from src.simulate import volume_config

    seq = T.simulate_sequence(volume_config(20, 20, 20, 48, snr_db=-15.0), n_blocks=4, seed=7)
    fg = float((seq["mask"] > 0.5).mean())

    def thin_recall(weights):
        accumulated = np.tensordot(weights, seq["power"][:4], axes=(0, 0))
        pred = T.segment(accumulated, fg)
        return T.recall_by_radius(pred, seq["mask"], seq["radius"], T.DEFAULT_BANDS)["r_0_1.5"]

    uniform = np.ones(4) / 4
    skewed = np.array([0.4, 0.3, 0.2, 0.1])
    assert abs(thin_recall(skewed) - thin_recall(uniform)) < 0.15


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
