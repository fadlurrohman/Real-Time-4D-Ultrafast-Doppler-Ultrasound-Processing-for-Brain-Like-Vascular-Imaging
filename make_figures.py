"""Generate the README figures.

    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import filters as F  # noqa: E402
from src import metrics as M  # noqa: E402
from src import profiling as P  # noqa: E402
from src import streaming as S  # noqa: E402
from src.simulate import plane_config, simulate_block, volume_config  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.titlesize": 8})


def mip(volume: np.ndarray) -> np.ndarray:
    """Maximum intensity projection along the first axis, for 3D power maps."""
    return volume.max(axis=0) if volume.ndim == 3 else volume


def figure_dimensionality():
    """The same code on a plane and on a volume."""
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2))

    for row, (tag, cfg) in enumerate(
        [("2D+t plane", plane_config(96, 128, 200)), ("3D+t volume", volume_config(40, 40, 40, 128))]
    ):
        d = simulate_block(cfg, seed=0)
        k = 16
        filtered = F.svd_filter(d["iq"], k)

        axes[row, 0].imshow(mip(M.to_db(M.power_doppler(d["iq"]))), cmap="hot", aspect="auto")
        axes[row, 0].set_title(f"{tag}: unfiltered")
        axes[row, 1].imshow(mip(M.to_db(M.power_doppler(filtered))), cmap="hot", aspect="auto")
        axes[row, 1].set_title(f"{tag}: SVD k={k}")
        for ax in axes[row, :2]:
            ax.set_xticks([])
            ax.set_yticks([])

        ax = axes[row, 2]
        X = F.to_casorati(d["iq"]).astype(np.complex128)
        s_iq, _ = F.gram_spectrum(X)
        s_ti, _ = F.gram_spectrum(F.to_casorati(d["tissue"]).astype(np.complex128))
        s_bl, _ = F.gram_spectrum(F.to_casorati(d["blood"]).astype(np.complex128))
        for s, style, label in ((s_ti, "-", "tissue"), (s_bl, "--", "blood"), (s_iq, ":", "IQ")):
            ax.semilogy(s / s[0], style, lw=1.2, label=label)
        ax.set_xlim(0, 60)
        ax.set_ylim(1e-4, 1.5)
        ax.set_xlabel("singular index")
        ax.set_title(f"{tag}: Casorati {cfg.n_voxels}x{cfg.nt}")
        ax.legend(frameon=False, fontsize=6.5)
        ax.grid(alpha=0.3)

    fig.suptitle("one implementation, both dimensionalities", y=1.0)
    fig.tight_layout()
    out = FIG / "01_dimensionality.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def figure_latency():
    """Roofline, scaling law, and the chunk plateau."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    # --- flop split -------------------------------------------------------
    ax = axes[0]
    labels = ["exact", "subsample 5%"]
    models = [P.CostModel(64000, 128, 6), P.CostModel(64000, 128, 6, subsample_rows=3200)]
    parts = np.array([[m.gram_flops, m.eigh_flops, m.projection_flops] for m in models]) / 1e9
    bottom = np.zeros(2)
    for i, name in enumerate(["Gram", "eigh", "projection"]):
        ax.bar(labels, parts[:, i], 0.55, bottom=bottom, label=name)
        bottom += parts[:, i]
    for i, m in enumerate(models):
        ax.text(i, bottom[i] * 1.04, f"{m.arithmetic_intensity:.1f} f/B", ha="center", fontsize=6.5)
    ax.set_ylabel("GFLOP per block")
    ax.set_title("where the work goes\n(label = arithmetic intensity)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.3, axis="y")

    # --- scaling ----------------------------------------------------------
    ax = axes[1]
    res = P.scaling_experiment(
        lambda b: F.svd_filter(b.reshape(b.shape[0], 1, b.shape[1]), 6),
        P.random_block,
        voxel_counts=[8000, 16000, 32000, 64000, 128000],
        n_frames=128,
        repeats=2,
    )
    x = np.array([r["n_voxels"] for r in res["rows"]], dtype=float)
    y = np.array([r["ms"] for r in res["rows"]], dtype=float)
    ax.loglog(x, y, "o-", lw=1.3, label=f"measured (p={res['exponent']:.2f})")
    ax.loglog(x, y[0] * (x / x[0]), "--", lw=1.2, label="cost model (p=1)")
    ax.set_xlabel("voxels")
    ax.set_ylabel("ms per block")
    ax.set_title("latency scaling")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.3, which="both")

    # --- chunk plateau ----------------------------------------------------
    ax = axes[2]
    d = simulate_block(volume_config(40, 40, 40, 128), seed=0)
    sweep = S.chunk_sweep(d["iq"], 6, [64, 256, 1024, 4096, 16384, 64000], repeats=5)
    chunks = [r["chunk_voxels"] for r in sweep["rows"]]
    ax.semilogx(chunks, [r["projection_ms"] for r in sweep["rows"]], "o-", lw=1.3, color="crimson")
    ax.set_xlabel("voxels per chunk")
    ax.set_ylabel("projection ms")
    ax.set_title("chunk size: flat, then cache-bound")
    ax.grid(alpha=0.3, which="both")
    twin = ax.twinx()
    twin.semilogx(
        chunks, [r["working_set_mb"] for r in sweep["rows"]], "s--", lw=1.0, color="steelblue"
    )
    twin.set_ylabel("working set MB", color="steelblue")

    fig.suptitle("latency analysis, not just a stopwatch", y=1.03)
    fig.tight_layout()
    out = FIG / "02_latency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")
    return res


def figure_realtime():
    """Every method against the acquisition deadline."""
    cfg = volume_config(40, 40, 40, 128)
    d = simulate_block(cfg, seed=0)
    iq, mask = d["iq"], d["mask"]
    k = 16
    deadline = 1000.0 * cfg.nt / cfg.prf

    import gc

    def timed(fn, repeats=3):
        gc.collect()
        out = fn()
        times = []
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
        return out, float(np.median(times))

    methods = [
        ("highpass", lambda: F.highpass_filter(iq, cfg.prf, 40.0)),
        ("exact SVD", lambda: F.svd_filter(iq, k)),
        ("subsample 20%", lambda: F.subsampled_svd_filter(iq, k, fraction=0.20)),
        ("subsample 5%", lambda: F.subsampled_svd_filter(iq, k, fraction=0.05)),
        ("streaming", lambda: S.streaming_filter(iq, k, 0.05, 4096)),
    ]

    names, times_ms, cnrs = [], [], []
    for name, fn in methods:
        est, ms = timed(fn)
        names.append(name)
        times_ms.append(ms)
        cnrs.append(M.cnr_db(M.power_doppler(est), mask))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))

    ax = axes[0]
    colours = ["seagreen" if t <= deadline else "indianred" for t in times_ms]
    ax.bar(names, times_ms, 0.6, color=colours)
    ax.axhline(deadline, color="black", ls="--", lw=1.2, label=f"deadline {deadline:.0f} ms")
    ax.set_ylabel("ms per block")
    ax.set_title("4D block, single CPU core")
    ax.tick_params(axis="x", labelrotation=25, labelsize=6.5)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.scatter(times_ms, cnrs, s=42, c=colours)
    for n, t, c in zip(names, times_ms, cnrs):
        ax.annotate(n, (t, c), fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.axvline(deadline, color="black", ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("ms per block")
    ax.set_ylabel("CNR (dB)")
    ax.set_title("quality is flat; only latency moves")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("real-time feasibility on 3D+time data", y=1.03)
    fig.tight_layout()
    out = FIG / "03_realtime.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")

    print("\n  method            ms      CNR dB   deadline")
    for n, t, c in zip(names, times_ms, cnrs):
        print(f"  {n:<16}{t:>7.1f}{c:>11.2f}   {'ok' if t <= deadline else 'MISS'}")


def main():
    print("[dimensionality]")
    figure_dimensionality()
    print("[latency]")
    res = figure_latency()
    print(f"  fitted scaling exponent p = {res['exponent']:.2f}")
    print("[real-time]")
    figure_realtime()
    print("[learned]")
    figure_learned()


if __name__ == "__main__":
    main()


def figure_learned():
    """The negative result: a trained unrolling against a plain SVD."""
    from src import learned as LN
    from src.simulate import volume_config

    params = np.load(Path(__file__).resolve().parents[1] / "learned_params.npz")["params"]
    cfg = volume_config(16, 16, 16, 48, tissue_db=32)
    d = simulate_block(cfg, seed=99)
    iq, blood, mask = d["iq"], d["blood"], d["mask"]

    entries = [
        ("unrolled\n(classical init)", LN.filter_block(iq, LN.initial_params(4))),
        ("unrolled\n(trained)", LN.filter_block(iq, params)),
        ("SVD k=8", F.svd_filter(iq, 8)),
        ("oracle", blood),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.9))

    for ax, (name, est) in zip(axes[:4], entries):
        ax.imshow(mip(M.to_db(M.power_doppler(est))), cmap="hot", aspect="auto")
        cnr = M.cnr_db(M.power_doppler(est), mask)
        ax.set_title(f"{name}\nCNR {cnr:.1f} dB")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[4]
    names = [n.replace("\n", " ") for n, _ in entries]
    cnrs = [M.cnr_db(M.power_doppler(e), mask) for _, e in entries]
    colours = ["indianred", "indianred", "seagreen", "0.6"]
    ax.barh(range(len(names)), cnrs, color=colours)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("CNR (dB)")
    ax.set_title("training helps,\nand still loses to SVD")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("learned clutter filter, trained on matched data", y=1.04)
    fig.tight_layout()
    out = FIG / "04_learned_vs_svd.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")
    for n, c in zip(names, cnrs):
        print(f"    {n:<26}{c:>7.2f} dB")


def figure_temporal():
    """Time-to-detection: what accumulation buys, and what it costs in seconds."""
    from src import temporal as T
    from src.simulate import volume_config

    cfg = volume_config(24, 24, 24, 64, snr_db=-15.0)
    seq = T.simulate_sequence(cfg, n_blocks=16, seed=0)
    rows = T.accumulation_experiment(seq, depths=(1, 2, 4, 8, 16))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

    ax = axes[0]
    blocks = [r["n_blocks"] for r in rows]
    for key, label in (("r_0_1.5", "thin (r<=1.5)"), ("r_1.5_2.5", "medium"), ("r_2.5_inf", "thick")):
        ax.plot(blocks, [r[key] for r in rows], "o-", lw=1.3, label=label)
    ax.axhline(0.8, color="black", ls="--", lw=1, label="80% target")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("blocks accumulated")
    ax.set_ylabel("recall")
    ax.set_title("accumulation rescues what one block misses")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([r["seconds"] for r in rows], [r["r_0_1.5"] for r in rows], "o-", lw=1.4, color="crimson")
    ax.axhline(0.8, color="black", ls="--", lw=1)
    ax.set_xlabel("seconds of acquisition")
    ax.set_ylabel("thin-vessel recall")
    ax.set_title("the cost is measured in seconds,\nnot milliseconds")
    ax.grid(alpha=0.3)

    ax = axes[2]
    single = seq["power"][0]
    many = seq["power"].mean(axis=0)
    combined = np.concatenate([mip(M.to_db(single)), mip(M.to_db(many))], axis=1)
    ax.imshow(combined, cmap="hot", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("1 block  |  16 blocks")

    fig.suptitle("temporal accumulation at SNR -15 dB", y=1.04)
    fig.tight_layout()
    out = FIG / "05_temporal_accumulation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")
    for r in rows:
        print(f"    {r['n_blocks']:>3} blocks  {r['seconds']:>5.2f}s  thin={r['r_0_1.5']:.3f}")
