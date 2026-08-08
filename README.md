# Real-time clutter filtering for 4D ultrafast Doppler

Separating blood from tissue clutter in ultrafast ultrasound, on **3D+time
volumes**, benchmarked against the **deadline set by the acquisition** rather
than against image quality alone.

A Doppler ensemble of 128 frames at a 1 kHz volume rate takes 128 ms to acquire.
If the filter takes longer, blocks arrive faster than they are consumed and the
pipeline falls behind without bound. That number, not a preference for speed, is
what makes this a real-time problem.

**Scope, stated up front.** This is a self-initiated study on simulated data, not
a research contribution. SVD clutter filtering is Demené et al. (2015), adaptive
thresholding is Baranger et al. (2018), robust PCA is Lin/Chen/Ma (2010), and
deep unrolling for this problem is CORONA (Solomon et al., 2020). What is mine is
the implementation, the 4D cost analysis, and the failure analysis below.

---

## What this covers

| | |
|---|---|
| **3D volumetric data** | Simulator and every filter run on `(nz, ny, nx, nt)` |
| **4D (3D+time)** | The same, with slow time as the fourth axis |
| **Latency analysis** | Analytic cost model, measured roofline, scaling laws |
| **Real-time deployment** | Bounded-memory streaming path measured against the deadline |

| Module | Contents |
|---|---|
| `src/simulate.py` | Ultrafast Doppler simulator, 2D+t and 3D+t, exact ground truth |
| `src/filters.py` | Dimension-agnostic SVD, voxel subsampling, randomized sketch, RPCA |
| `src/profiling.py` | Flop/byte model, machine calibration, roofline, scaling fits |
| `src/streaming.py` | Chunked projection, real-time budget arithmetic |
| `src/metrics.py` | CNR, scale-invariant NMSE, power Doppler correlation |
| `src/learned.py` | **Trainable** unrolled filter in NumPy, plus its training loop |
| `src/acquisition.py` | Acquisition planner and the physical-experiment design |
| `src/temporal.py` | Accumulation over blocks, time-to-detection, learned temporal weighting |
| `src/models.py` | PyTorch unrolled RPCA (untrained — the NumPy one is the trained version) |
| `src/benchmark.py` | The quality/latency/deadline table |
| `tests/` | 29 tests pinning every claim below |

```bash
pip install -r requirements.txt
python tests/test_pipeline.py                    # 29/29
python -m src.temporal                           # time-to-detection study
python -m src.acquisition                        # acquisition design + protocol
python -m src.benchmark --volume --roofline
python scripts/make_figures.py
```

---

## Why 4D is free for the algorithm and expensive for the machine

Every filter operates on the Casorati matrix: space down the rows, slow time
across the columns. A `(96, 128, 200)` plane and a `(40, 40, 40, 128)` volume both
flatten to `(n_voxels, nt)`, so **one implementation covers both** — the reshape
is the entire "extension to 4D" as far as the mathematics is concerned.

| | 2D+t | 3D+t |
|---|---|---|
| block | `(96, 128, 200)`, 20 MB | `(40, 40, 40, 128)`, 66 MB |
| Casorati | 12288 × 200 | 64000 × 128 |
| vessels | 61 | 61 |
| CNR, unfiltered → SVD k=16 | −33.8 → 6.7 dB | −18.2 → 5.1 dB |

![dimensionality](figures/01_dimensionality.png)

What does not extend for free is cost. For `m` voxels, `n` frames, clutter rank
`k`:

```
Gram matrix   X^H X      ->  m·n²   complex MACs
eigendecomposition       ->  O(n³)  negligible while n << m
clutter projection       ->  2·m·n·k
memory traffic           ->  read m·n, write m·n
```

Measured scaling exponent in `m`: **1.03**, against 1.00 from the model.

The roofline verdict depends on both the subsampling and the clutter rank, and
saying otherwise would be wrong. For a 64000 × 128 block:

| | k=6, exact | k=6, 5% subsample | k=16, 5% subsample |
|---|---:|---:|---:|
| work | 9.34 GFLOP | 1.37 GFLOP | 2.68 GFLOP |
| Gram / eigh / projection | 90 / 2 / 8 % | 31 / 12 / 57 % | 16 / 6 / 78 % |
| arithmetic intensity | 47.5 f/B | 10.2 f/B | 20.0 f/B |
| bound by | compute | memory | compute |

Subsampling cuts arithmetic intensity by roughly 4.6×, which is machine- and
rank-independent and is asserted in `tests/`. Whether that is enough to cross
into memory-bound territory is neither: it depends on the machine's
flops-per-byte ratio and on `k`, which sets how much of the work sits in the
projection. `tests/` asserts the direction, not the verdict.

![latency](figures/02_latency.png)

---

## Real-time results

`(40, 40, 40, 128)` block, 66 MB, k=16, single CPU core, 7 independent trials.
Deadline: **128 ms**.

| method | min | median | max | verdict | CNR |
|---|---:|---:|---:|---|---:|
| exact SVD | 344 | 928 | 1714 | **always misses** | 5.13 dB |
| subsample 5% | 101 | 103 | 111 | **always meets** | 5.14 dB |
| streaming (chunked) | 92 | 100 | 198 | **borderline** | 5.14 dB |

![real-time](figures/03_realtime.png)

Three things worth reading off that table.

**The exact solve is not close.** Its best trial is 2.7× over the deadline. This
is not a tuning problem.

**Approximation costs nothing measurable in image quality here.** 5.13 dB against
5.14 dB, for a speedup that changes the answer from "misses" to "meets". That
result does not generalise — see finding 7, where the same approximation costs
over a dB once the subset is too small for the rank being estimated.

**Median latency is the wrong statistic for a hard deadline.** Streaming has the
better median (100 ms against 103 ms) and a far worse tail (198 ms against
111 ms). A system that meets its deadline nine times in ten does not meet its
deadline. On this evidence the plain subsampled path is the one to deploy,
despite being marginally slower on average.

---

## Seven things that went wrong

These cost the most time and are the most useful part of the exercise.

**1. The projection was written against the wrong subspace.**
Removing clutter by keeping its complement, `X·V_k·V_kᴴ`, uses `nt − k` columns
out of `nt`. Since V is unitary, subtracting the clutter projection
`X − (X·V_c)·V_cᴴ` gives an identical result from `k` columns. Before the fix,
voxel subsampling barely helped — it was shrinking a term that was not the
bottleneck. The identity is asserted in `tests/`.

**2. I reported a 4.5× speedup that was measurement noise.**
An early benchmark showed the chunked path beating the unchunked one 99 ms to
447 ms with identical output, and I wrote it up as evidence that memory layout
matters as much as algorithm choice. Repeating the measurement seven times showed
the 447 ms figure was an allocator artefact. Run-to-run variation on this single
shared core reaches 5×, and even the machine calibration is unstable — the
streaming-copy bandwidth measured 10.1 GB/s in one session and 21.7 GB/s in
another. The *ordering* is stable across every trial; the ratios are not, and
this README no longer claims them.

**3. Doppler velocity was assigned without regard to vessel orientation.**
Doppler measures only the beam-axis component, so a vessel running perpendicular
to the beam has `v_axial ≈ 0` and is invisible to any filter, however good. The
tree originally assigned a random speed to every branch regardless of direction
— harmless in a plane, where orientations can be chosen to all carry signal, and
physically wrong in a volume. The simulator now projects onto the beam axis and
reports the resulting blind-zone fraction (7% of vessels in 3D, 10% in 2D at the
default seed). A metric that ignores that zone flatters every method equally.

**4. Clipping the vessel endpoint silently rotated the vessel.**
The first version of the fix above took the cosine from the pre-clip growth
direction, so the stored geometry and the stored velocity disagreed. The second
version clamped endpoints to the volume bounds, which pinned the depth
coordinate once the tree reached the far wall and drove the reported blind zone
to 77% with a median Doppler shift of 0 Hz. Solving for the exit parameter along
the ray — truncating the segment instead of moving its endpoint — preserves the
direction exactly. A test that recomputes the angle from `end − start` caught
both versions.

**5. There are three subspaces, not two.**
Tissue sits near DC, blood at intermediate Doppler frequencies, and the noise
floor highest of all: a white sequence has a flat spectrum, so its power-weighted
mean `|f|` is `PRF/4`. A "cut at the largest jump in the curve" rule finds the
blood-to-noise transition rather than the tissue-to-blood one. Thresholding at a
fraction of `PRF/4` instead is scale-free. Separately, spectral leakage from the
strong DC component inverted the frequency curve entirely on short ensembles
until a Hann window was applied.

**6. The automatic cutoff undershoots on small volumes, and the guards do not catch it.**
The rule returns k=6 on a 28³ volume where k=16 is much better. The Casorati
aspect ratio there is 171:1, so the existing guard has nothing to complain about
— the ratio is fine and the rule is still wrong. Pinned as a failing-behaviour
test rather than tuned away, because fitting a constant per volume size would
hide a real failure mode.

**7. The subsample floor should depend on the clutter rank, and does not.**
Estimating a 16-dimensional subspace needs far more samples than a 6-dimensional
one. Relative error against the exact solve on a 28³ volume:

| subset | k=6 | k=16 |
|---|---:|---:|
| 10 × nt rows (the floor) | 0.22 | 0.61 |
| 86 × nt rows | 0.05 | 0.19 |

At k=6 a 5% subsample cost nothing; at k=16 it costs over a dB of CNR. The floor
is left rank-independent because choosing the right scaling is a question this
study did not answer — but the dependence is real, and `fraction` should be
raised when the clutter rank is high rather than trusted at its default.

**Bonus.** Relative Frobenius error is a poor proxy for image quality. At the
aspect-ratio floor with k=6 the subsampled filter differs from the exact one by
22% and moves CNR by less than 0.5 dB. Judging the approximation by its complex
error alone would have rejected a method that is visually indistinguishable.
Both are measured separately in `tests/`.

---

## The learned filter, trained — and it loses

`models.py` holds a PyTorch unrolling that was never run, which is a claim rather
than a result. `learned.py` closes that: strip the convolutional gate, keep four
scalars per layer, and a four-layer filter has sixteen parameters — few enough to
train by Powell's method on one CPU core, no GPU required. Those sixteen scalars
are exactly the threshold and step schedule the classical solver fixes by hand.

Training works. On held-out blocks, against its own initialisation:

| | NMSE | power-Doppler term | loss |
|---|---:|---:|---:|
| classical schedule | 0.999 | 2.854 | 2.426 |
| after training | 0.835 | 1.145 | **1.407** |

A 42% improvement that generalises from training to validation. And it does not
matter, because the baseline is elsewhere entirely:

| method | NMSE | CNR | PD corr | ms |
|---|---:|---:|---:|---:|
| learned, trained (4 layers) | 0.835 | −7.62 dB | 0.548 | 79.1 |
| learned, classical init | 0.999 | −12.41 dB | −0.017 | 63.2 |
| **SVD k=8** | **0.624** | **+4.84 dB** | **0.893** | **4.7** |
| SVD k=16 | 0.936 | +5.51 dB | 0.860 | 4.7 |
| oracle | 0.000 | +3.41 dB | 1.000 | — |

**SVD wins on every metric and is 17× faster.** The learned filter is not
competitive, and the honest conclusion is that the deficit is in the model class
rather than in the optimiser. Three reasons, in the order I believe them:

1. Robust PCA assumes the blood component is *sparse*. That holds for injected
   microbubbles and does not hold for diffuse non-contrast blood, which is the
   regime measured here. The prior is wrong for the data.
2. Four layers is nowhere near convergence for an iteration that needs forty.
3. Stripping the spatial gate removed the one thing a learned filter can express
   that a global SVD cutoff cannot — a threshold that varies across the field of
   view. What is left is a worse solver for the same problem.

![learned vs SVD](figures/04_learned_vs_svd.png)

Reported rather than buried. A negative result from an experiment that was
actually run is worth more than an untrained model with an optimistic paragraph
attached, and `tests/` pins the comparison so it cannot quietly change.

---

## Time-to-detection: the temporal question

One clutter-filtered block gives one power Doppler volume, and at realistic SNR
the smallest vessels sit inside its noise. The standard remedy is to accumulate
several consecutive blocks before deciding what is vessel. That converts a
spatial question into a temporal one with a cost that matters in theatre: each
block takes `nt/PRF` seconds to acquire, so **time-to-detection**, not
latency-per-block, is what decides whether a method is usable.

24³ volume, 64-frame ensembles, SNR −15 dB, threshold calibrated to the true
vessel fraction at every depth:

| blocks | seconds | thin (r≤1.5) | medium | thick |
|---:|---:|---:|---:|---:|
| 1 | 0.06 | 0.400 | 0.459 | 0.377 |
| 4 | 0.26 | 0.440 | 0.699 | 0.574 |
| 8 | 0.51 | 0.720 | 0.801 | 0.705 |
| 16 | 1.02 | 0.800 | 0.887 | 0.823 |

Time to 80% recall: **medium vessels 0.51 s, thin and thick 1.02 s.** A single
"how many blocks do we need" number does not exist — the answer depends on which
vessels you care about.

![temporal accumulation](figures/05_temporal_accumulation.png)

### Learned temporal weighting does not work, and could not

Uniform averaging is only optimal when every block carries the same noise level,
so learning eight weights instead looks like an easy win. Fitting them gains 4.5
points of thin-vessel recall — on the sequences they were fitted to.

On three held-out sequences the gain is **exactly 0.000**.

The reason is structural rather than a training failure. This simulator draws
every block from the same distribution: identical signal, independent
identically-distributed noise. The blocks are *exchangeable*, uniform weighting
is optimal in expectation, and what the optimiser fitted was the noise
realisation of two particular sequences. Temporal weighting can only earn its
place when blocks are **not** exchangeable — probe drift, changing tissue motion,
contrast washout. Adding that non-stationarity is the experiment this points at,
and it has not been done here.

---

## Designing the acquisition that would produce real data

The job this project is aimed at asks for someone who can *design and execute
experiments to obtain suitable training data*. A simulator produces training
data; it does not design an experiment. `acquisition.py` does the design half and
says plainly that it cannot do the execution half.

Four constraints, each of which has already caused a failure in this repository:

```
Nyquist               f_d < PRF/2, so the fastest vessel sets a floor on PRF
frequency resolution  Hann main lobe ~ 4·PRF/nt must clear the tissue-blood gap
Casorati aspect       n_voxels/nt below ~10:1 and the subspace is unestimable
real-time deadline    nt/PRF seconds to acquire, and the filter must fit inside
```

They pull against each other, so the planner reports which one binds rather than
returning a number:

```
f0 15.6 MHz, PRF 3044 Hz, 601 frames, 64000 voxels
  Nyquist velocity      75.0 mm/s (need 50.0)
  frequency resolution  20 Hz (slowest vessel at 41 Hz)
  Casorati aspect       106:1
  block size            308 MB
  deadline              197 ms
  status                OK
```

and it rejects plans that look reasonable and are not — a 500 Hz frame rate folds
a 50 mm/s vessel to a twelfth of its velocity, silently.

`PHANTOM_PROTOCOL` in the same module specifies the three paired acquisitions
that construct ground truth without manual labelling (static / flow-only /
combined), the four variables worth sweeping, and the verification step: (1) plus
(2) must reconstruct (3) to within the noise floor, or the pairing is broken and
nothing downstream can repair it.

---

## What this project does not close

Three gaps remain, and no amount of code closes them:

| gap | why it stays open |
|---|---|
| **Clinical data** | Needs data access and ethics approval, not a repository |
| **Real brain ultrasound** | Needs a scanner and a subject. Everything here is simulated |
| **Multimodal clinical imaging** | Needs co-registered MRI/fUS/ESM from an operating theatre |
| **Non-stationary temporal data** | Needs real drift and motion, not i.i.d. simulated blocks |

These are not weaknesses of the implementation. They are the reason the PhD
exists. Loaders and an acquisition design are the honest limit of what can be
built without a lab, and claiming more would fall apart in the first five minutes
of a technical conversation.

---

## The streaming path

Once the clutter subspace is known, the projection is **row-independent**: each
voxel's time series is filtered using only itself and the shared basis. So the
block never needs to be resident in full.

```python
basis, _ = estimate_subspace(block, cutoff, fraction=0.05)   # 5% of voxels
filtered = apply_projection_streaming(block, basis, chunk_voxels=4096)
```

Peak extra memory is `chunk_voxels × nt` regardless of block size. In a real
pipeline the chunks would arrive from the beamformer and nothing in the loop
would change.

Chunk size turned out not to need tuning: latency is flat within ~10% from 64 to
4096 voxels per chunk and degrades only once the working set stops fitting in
cache. I had expected a sharp interior optimum and documented a plateau instead.

---

## Limitations

- **Simulated data only.** The simulator models phase modulation, decorrelation,
  Doppler angle and shift, but not aberration, out-of-plane motion, or a real
  transducer response. A physical simulator such as
  [MUST/PyMUST](https://www.biomecardio.com/MUST/) would be more defensible.
- **The learned filter is trained and loses.** See the section above. The
  PyTorch version in `models.py` — with the spatial gate that might have made the
  difference — is still untrained and unclaimed.
- **No real data of any kind.** No clinical acquisition, no in vivo brain
  ultrasound, no multimodal imaging. The physical experiment in
  `acquisition.py` is a design, not something that was run.
- **Single-core CPU timings, and noisy ones.** Real deployment is GPU, where the
  roofline balance and therefore the conclusions would shift. The analytic cost
  model is the part that transfers; the wall-clock numbers are not.
- **Modest volumes.** 40³ × 128 is 66 MB. A clinical 4D block is one to two
  orders of magnitude larger, and the extrapolated ceiling on this machine
  (~43³ at 128 frames on one core) says so plainly.
- Only the diffuse (non-contrast) regime is exercised in the headline table.
  `--contrast` switches the simulator to microbubbles, where robust PCA becomes
  appropriate and CNR stops being a sound ranking metric.

## References

- Demené et al., *IEEE TMI* 2015 — spatiotemporal clutter filtering by SVD
- Baranger et al., *IEEE TMI* 2018 — adaptive SVD threshold selection
- Lin, Chen & Ma, 2010 — inexact ALM for principal component pursuit
- Solomon et al., *IEEE TMI* 2020 — CORONA, deep unfolded robust PCA
- Halko, Martinsson & Tropp, *SIAM Review* 2011 — randomized matrix decompositions
- Williams, Waterman & Patterson, *CACM* 2009 — the roofline model
- Heiles et al., *Nat. Biomed. Eng.* 2022 — the PALA benchmark
