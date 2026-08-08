"""
Designing the acquisition that would produce real training data.

The job description asks for someone who can *design and execute experiments to
obtain suitable training data*. A simulator produces training data, but it does
not design an experiment, and the difference matters: a real acquisition has
parameters that constrain each other, and picking them wrongly produces data that
looks fine and is unusable.

This module does the design half. It cannot do the execution half -- that needs a
scanner, a phantom and a lab -- and nothing here pretends otherwise.

Four constraints, and every one of them has bitten this project already:

**Nyquist.** A Doppler shift above ``PRF/2`` folds. The fastest vessel you intend
to measure sets a floor on the frame rate.

**Frequency resolution.** Separating tissue near DC from blood needs the spectral
main lobe narrower than the gap between them. With a Hann window that is roughly
``4 * PRF / nt``, which sets a floor on the ensemble length.

**Casorati aspect ratio.** The temporal singular vectors are estimated from an
``n_voxels x nt`` matrix and become unreliable below about 10:1. That sets a
floor on the field of view for a given ensemble length -- measured failure:
a cutoff of 4 where 21 was right, and a 25 dB CNR loss with nothing raised.

**The real-time deadline.** Acquiring ``nt`` frames at ``PRF`` takes ``nt / PRF``
seconds, and the filter has to finish inside that.

The four pull against each other. Raising ``nt`` buys frequency resolution and
aspect ratio and costs deadline. Raising ``PRF`` buys Nyquist headroom and costs
frequency resolution at fixed ``nt``. :func:`plan` reports which constraint binds
rather than silently returning a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_SOUND = 1540.0


@dataclass
class AcquisitionPlan:
    """A candidate acquisition, with every constraint checked."""

    f0: float  # Hz, transducer centre frequency
    prf: float  # Hz, compounded volume rate
    nt: int  # frames per Doppler ensemble
    n_voxels: int
    v_max: float  # m/s, fastest axial velocity to be measured
    v_min: float  # m/s, slowest vessel that must stay visible

    @property
    def wavelength(self) -> float:
        return C_SOUND / self.f0

    def doppler_hz(self, v: float) -> float:
        return 2.0 * v * self.f0 / C_SOUND

    @property
    def nyquist_velocity(self) -> float:
        """Fastest unaliased axial velocity, in m/s."""
        return self.prf * C_SOUND / (4.0 * self.f0)

    @property
    def frequency_resolution_hz(self) -> float:
        """Hann main-lobe width."""
        return 4.0 * self.prf / self.nt

    @property
    def aspect_ratio(self) -> float:
        return self.n_voxels / self.nt

    @property
    def deadline_ms(self) -> float:
        return 1000.0 * self.nt / self.prf

    @property
    def block_mbytes(self) -> float:
        return self.n_voxels * self.nt * 8 / 1e6

    def check(self) -> list[str]:
        """Return a list of violated constraints. Empty means the plan is sound."""
        problems = []

        if self.v_max >= self.nyquist_velocity:
            problems.append(
                f"aliasing: v_max {self.v_max * 1e3:.1f} mm/s exceeds the Nyquist "
                f"velocity {self.nyquist_velocity * 1e3:.1f} mm/s; raise PRF or lower f0"
            )

        slow_hz = self.doppler_hz(self.v_min)
        if slow_hz < self.frequency_resolution_hz:
            problems.append(
                f"resolution: the slowest vessel sits at {slow_hz:.0f} Hz, inside the "
                f"{self.frequency_resolution_hz:.0f} Hz main lobe; lengthen the ensemble"
            )

        if self.aspect_ratio < 10.0:
            problems.append(
                f"aspect ratio {self.aspect_ratio:.1f}:1 is below 10:1; the temporal "
                f"singular vectors will not be estimable. Widen the field of view "
                f"or shorten the ensemble"
            )

        return problems

    def describe(self) -> str:
        lines = [
            f"f0 {self.f0 / 1e6:.1f} MHz, PRF {self.prf:.0f} Hz, {self.nt} frames, "
            f"{self.n_voxels} voxels",
            f"  Nyquist velocity      {self.nyquist_velocity * 1e3:.1f} mm/s "
            f"(need {self.v_max * 1e3:.1f})",
            f"  frequency resolution  {self.frequency_resolution_hz:.0f} Hz "
            f"(slowest vessel at {self.doppler_hz(self.v_min):.0f} Hz)",
            f"  Casorati aspect       {self.aspect_ratio:.0f}:1",
            f"  block size            {self.block_mbytes:.0f} MB",
            f"  deadline              {self.deadline_ms:.0f} ms",
        ]
        problems = self.check()
        lines.append("  status                " + ("OK" if not problems else "INFEASIBLE"))
        lines.extend(f"    - {p}" for p in problems)
        return "\n".join(lines)


def minimum_prf(f0: float, v_max: float, margin: float = 1.5) -> float:
    """Lowest frame rate that keeps ``v_max`` unaliased, with headroom."""
    return margin * 4.0 * v_max * f0 / C_SOUND


def minimum_frames(prf: float, f0: float, v_min: float, margin: float = 2.0) -> int:
    """Shortest ensemble that resolves the slowest vessel from DC.

    Requires the Hann main lobe to be ``margin`` times narrower than the slowest
    Doppler shift being separated.
    """
    slow_hz = 2.0 * v_min * f0 / C_SOUND
    return int(np.ceil(margin * 4.0 * prf / max(slow_hz, 1e-9)))


def plan(
    f0: float = 15.625e6,
    v_max: float = 0.05,
    v_min: float = 2e-3,
    n_voxels: int = 64000,
    prf: float | None = None,
    nt: int | None = None,
) -> AcquisitionPlan:
    """Derive an acquisition from the velocities you need to measure.

    ``PRF`` comes from the fastest vessel, then the ensemble length comes from
    the slowest. Pass either explicitly to override.
    """
    prf = minimum_prf(f0, v_max) if prf is None else prf
    nt = minimum_frames(prf, f0, v_min) if nt is None else nt
    return AcquisitionPlan(f0, prf, int(nt), n_voxels, v_max, v_min)


# --------------------------------------------------------------------------- #
# the physical experiment this design implies
# --------------------------------------------------------------------------- #
PHANTOM_PROTOCOL = """
Physical experiment to obtain training data (design only -- not executed)

The problem this solves: clutter filtering has no manual ground truth. Nobody can
label a voxel as blood. So the ground truth has to be *constructed* by an
acquisition that isolates one component at a time.

Three acquisitions, in this order:

1. STATIC, PUMP OFF
   Tissue-mimicking medium only, flow channel filled but not moving.
   Gives: the clutter component alone, with the exact speckle realisation that
   will be present in acquisition 3.

2. FLOW ONLY
   Blood-mimicking fluid circulating through a channel in a low-scattering
   surround (water or degassed gel).
   Gives: the blood component alone, at a known, pump-controlled velocity.

3. COMBINED
   The full phantom, pump on, same probe position and same settings.
   Gives: the mixture, whose ground-truth split is (1) and (2).

The controlled variables, and why each one earns its place:

  pump rate        sweeps the Doppler axis; must stay under the Nyquist
                   velocity computed above
  channel angle    sweeps the Doppler angle. A vessel perpendicular to the beam
                   is invisible to any filter -- the blind zone has to be
                   measured, not assumed away
  probe motion     a translation stage reproduces the pulsatility that makes the
                   clutter subspace higher rank than a static phantom suggests
  scatterer conc.  sets the tissue-to-blood ratio, which is the single parameter
                   that decides how hard the problem is

What would make the result untrustworthy:

  - moving the probe between acquisitions 1 and 3, which destroys the pairing
  - changing gain or focus between acquisitions, same reason
  - a channel so large that no vessel in the set is near the resolution limit,
    which is exactly where filters differ
  - reporting only the mean over the sweep, which hides the blind zone

Verification before the data is used for anything: acquisition 1 plus acquisition
2 should reconstruct acquisition 3 to within the noise floor. If it does not, the
pairing is broken and no amount of modelling downstream will fix it.
"""


def print_protocol() -> None:
    print(PHANTOM_PROTOCOL)


if __name__ == "__main__":
    print("A plan derived from the velocities to be measured:\n")
    print(plan().describe())

    print("\nA plan that looks reasonable and is not:\n")
    bad = AcquisitionPlan(
        f0=15.625e6, prf=500.0, nt=64, n_voxels=4000, v_max=0.05, v_min=2e-3
    )
    print(bad.describe())

    print_protocol()
