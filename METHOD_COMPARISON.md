# Δ⁺(1232) analysis — old vs. new reconstruction method

Two methods for reconstructing the hadronic invariant mass **W**, run on the same
BL4S-2020 data (CERN PS/SPS). The goal is a Breit–Wigner bump at **W = 1.232 GeV**
(the Δ⁺ mass) in the tungsten target, absent in the controls.

Compare the committed plots pairwise, e.g. `Runs/Plots/W_1mm_W.png` (old) vs
`Runs/Plots/W_1mm_W_matcher.png` (new).

## Old method — `bl4s_nff.py` (baseline)

Seeded straight-track fit per arm (upstream: M0/CMSPixel/M1/M2; downstream: M3/M4/M5),
1–2 hit selection, target-intersection pairing cut, φ cut, unbinned BW-over-linear-background
likelihood-ratio test. **One best track per arm per event.**

- **1mm W: 2.47σ** at W = 1.232 GeV  ·  0.5mm W: ~0σ  ·  2cm H₂O: ~0σ

## New method — `matcher.py` (arm re-pairing)

Adds a **second candidate track per arm** (re-runs the *same* tracker with candidate-1's
hits excluded) and a **best-pair match between the two arms** by target-intersection distance.
This corrects events where the two arms' best tracks belonged to *different* pile-up particles
— a mismatch the old single-best-track method could not catch.

Built by *wrapping* the baseline tracker, not reimplementing it: the `bl4s_nff` hooks
(`return_idx`, `exclude`, `max_hits`) are all default-off, so with matching disabled the
pipeline reproduces the baseline **byte-for-byte**.

- **1mm W: 2.73σ** at W = 1.232 GeV  ·  0.5mm W: 0.0σ  ·  2cm H₂O: 0.0σ

## Validation

- **Phase-A gate:** matching off reproduces the baseline exactly (2.47 / 0.01 / 0.00).
- **Freeze-sample test:** on the *same* event set, re-pairing raises 1mm W 2.47 → 2.73σ while
  both controls stay ~0σ. The gain is a genuine resolution improvement, not an artefact of the
  self-referential φ cut (which had inflated a naïve end-to-end run to 2.89σ).

## Avenues tested and rejected

- **Relaxing the 1–2 hit cap** (to recover pile-up statistics): the recovered busy events carry
  *zero* signal (they are unrelated pile-up / secondaries), so they dilute rather than help
  (2.9 → 0.6 → 0.0σ at caps 2 → 3 → 4). The cap is physically correct — Δ⁺ production is a clean,
  low-multiplicity scatter.
- **MM-through-magnet hit matching:** the two-chamber + 150 cm-magnet geometry gives only
  ~1.5 cm matching resolution (thin-lens sagitta ≈ L·φ/4), too coarse to pick the right hit;
  net-*negative* on signal resolution.

## Bottom line (Part I)

A **~2.7σ hint** at the correct Δ⁺ mass in 1mm W, **absent in both controls** — below the 3σ
evidence threshold. The analysis is apparatus-limited (2 MM chambers, one dipole), not
software-limited.

*Superseded by Part II below: the hint is explained by radiative background.*

---

# Part II — resolution, validated statistics, and the origin of the excess

The Part-I significances came from fitting a **bare Breit–Wigner** (Γ = 117 MeV) over a
linear background, with Wilks' theorem for the p-value. Both assumptions were then tested.

## W resolution: σ_W = 295 MeV = 2.5 × Γ  (`sigma_w_propagation.py`)

The instrumental angular resolutions, measured on the 600A no-target run (σ_φ = 4.61 mrad,
σ_θx/y = 0.94/0.39 mrad), propagate through the W formula to **σ_W ≈ 295 MeV** at the Δ peak
(analytic derivatives and MC smearing agree). Key facts:

- σ_W is **100% φ-driven** (σ_φ/φ ≈ 11% momentum resolution on p′). The θ term contributes
  < 1 MeV, so tracking/matching improvements cannot sharpen W — the MM/φ arm is the only lever.
- A real Δ⁺ would appear as a **Voigt profile with FWHM ≈ 760 MeV** — wider than the whole
  [1.0, 1.7] fit window. The bare 117-MeV BW template physically cannot describe real signal.
- `signal_pdf(W, sigma_res)` (BW ⊗ Gaussian) was added to `bl4s_nff.py` and
  `unbinned_lrt(W, sigma_res=…)` fits it.

## Toy-MC validation (`matcher_stats.py`, `run_10k_toys.py`)

Wilks' χ²(1) is unreliable here (f_s on its boundary; smeared template nearly degenerate with
the linear background), so every p-value was re-derived with background-only pseudo-experiments
through the identical LRT. With 10k toys, 1mm W:

| fit | Wilks | toys |
|---|---|---|
| old selection, smeared template | 2.96σ | 2.78σ (p = 0.0027) |
| re-pairing, bare BW | 2.73σ | 2.72σ (p = 0.0033) |
| re-pairing, smeared | 2.50σ | 2.54σ (p = 0.0056) |

Stable at 2.5–2.8σ across method/template variants; controls stay at zero everywhere.

## The excess scales like radiation length, not like protons

| Target | protons/cm² (rel. 1mm W) | t/X₀ (rel. H₂O) | excess |
|---|---|---|---|
| 2cm H₂O | 1.43 | 1.0 | 0.0σ |
| 1mm W | 1.00 | 5.1 | ~2.5σ |
| 0.5mm W | 0.50 | 2.6 | ~0.5σ |

Real Δ⁺ production predicts H₂O ≥ 1mm W > 0.5mm W (water has *more* protons per cm², a fifth
of them free hydrogen — the sharpest possible peak). The data instead follow **t/X₀** — the
bremsstrahlung scale.

## Radiative-tail background (`radiative_tail.py`)

An elastically scattered electron that radiates a collinear photon of energy ΔE anywhere in
the target reconstructs at exactly **W = √(m_p² + 2·m_p·ΔE)**; ΔE = 0.35 GeV lands on the Δ
mass and passes the φ cut. The model is fully derived, with **zero tuned shape parameters**:
Bethe–Heitler straggling (u = ln(E₀/E′) ~ Gamma(t/ln2), reproducing ⟨E′⟩ = E₀e^-t), mapped to
W and convolved with the measured σ_W. The u ≈ 0 mass doubles as the smeared elastic-peak leak
from W = 0.938, so one component models both; only its fraction floats.

With the tail in the null (and in the toys):

- **1mm W: 2.54σ → 1.0σ** (toys p = 0.16). The data actively prefer the tail (2ΔNLL = 5.5).
- 0.5mm W: 0.46σ. H₂O: 0σ (its component is the elastic leak — the t/X₀ dependence is in the
  template *shape*, which is fixed per target, not in the floated fraction).

## Upper limit (`final_result.py`, plot: `Runs/Plots/W_1mm_W_final.png`)

Profile-likelihood scan on f_s, toy-calibrated at the limit (same distrust-Wilks policy):

- **f_s < 0.67 at 90% CL** (asymptotic: 0.61), i.e. N_sig < ~990 of 1484 fit-window events.
  The limit is weak because the smeared signal is nearly degenerate with the background —
  a direct consequence of σ_W = 2.5Γ.

## Bottom line (final)

The ~2.5σ excess at W = 1.232 GeV in 1mm W is **quantitatively consistent with bremsstrahlung
energy loss in the target** — a background whose shape is fixed by t/X₀ with nothing tuned —
and its scaling across targets matches radiation length, not proton density. Residual signal
significance: **1.0σ**. **No evidence for Δ⁺(1232) electroproduction**; f_s < 0.67 at 90% CL.
The search is closed by physics understanding, not by statistics: with σ_W = 2.5Γ (set by the
2-chamber φ arm), this apparatus cannot distinguish a Δ⁺ line from its own radiative tail.
