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

## Bottom line

A **~2.7σ hint** at the correct Δ⁺ mass in 1mm W, **absent in both controls** — below the 3σ
evidence threshold. The analysis is apparatus-limited (2 MM chambers, one dipole), not
software-limited.
