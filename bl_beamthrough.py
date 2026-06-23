#!/usr/bin/env python3
"""
Beam-through BL calibration using the no-target 700A run (1601651352).

Two estimators:
  1. Angle subtraction:  phi = theta_MM_slope - theta_upstream   (reuses mag_def_angle)
  2. Position residuals: dx_i = x_MM_i - x_pred_i  ->  dtheta_i = dx_i / (z_MM_i - z_brmC)

Both estimated at 700A then scaled to 550A via linear I->B assumption.
Result is compared against BL_CALIBRATED (from target runs) and the spec prediction.
"""

import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from bl4s_nff import (
    open_tree, upstream_valid, post_BRM_valid, upstream_track,
    mag_def_angle, mag_field_calibration, BL_from_current, BL_CALIBRATED,
    zmm, zbrmc, GeV_factor, pelectron,
)
from mimosa_residuals import compute_alignment

BEAMTHROUGH_PATH = "Runs/No Target Runs/1601651352.root"
NOTARGET_ALIGN   = "Runs/No Target Runs/1601727557.root"
BEAMTHROUGH_A    = 700
TARGET_A         = 550
OUTDIR           = Path("Runs/Beamthrough")


def pos_residuals(reco, track_up, up_valid, post_valid):
    """
    Per-event kick angle from position residuals at each MM chamber.
    Uses thin-lens approximation: kick at zbrmc, so lever arm = z_MM - z_brmC.
    Returns (dtheta_mm1, dtheta_mm2) in radians.
    """
    pos1 = ak.to_numpy(ak.mean(reco["Chamber1_position"].array()[up_valid][post_valid], axis=1)) * 0.1
    pos2 = ak.to_numpy(ak.mean(reco["Chamber2_position"].array()[up_valid][post_valid], axis=1)) * 0.1

    x_pred1 = track_up.b0_upx[post_valid] + track_up.b1_upx[post_valid] * zmm[0]
    x_pred2 = track_up.b0_upx[post_valid] + track_up.b1_upx[post_valid] * zmm[1]

    dtheta1 = np.arctan((pos1 - x_pred1) / (zmm[0] - zbrmc))
    dtheta2 = np.arctan((pos2 - x_pred2) / (zmm[1] - zbrmc))
    return dtheta1, dtheta2


def main():
    align_x, align_y = compute_alignment(NOTARGET_ALIGN)

    reco     = open_tree(BEAMTHROUGH_PATH)
    up_v     = upstream_valid(reco)
    post_v   = post_BRM_valid(reco, up_v)
    track_up = upstream_track(reco, up_v, align_x, align_y)

    # Method 1: angle subtraction
    phi      = mag_def_angle(reco, track_up, up_v, post_v)
    BL_ang   = mag_field_calibration(phi)
    BL_ang_550 = BL_ang * (TARGET_A / BEAMTHROUGH_A)

    # Method 2: position residuals
    dth1, dth2      = pos_residuals(reco, track_up, up_v, post_v)
    BL_res1         = mag_field_calibration(dth1)
    BL_res2         = mag_field_calibration(dth2)
    BL_res1_550     = BL_res1 * (TARGET_A / BEAMTHROUGH_A)
    BL_res2_550     = BL_res2 * (TARGET_A / BEAMTHROUGH_A)
    BL_res_avg_550  = np.mean([BL_res1_550, BL_res2_550])

    BL_spec_700 = BL_from_current(BEAMTHROUGH_A)
    BL_spec_550 = BL_from_current(TARGET_A)

    # --- Summary ---
    print(f"\n{'='*65}")
    print(f"  Beam-through BL calibration  ({BEAMTHROUGH_A}A, 4.0 GeV, no target)")
    print(f"  Events passing upstream + MM cuts: {post_v.sum():,}")
    print(f"{'='*65}")
    print(f"  {'Method':<32} {'BL @700A (T·m)':>14} {'BL @550A (T·m)':>14}")
    print(f"  {'-'*60}")
    print(f"  {'Angle subtraction':<32} {BL_ang:>14.5f} {BL_ang_550:>14.5f}")
    print(f"  {'Pos. residual MM1':<32} {BL_res1:>14.5f} {BL_res1_550:>14.5f}")
    print(f"  {'Pos. residual MM2':<32} {BL_res2:>14.5f} {BL_res2_550:>14.5f}")
    print(f"  {'Pos. residual (avg)':<32} {'—':>14} {BL_res_avg_550:>14.5f}")
    print(f"  {'-'*60}")
    print(f"  {'Spec (BL_from_current)':<32} {BL_spec_700:>14.5f} {BL_spec_550:>14.5f}")
    print(f"  {'Target-run calibration':<32} {'—':>14} {BL_CALIBRATED:>14.5f}")
    print(f"  {'-'*60}")
    print(f"  Spec / beam-through ratio    : {BL_spec_550 / BL_ang_550:.1f}x")
    print(f"  Beam-through vs target-calib : "
          f"{abs(BL_ang_550 - BL_CALIBRATED)/BL_CALIBRATED*100:.1f}% difference")
    print(f"  MM1 vs MM2 consistency       : "
          f"{abs(BL_res1_550 - BL_res2_550)/np.mean([BL_res1_550, BL_res2_550])*100:.1f}% difference")
    print(f"{'='*65}\n")

    # --- Plots ---
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Beam-through BL calibration  —  {BEAMTHROUGH_A}A no-target, scaled to {TARGET_A}A\n"
        f"Angle: {BL_ang_550:.4f} T·m  |  Res. avg: {BL_res_avg_550:.4f} T·m  |  "
        f"Target-calib: {BL_CALIBRATED:.4f} T·m  |  Spec: {BL_spec_550:.4f} T·m",
        fontsize=9,
    )

    # Panel 1: angle-subtraction phi distribution
    ax = axes[0]
    phi_mrad = phi * 1e3
    med = np.nanmedian(phi_mrad)
    std = np.nanstd(phi_mrad)
    ax.hist(phi_mrad, bins=80, color="steelblue", alpha=0.8)
    ax.axvline(med, color="red", lw=2, label=f"median = {med:.2f} mrad\nstd = {std:.2f} mrad")
    ax.set_xlabel("φ  [mrad]")
    ax.set_ylabel("Events / bin")
    ax.set_title("Angle subtraction: φ = θ_MM − θ_up")
    ax.legend(fontsize=9)

    # Panel 2: position residuals — MM1 vs MM2
    ax = axes[1]
    med1, med2 = np.nanmedian(dth1) * 1e3, np.nanmedian(dth2) * 1e3
    ax.hist(dth1 * 1e3, bins=80, color="steelblue", alpha=0.6,
            label=f"MM1  (z={zmm[0]:.0f} cm)")
    ax.hist(dth2 * 1e3, bins=80, color="darkorange", alpha=0.6,
            label=f"MM2  (z={zmm[1]:.0f} cm)")
    ax.axvline(med1, color="steelblue", lw=2, ls="--", label=f"MM1 median = {med1:.2f} mrad")
    ax.axvline(med2, color="darkorange", lw=2, ls="--", label=f"MM2 median = {med2:.2f} mrad")
    ax.set_xlabel("Δθ_residual  [mrad]")
    ax.set_ylabel("Events / bin")
    ax.set_title("Position residuals: Δx / (z_MM − z_brmC)")
    ax.legend(fontsize=8)

    # Panel 3: BL comparison at 550A
    ax = axes[2]
    labels = ["Angle\n(beam-thru)", "Res. MM1\n(beam-thru)", "Res. MM2\n(beam-thru)",
              "Target\nruns", "Spec"]
    values = [BL_ang_550, BL_res1_550, BL_res2_550, BL_CALIBRATED, BL_spec_550]
    colors = ["steelblue", "steelblue", "steelblue", "darkorange", "grey"]
    alphas = [1.0, 0.7, 0.5, 1.0, 1.0]
    bars = ax.bar(labels, values, color=colors)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    ax.set_ylabel("B·L  [T·m]  at 550A")
    ax.set_title("BL at 550A — method comparison")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = OUTDIR / "bl_beamthrough_calibration.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
