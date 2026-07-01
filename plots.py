#!/usr/bin/env python3
"""
Generate all analysis and alignment plots into Runs/Plots/.
Run this script to produce every figure; run bl4s_nff.py for numerical results only.
"""

import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from bl4s_nff import (
    open_tree, upstream_valid, downstream_valid,
    BW_plus_bg, BL_CALIBRATED, nrg_beam, melectron, mdelta, mproton,
    GeV_factor, BL_from_current, run_analysis, phi_sanity_check,
    compute_phi_reference, compute_target_pair_reference, PHI_NSIGMA, TARGET_PAIR_NSIGMA,
    NOTARGET_600A, TARGET_RUNS, zmimosa, zmm, zcmspixel,
)
from alignment import (
    compute_alignment, compute_tilt_alignment,
    compute_mm_alignment_track, compute_cmspixel_alignment,
    ALIGN_PATH,
)

OUTDIR = Path("Runs/Plots")


# ── Alignment plots ────────────────────────────────────────────────────────────

def plot_prealignment(raw_x, raw_y, raw_cms_x=None, raw_cms_y=None):
    n_sc   = min(2000, raw_x.shape[1])
    n_cols = 6 if raw_cms_x is not None else 5
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    fig.suptitle(
        "Pre-alignment: detector vs M0 raw cluster position (no correction applied)\n"
        "Tight diagonal cigar = correlated hits from same particle; offset from diagonal = misalignment",
        fontsize=10,
    )
    # Mimosa 1-5
    for col, i in enumerate(range(1, 6)):
        for row, (xi, x0, coord, color) in enumerate([
            (raw_x[i], raw_x[0], "x", "steelblue"),
            (raw_y[i], raw_y[0], "y", "darkorange"),
        ]):
            ax = axes[row, col]
            ax.scatter(x0[:n_sc], xi[:n_sc], s=1, alpha=0.3, color=color)
            lo, hi = min(x0.min(), xi.min()), max(x0.max(), xi.max())
            ax.plot([lo, hi], [lo, hi], "r-", lw=1, alpha=0.5)
            dx = np.mean(xi) - np.mean(x0)
            ax.set_xlabel(f"M0 {coord} [cm]", fontsize=8)
            ax.set_ylabel(f"M{i} {coord} [cm]", fontsize=8)
            ax.set_title(f"M{i} vs M0 ({coord}),  D = {dx*10:.2f} mm", fontsize=8)
    # CMSPixel
    if raw_cms_x is not None:
        for row, (xi, x0, coord, color) in enumerate([
            (raw_cms_x, raw_x[0], "x", "steelblue"),
            (raw_cms_y, raw_y[0], "y", "darkorange"),
        ]):
            ax  = axes[row, 5]
            ax.scatter(x0[:n_sc], xi[:n_sc], s=1, alpha=0.3, color=color)
            lo  = min(x0.min(), xi.min())
            hi  = max(x0.max(), xi.max())
            ax.plot([lo, hi], [lo, hi], "r-", lw=1, alpha=0.5)
            dx  = np.mean(xi) - np.mean(x0)
            ax.set_xlabel(f"M0 {coord} [cm]", fontsize=8)
            ax.set_ylabel(f"CMS {coord} [cm]", fontsize=8)
            ax.set_title(f"CMS vs M0 ({coord}),  D = {dx*10:.2f} mm", fontsize=8)
    plt.tight_layout()
    out = OUTDIR / "prealignment_scatter.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved: {out}")


def plot_mimosa_alignment(raw_x, raw_y, align_x, align_y,
                           raw_cms_x=None, raw_cms_y=None, cms_align=None):
    cx = raw_x - align_x[:, np.newaxis]
    cy = raw_y - align_y[:, np.newaxis]
    coeff_x = np.polyfit(zmimosa, cx, 1)   # (2, n_events)
    coeff_y = np.polyfit(zmimosa, cy, 1)

    n_cols = 7 if raw_cms_x is not None else 6
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 6))
    fig.suptitle(
        "Translational alignment (0A run)  --  hit minus 6-plane track prediction\n"
        "Mimosa planes 0-5 (self-consistent); CMSPixel (independent check via Mimosa track interpolation)",
        fontsize=10,
    )
    # Mimosa 0-5
    for i in range(6):
        res_x = cx[i] - (coeff_x[0] * zmimosa[i] + coeff_x[1])
        res_y = cy[i] - (coeff_y[0] * zmimosa[i] + coeff_y[1])
        for row, res, off, ylabel, color in [
            (0, res_x, align_x[i], "track residual x [cm]", "steelblue"),
            (1, res_y, align_y[i], "track residual y [cm]", "darkorange"),
        ]:
            ax = axes[row, i]
            ax.hist(res, bins=60, color=color, alpha=0.8)
            ax.axvline(0, color="red", lw=2, ls="--", label=f"offset = {off:.4f} cm")
            ax.set_title(f"M{i}", fontsize=9)
            ax.set_xlabel(ylabel, fontsize=8)
            ax.set_ylabel("Events / bin", fontsize=8)
            ax.legend(fontsize=7)
    # CMSPixel
    if raw_cms_x is not None and cms_align is not None:
        x_pred_cms = coeff_x[0] * zcmspixel + coeff_x[1]
        y_pred_cms = coeff_y[0] * zcmspixel + coeff_y[1]
        res_x_cms  = (raw_cms_x - cms_align[0]) - x_pred_cms
        res_y_cms  = (raw_cms_y - cms_align[1]) - y_pred_cms
        for row, res, off, ylabel, color in [
            (0, res_x_cms, cms_align[0], "track residual x [cm]", "steelblue"),
            (1, res_y_cms, cms_align[1], "track residual y [cm]", "darkorange"),
        ]:
            ax = axes[row, 6]
            ax.hist(res, bins=60, color=color, alpha=0.8)
            ax.axvline(0, color="red", lw=2, ls="--", label=f"offset = {off:.4f} cm")
            ax.set_title("CMS (z=36.9 cm)", fontsize=9)
            ax.set_xlabel(ylabel, fontsize=8)
            ax.set_ylabel("Events / bin", fontsize=8)
            ax.legend(fontsize=7)
    plt.tight_layout()
    out = OUTDIR / "mimosa_alignment.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved: {out}")


def plot_track_residual_validation(raw_x, raw_y, align_x, align_y,
                                    raw_cms_x=None, raw_cms_y=None, cms_align=None):
    planes_x_aln = raw_x - align_x[:, np.newaxis]
    planes_y_aln = raw_y - align_y[:, np.newaxis]
    coeff_x = np.polyfit(zmimosa, planes_x_aln, 1)
    coeff_y = np.polyfit(zmimosa, planes_y_aln, 1)

    n_cols = 7 if raw_cms_x is not None else 6
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 6))
    fig.suptitle(
        "Track-residual validation: 6-plane OLS residuals after translational alignment\n"
        "Mimosa mean ~ 0 (self-check); CMSPixel is independent prediction via Mimosa track interpolation",
        fontsize=10,
    )
    for i in range(6):
        for row, planes_aln, coeff, ylabel, color in [
            (0, planes_x_aln, coeff_x, "res_x [cm]", "steelblue"),
            (1, planes_y_aln, coeff_y, "res_y [cm]", "darkorange"),
        ]:
            pred = coeff[0] * zmimosa[i] + coeff[1]
            res  = planes_aln[i] - pred
            ax   = axes[row, i]
            ax.hist(res, bins=50, color=color, alpha=0.8)
            ax.axvline(np.mean(res), color="red", lw=2,
                       label=f"mean = {np.mean(res)*1e4:.1f}e-4 cm")
            ax.set_title(f"M{i}", fontsize=9)
            ax.set_xlabel(ylabel, fontsize=8)
            ax.set_ylabel("Events / bin", fontsize=8)
            ax.legend(fontsize=7)
    if raw_cms_x is not None and cms_align is not None:
        x_pred_cms = coeff_x[0] * zcmspixel + coeff_x[1]
        y_pred_cms = coeff_y[0] * zcmspixel + coeff_y[1]
        res_x_cms  = (raw_cms_x - cms_align[0]) - x_pred_cms
        res_y_cms  = (raw_cms_y - cms_align[1]) - y_pred_cms
        for row, res, ylabel, color in [
            (0, res_x_cms, "res_x [cm]", "steelblue"),
            (1, res_y_cms, "res_y [cm]", "darkorange"),
        ]:
            ax = axes[row, 6]
            ax.hist(res, bins=50, color=color, alpha=0.8)
            ax.axvline(np.mean(res), color="red", lw=2,
                       label=f"mean = {np.mean(res)*1e4:.1f}e-4 cm")
            ax.set_title("CMS (independent)", fontsize=9)
            ax.set_xlabel(ylabel, fontsize=8)
            ax.set_ylabel("Events / bin", fontsize=8)
            ax.legend(fontsize=7)
    plt.tight_layout()
    out = OUTDIR / "track_residual_validation.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved: {out}")


def plot_tilt_diagnostic(raw_x, raw_y, align_x, align_y, rot_x, rot_y,
                          raw_cms_x=None, raw_cms_y=None, cms_align=None, cms_rot=None):
    planes_x_aln = raw_x - align_x[:, np.newaxis]
    planes_y_aln = raw_y - align_y[:, np.newaxis]
    n_sc = min(500, raw_x.shape[1])

    n_cols = 7 if raw_cms_x is not None else 6
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))
    fig.suptitle(
        "In-plane rotation diagnostic (0A run)\n"
        "Mimosa: leave-one-out fit.  CMSPixel: full 6-plane tilt-corrected Mimosa track as reference.\n"
        "top: res_x vs y_pred  [slope = rot_x]     bottom: res_y vs x_pred  [slope = rot_y]",
        fontsize=9,
    )
    # Mimosa planes (leave-one-out)
    for i in range(6):
        others  = [j for j in range(6) if j != i]
        z_oth   = zmimosa[others]
        cx = np.polyfit(z_oth, planes_x_aln[others], 1)
        cy = np.polyfit(z_oth, planes_y_aln[others], 1)
        x_pred = cx[0] * zmimosa[i] + cx[1]
        y_pred = cy[0] * zmimosa[i] + cy[1]
        res_x  = planes_x_aln[i] - x_pred
        res_y  = planes_y_aln[i] - y_pred
        cx_vy  = np.polyfit(y_pred, res_x, 1)
        cy_vx  = np.polyfit(x_pred, res_y, 1)

        ax = axes[0, i]
        ax.scatter(y_pred[:n_sc], res_x[:n_sc], s=1, alpha=0.3, color="steelblue")
        yf = np.linspace(y_pred.min(), y_pred.max(), 100)
        ax.plot(yf, cx_vy[0]*yf + cx_vy[1], "r-", lw=1.5)
        ax.set_title(f"M{i}: rot_x = {rot_x[i]*1e3:.1f} mrad", fontsize=8)
        ax.set_xlabel("y_pred [cm]", fontsize=7)
        ax.set_ylabel("res_x [cm]", fontsize=7)

        ax = axes[1, i]
        ax.scatter(x_pred[:n_sc], res_y[:n_sc], s=1, alpha=0.3, color="darkorange")
        xf = np.linspace(x_pred.min(), x_pred.max(), 100)
        ax.plot(xf, cy_vx[0]*xf + cy_vx[1], "r-", lw=1.5)
        ax.set_title(f"M{i}: rot_y = {rot_y[i]*1e3:.1f} mrad", fontsize=8)
        ax.set_xlabel("x_pred [cm]", fontsize=7)
        ax.set_ylabel("res_y [cm]", fontsize=7)

    # CMSPixel: use tilt-corrected 6-plane Mimosa track as reference
    if raw_cms_x is not None and cms_align is not None and cms_rot is not None:
        x0, y0      = planes_x_aln.copy(), planes_y_aln.copy()
        px_tilt     = x0 - rot_x[:, np.newaxis] * y0
        py_tilt     = y0 - rot_y[:, np.newaxis] * x0
        cx6         = np.polyfit(zmimosa, px_tilt, 1)
        cy6         = np.polyfit(zmimosa, py_tilt, 1)
        x_pred_cms  = cx6[0] * zcmspixel + cx6[1]
        y_pred_cms  = cy6[0] * zcmspixel + cy6[1]
        res_x_cms   = (raw_cms_x - cms_align[0]) - x_pred_cms
        res_y_cms   = (raw_cms_y - cms_align[1]) - y_pred_cms
        cx_vy_cms   = np.polyfit(y_pred_cms, res_x_cms, 1)
        cy_vx_cms   = np.polyfit(x_pred_cms, res_y_cms, 1)

        ax = axes[0, 6]
        ax.scatter(y_pred_cms[:n_sc], res_x_cms[:n_sc], s=1, alpha=0.3, color="steelblue")
        yf = np.linspace(y_pred_cms.min(), y_pred_cms.max(), 100)
        ax.plot(yf, cx_vy_cms[0]*yf + cx_vy_cms[1], "r-", lw=1.5)
        ax.set_title(f"CMS: rot_x = {cms_rot[0]*1e3:.1f} mrad", fontsize=8)
        ax.set_xlabel("y_pred [cm]", fontsize=7)
        ax.set_ylabel("res_x [cm]", fontsize=7)

        ax = axes[1, 6]
        ax.scatter(x_pred_cms[:n_sc], res_y_cms[:n_sc], s=1, alpha=0.3, color="darkorange")
        xf = np.linspace(x_pred_cms.min(), x_pred_cms.max(), 100)
        ax.plot(xf, cy_vx_cms[0]*xf + cy_vx_cms[1], "r-", lw=1.5)
        ax.set_title(f"CMS: rot_y = {cms_rot[1]*1e3:.1f} mrad", fontsize=8)
        ax.set_xlabel("x_pred [cm]", fontsize=7)
        ax.set_ylabel("res_y [cm]", fontsize=7)

    plt.tight_layout()
    out = OUTDIR / "tilt_diagnostic.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved: {out}")


def plot_mm_alignment(res1, res2, mm_align):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        "MM alignment via Mimosa track extrapolation (B=0 run)\n"
        "Residual = MM_hit − extrapolated 6-plane Mimosa track, after offset subtraction\n"
        "Distributions centred at 0 confirm alignment correction",
        fontsize=10,
    )
    for ax, res, label, off in zip(axes, [res1, res2], ["MM1", "MM2"], mm_align * 10):
        ax.hist((res * 10) - off, bins=60, color="steelblue", alpha=0.8)
        ax.axvline(0, color="red", lw=2, ls="--", label=f"offset = {off:.2f} mm")
        ax.set_xlabel("MM_hit − x_track_pred  [mm]")
        ax.set_ylabel("Events / bin")
        ax.set_title(label)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = OUTDIR / "mm_alignment.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_mm_tilt(res1, res2, mm_align_new, mm_rot, yp1, yp2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        "MM in-plane rotation diagnostic: x_residual vs y_pred at MM z-positions (B=0 run)\n"
        "Slope = rotation coefficient (same structure as Mimosa tilt); values reported but not applied",
        fontsize=10,
    )
    for ax, res_c, yp, label, rot_val in zip(
        axes,
        [res1 - mm_align_new[0], res2 - mm_align_new[1]],
        [yp1, yp2],
        ["MM1", "MM2"],
        mm_rot,
    ):
        n_mm = min(500, len(yp))
        ax.scatter(yp[:n_mm], res_c[:n_mm], s=2, alpha=0.4, color="steelblue")
        yf = np.linspace(yp.min(), yp.max(), 100)
        ax.plot(yf, rot_val * yf + (np.mean(res_c) - rot_val * np.mean(yp)), "r-", lw=1.5)
        ax.set_xlabel("y_pred [cm]", fontsize=9)
        ax.set_ylabel("x_residual (corr.) [cm]", fontsize=9)
        ax.set_title(f"{label}: rot = {rot_val*1e3:.2f} mrad", fontsize=9)
    plt.tight_layout()
    out = OUTDIR / "mm_tilt_diagnostic.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  Saved: {out}")


# ── Analysis plots ─────────────────────────────────────────────────────────────

def plot_phi_sanity_check(phi, phi_pred, BL_check, current_A=600, beam_energy=4.6):
    phi_med = np.nanmedian(phi)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(phi * 1e3, bins=60, color="steelblue", alpha=0.8)
    ax.axvline(phi_med  * 1e3, color="red",   lw=2,
               label=f"Measured median = {phi_med*1e3:.2f} mrad")
    ax.axvline(phi_pred * 1e3, color="green", lw=2, ls="--",
               label=f"Predicted = {phi_pred*1e3:.2f} mrad  (BL = {BL_check:.4f} T·m)")
    ax.set_xlabel("φ  [mrad]")
    ax.set_ylabel("Events / bin")
    ax.set_title(f"Phi sanity check — {current_A}A no-target run  (E = {beam_energy} GeV)")
    ax.legend()
    plt.tight_layout()
    out = OUTDIR / "phi_sanity_check.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_phi(phi, label, slug):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(phi * 1e3, bins=60, color="steelblue", alpha=0.8)
    ax.axvline(np.nanmedian(phi) * 1e3, color="r", ls="--",
               label=f"median = {np.nanmedian(phi)*1e3:.2f} mrad")
    ax.set_xlabel("φ  [mrad]")
    ax.set_ylabel("Events / bin")
    ax.set_title(f"Magnetic deflection angle — {label}")
    ax.legend()
    plt.tight_layout()
    out = OUTDIR / f"phi_{slug}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_theta(theta, label, slug):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(theta * 1e3, bins=60, color="darkorange", alpha=0.8)
    ax.axvline(np.nanmedian(theta) * 1e3, color="r", ls="--",
               label=f"median = {np.nanmedian(theta)*1e3:.2f} mrad")
    ax.set_xlabel("θ  [mrad]")
    ax.set_ylabel("Events / bin")
    ax.set_title(f"Scattering angle — {label}")
    ax.legend()
    plt.tight_layout()
    out = OUTDIR / f"theta_{slug}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_W(W, label, slug, lrt_results=None):
    W_plot  = W[(W > 0.5) & (W < 2.0)]
    bins    = np.linspace(1.0, 1.7, 36)
    counts, edges = np.histogram(W[(W > 1.0) & (W < 1.7)], bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    errors  = np.sqrt(np.maximum(counts, 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Hadronic invariant mass W  —  {label}\n"
        f"B*L = {BL_CALIBRATED:.5f} T·m (calibrated),  "
        f"E_beam = {nrg_beam:.1f} GeV,  N_W = {len(W):,}",
        fontsize=11,
    )

    ax = axes[0]
    ax.hist(W_plot, bins=40, range=(0.5, 2.0), color="steelblue", alpha=0.8)
    ax.axvline(mdelta,  color="red",  lw=2, ls="--", label=f"$\\Delta^+$(1232) = {mdelta:.3f} GeV")
    ax.axvline(mproton, color="grey", lw=1, ls=":",  label=f"p = {mproton:.3f} GeV")
    ax.set_xlabel("W  [GeV/c$^2$]")
    ax.set_ylabel("Events / bin")
    ax.set_title("Full range (0.5 – 2.0 GeV)")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.errorbar(centres, counts, yerr=errors, fmt="o", color="steelblue", ms=4, label="Data")
    ax2.axvline(mdelta,  color="red",  lw=1.5, ls="--", label="$\\Delta^+$ PDG")
    ax2.axvline(mproton, color="grey", lw=1,   ls=":",  label="p threshold")

    if lrt_results is not None:
        W_dense = np.linspace(1.0, 1.7, 400)
        ax2.plot(W_dense, lrt_results['bg_curve'](W_dense), "k--", lw=1.5,
                 label="BG only (null)")
        ax2.plot(W_dense, lrt_results['total_curve'](W_dense), "r-", lw=2,
                 label=f"BW+BG  f_s={lrt_results['f_s']:.4f},  "
                       f"{lrt_results['significance']:.1f}σ")

    ax2.set_xlabel("W  [GeV/c$^2$]")
    ax2.set_ylabel("Events / bin")
    ax2.set_title("Fit range (1.0 – 1.7 GeV)")
    ax2.set_xlim(1.0, 1.7)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = OUTDIR / f"W_{slug}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Alignment ────────────────────────────────────────────────────────────
    print("Computing alignment...")
    align_x, align_y                         = compute_alignment()
    rot_x, rot_y                              = compute_tilt_alignment(align_x=align_x, align_y=align_y)
    mm_align, res1, res2, mm_rot, yp1, yp2  = compute_mm_alignment_track(
        align_x=align_x, align_y=align_y, rot_x=rot_x, rot_y=rot_y)
    cms_align, cms_rot                        = compute_cmspixel_alignment(
        align_x=align_x, align_y=align_y, rot_x=rot_x, rot_y=rot_y)
    pair_median, pair_sigma                   = compute_target_pair_reference(
        align_x, align_y, rot_x, rot_y, cms_align, cms_rot)
    pair_tol                                  = pair_median + TARGET_PAIR_NSIGMA * pair_sigma
    phi_ref                                   = compute_phi_reference(
        align_x, align_y, mm_align, rot_x, rot_y, cms_align, cms_rot, pair_tol=pair_tol)
    print(f"  Target-pair reference (600A): median = {pair_median*1e4:.1f} um, "
          f"sigma_robust = {pair_sigma*1e4:.1f} um -> tolerance = {pair_tol*1e4:.1f} um")
    print(f"  Phi reference (600A): center = {phi_ref[0]*1e3:.2f} mrad, "
          f"sigma_robust = {phi_ref[1]*1e3:.2f} mrad  (+/-{PHI_NSIGMA}*sigma cut)")

    reco      = open_tree(ALIGN_PATH)
    both_v    = upstream_valid(reco) & downstream_valid(reco)
    raw_x     = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_xPosition"].array()[both_v], axis=1))
        for i in range(6)
    ])
    raw_y     = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_yPosition"].array()[both_v], axis=1))
        for i in range(6)
    ])
    raw_cms_x = ak.to_numpy(ak.mean(reco["CMSPixel0_xPosition"].array()[both_v], axis=1))
    raw_cms_y = ak.to_numpy(ak.mean(reco["CMSPixel0_yPosition"].array()[both_v], axis=1))

    print("\nGenerating alignment plots...")
    plot_prealignment(raw_x, raw_y, raw_cms_x, raw_cms_y)
    plot_mimosa_alignment(raw_x, raw_y, align_x, align_y, raw_cms_x, raw_cms_y, cms_align)
    plot_track_residual_validation(raw_x, raw_y, align_x, align_y, raw_cms_x, raw_cms_y, cms_align)
    plot_tilt_diagnostic(raw_x, raw_y, align_x, align_y, rot_x, rot_y,
                         raw_cms_x, raw_cms_y, cms_align, cms_rot)
    plot_mm_alignment(res1, res2, mm_align)
    plot_mm_tilt(res1, res2, mm_align, mm_rot, yp1, yp2)

    # ── Phi sanity check ─────────────────────────────────────────────────────
    print("\nPhi sanity check...")
    phi_sc, phi_pred, BL_check = phi_sanity_check(
        align_x, align_y, mm_align, rot_x, rot_y, cms_align, cms_rot)
    plot_phi_sanity_check(phi_sc, phi_pred, BL_check)

    # ── Target run analysis ───────────────────────────────────────────────────
    for path, label, current_A in TARGET_RUNS:
        reco_t = open_tree(path)
        phi, W, theta, lrt_results = run_analysis(
            reco_t, label, align_x, align_y, current_A, mm_align, rot_x, rot_y,
            cms_align=cms_align, cms_rot=cms_rot, phi_ref=phi_ref, pair_tol=pair_tol)
        slug = label.replace(" ", "_").replace("/", "")
        plot_phi(phi,   label, slug)
        plot_theta(theta, label, slug)
        plot_W(W, label, slug, lrt_results)

    print(f"\nAll plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
