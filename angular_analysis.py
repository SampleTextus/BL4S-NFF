#!/usr/bin/env python3
"""
Angular distribution analysis for NFF BL4S-2020.

Compares the scattering angle (upstream vs downstream MIMOSA tracks) between
target and no-target runs. Does not require BRM momentum measurement.

Steps:
  1. Compute theta for each run using MIMOSA telescope only
  2. Normalise by number of upstream-valid events (exposure)
  3. Overlay and ratio plots (target / no-target)
  4. Chi-square test for shape compatibility
  5. Kinematic map: E_e' vs theta for elastic and Delta+(1232)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2 as chi2_dist

from bl4s_nff import open_tree, upstream_valid, downstream_valid
from bl4s_nff import upstream_track, downstream_track
from mimosa_residuals import compute_alignment

# ── Physical constants ────────────────────────────────────────────────────────
E_BEAM   = 4.0       # GeV (corrected)
M_PROTON = 0.938272  # GeV
M_DELTA  = 1.232     # GeV
M_PION   = 0.13957   # GeV

# ── Runs ──────────────────────────────────────────────────────────────────────
NOTARGET_PATH    = "Runs/No Target Runs/1601727557.root"
NOTARGET_ALIGN   = "Runs/No Target Runs/1601727557.root"  # same run used for alignment

# 4 GeV / 550A — main physics runs
TARGET_RUNS = [
    ("Runs/Target Runs/1601730086.root", "1mm W (4 GeV)",    "steelblue"),
    ("Runs/Target Runs/1601742188.root", "0.5mm W (4 GeV)",  "darkorange"),
    ("Runs/Target Runs/1601747831.root", "2cm H2O (4 GeV)",  "tomato"),
]

# 3 GeV runs at different BRM currents — separate group for comparison
TARGET_RUNS_3GEV = [
    ("Runs/Target Runs/1601736173.root", "1mm W (3 GeV, 430A)", "mediumpurple"),
    ("Runs/Target Runs/1601737081.root", "1mm W (3 GeV, 370A)", "mediumseagreen"),
]

OUTDIR = Path("Runs")

# ── Kinematics ────────────────────────────────────────────────────────────────

def e_prime(W, theta):
    """Scattered electron energy (GeV) for given invariant mass W and angle theta (rad)."""
    num = 2 * M_PROTON * E_BEAM + M_PROTON**2 - W**2
    den = 2 * (M_PROTON + E_BEAM * (1 - np.cos(theta)))
    return num / den


# ── Core computation ──────────────────────────────────────────────────────────

def get_scattering_angles(path, align_x, align_y):
    """
    Return (theta_rad, n_upstream_valid) for a run.
    theta is the total scattering angle for events with valid tracks on both arms.
    n_upstream_valid is used as the exposure normalisation.
    """
    reco  = open_tree(path)
    up_v  = upstream_valid(reco)
    dn_v  = downstream_valid(reco)

    trk_up = upstream_track(reco, up_v, align_x, align_y)
    trk_dn = downstream_track(reco, dn_v, align_x, align_y)

    both       = up_v & dn_v
    up_in_both = both[up_v]
    dn_in_both = both[dn_v]

    theta_x = (np.arctan(trk_dn.b1_dnx[dn_in_both])
             - np.arctan(trk_up.b1_upx[up_in_both]))
    theta_y = (np.arctan(trk_dn.b1_dny[dn_in_both])
             - np.arctan(trk_up.b1_upy[up_in_both]))
    theta   = np.sqrt(theta_x**2 + theta_y**2)

    return theta, up_v.sum()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_distributions(theta_nt, n_up_nt, target_data, sigma_beam, outdir):
    """
    Two-panel plot:
      Top   — shape comparison: each histogram normalised to its own total
              (shows whether the angular distribution is shifted)
      Bottom — ratio of normalised shapes (target / no-target per bin)
    """
    bins    = np.linspace(0, 60, 61)   # 1 mrad bins, 0-60 mrad
    centres = (bins[:-1] + bins[1:]) / 2

    h_nt      = np.histogram(theta_nt * 1e3, bins=bins)[0]
    h_nt_norm = h_nt / h_nt.sum()    # shape: fraction per bin

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 10),
                                          gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"Scattering angle distributions  (E_beam = {E_BEAM} GeV)", fontsize=13)

    # ── top panel: shape comparison ───────────────────────────────────────────
    ax_top.step(centres, h_nt_norm, where="mid",
                color="black", lw=1.5,
                label=f"No target  ({len(theta_nt):,} events, N_up = {n_up_nt:,})")

    for theta, n_up, label, colour in target_data:
        h      = np.histogram(theta * 1e3, bins=bins)[0]
        h_norm = h / h.sum()
        n_both = len(theta)
        rate   = n_both / n_up
        ax_top.step(centres, h_norm, where="mid", color=colour, lw=1.2,
                    label=f"{label}  ({n_both:,} events, track rate = {rate:.1%})")

    ax_top.axvspan(0, sigma_beam, alpha=0.12, color="grey",
                   label=f"Beam divergence  (1σ = {sigma_beam:.1f} mrad)")
    ax_top.set_yscale("log")
    ax_top.set_xlim(0, 60)
    ax_top.set_xlabel("θ  [mrad]")
    ax_top.set_ylabel("Fraction of events per bin  (shape normalised)")
    ax_top.legend(fontsize=9)

    # ── bottom panel: shape ratio ─────────────────────────────────────────────
    ax_bot.axhline(1.0, color="black", lw=1.2, ls="--", label="Same shape as no-target")
    ax_bot.axvspan(0, sigma_beam, alpha=0.12, color="grey")

    for theta, n_up, label, colour in target_data:
        h      = np.histogram(theta * 1e3, bins=bins)[0]
        h_norm = h / h.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(h_nt > 5, h_norm / h_nt_norm, np.nan)
        ax_bot.plot(centres, ratio, "o-", color=colour, ms=3, lw=1, label=label)

    ax_bot.set_xlim(0, 60)
    ax_bot.set_ylim(0, 5)
    ax_bot.set_xlabel("θ  [mrad]")
    ax_bot.set_ylabel("Shape ratio  (target / no-target)")
    ax_bot.legend(fontsize=9)

    plt.tight_layout()
    out = outdir / "angular_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_kinematic_map(sigma_beam, outdir):
    """
    E_e' vs theta for elastic scattering, Delta+(1232), and pion threshold.
    Shows which angle range corresponds to which physics.
    """
    theta_arr = np.linspace(1, 80, 400) * 1e-3  # rad

    ep_elastic = e_prime(M_PROTON, theta_arr)
    ep_delta   = e_prime(M_DELTA,  theta_arr)
    ep_piN     = e_prime(M_PROTON + M_PION, theta_arr)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(theta_arr * 1e3, ep_elastic, "b-",  lw=2, label=f"Elastic  (W = {M_PROTON:.3f} GeV)")
    ax.plot(theta_arr * 1e3, ep_delta,   "r--", lw=2, label=f"Δ+(1232)  (W = {M_DELTA:.3f} GeV)")
    ax.plot(theta_arr * 1e3, ep_piN,     "g:",  lw=2,
            label=f"π production threshold  (W = {M_PROTON + M_PION:.3f} GeV)")

    ax.axhline(E_BEAM,  color="grey", lw=0.8, ls=":", label=f"E_beam = {E_BEAM} GeV")
    ax.axhline(0,       color="grey", lw=0.8, ls=":")
    ax.axvspan(0, sigma_beam, alpha=0.12, color="grey", label=f"Beam divergence (1σ)")

    # Mark where Delta+ E_e' separates from elastic by > 100 MeV (roughly resolvable)
    separation = ep_elastic - ep_delta
    idx = np.where(separation > 0.1)[0]
    if len(idx):
        theta_sep = theta_arr[idx[0]] * 1e3
        ax.axvline(theta_sep, color="red", lw=1, ls=":",
                   label=f"Elastic/Δ+ separation > 100 MeV  (θ > {theta_sep:.1f} mrad)")

    ax.set_xlabel("θ_e  [mrad]")
    ax.set_ylabel("E_e'  [GeV]")
    ax.set_title(f"Kinematic map: scattered electron energy vs angle  (E_beam = {E_BEAM} GeV)")
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.1, E_BEAM + 0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = outdir / "kinematic_map.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Chi-square test ───────────────────────────────────────────────────────────

def chi2_test(theta_nt, n_up_nt, theta_tgt, n_up_tgt, sigma_beam, label):
    """
    Shape chi-square test: are the normalised angular distributions compatible?
    Both histograms normalised to unit area before comparison.
    Only bins above the beam divergence region with enough counts are used.
    """
    bins    = np.linspace(0, 60, 61)
    centres = (bins[:-1] + bins[1:]) / 2

    h_nt  = np.histogram(theta_nt  * 1e3, bins=bins)[0]
    h_tgt = np.histogram(theta_tgt * 1e3, bins=bins)[0]

    # Scale no-target shape to target total (shape comparison)
    scale    = h_tgt.sum() / h_nt.sum()
    expected = h_nt * scale

    # Only bins above beam divergence, with expected > 5
    use = (centres > sigma_beam) & (expected > 5)
    if use.sum() < 3:
        print(f"  {label}: too few usable bins")
        return

    chi2_val = float(np.sum((h_tgt[use] - expected[use])**2 / expected[use]))
    ndf      = int(use.sum()) - 1   # -1 for the scale free parameter
    p        = 1 - chi2_dist.cdf(chi2_val, ndf)
    excess   = float((h_tgt[use] - expected[use]).sum())
    rate_nt  = len(theta_nt)  / n_up_nt
    rate_tgt = len(theta_tgt) / n_up_tgt

    print(f"  {label}:")
    print(f"    downstream track rate:  no-target = {rate_nt:.1%},  target = {rate_tgt:.1%}  "
          f"(delta = {rate_tgt - rate_nt:+.1%})")
    print(f"    shape chi2/ndf = {chi2_val:.1f}/{ndf},  p = {p:.4f},  "
          f"shape excess = {excess:+.0f} events")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    align_x, align_y = compute_alignment(NOTARGET_PATH)

    # No-target baseline
    print("Loading no-target run...")
    theta_nt, n_up_nt = get_scattering_angles(NOTARGET_PATH, align_x, align_y)
    sigma_beam = float(np.nanstd(theta_nt) * 1e3)   # mrad
    print(f"  upstream-valid: {n_up_nt:,}  |  both-arm events: {len(theta_nt):,}")
    print(f"  theta: mean = {np.nanmean(theta_nt)*1e3:.2f} mrad,  "
          f"std = {sigma_beam:.2f} mrad  (beam divergence)\n")

    # 4 GeV target runs
    target_data = []
    for path, label, colour in TARGET_RUNS:
        print(f"Loading {label}...")
        theta, n_up = get_scattering_angles(path, align_x, align_y)
        target_data.append((theta, n_up, label, colour))
        print(f"  upstream-valid: {n_up:,}  |  both-arm events: {len(theta):,}")
        print(f"  theta: mean = {np.nanmean(theta)*1e3:.2f} mrad,  "
              f"std = {np.nanstd(theta)*1e3:.2f} mrad\n")

    # 3 GeV target runs (separate group)
    target_data_3gev = []
    for path, label, colour in TARGET_RUNS_3GEV:
        print(f"Loading {label}...")
        theta, n_up = get_scattering_angles(path, align_x, align_y)
        target_data_3gev.append((theta, n_up, label, colour))
        print(f"  upstream-valid: {n_up:,}  |  both-arm events: {len(theta):,}")
        print(f"  theta: mean = {np.nanmean(theta)*1e3:.2f} mrad,  "
              f"std = {np.nanstd(theta)*1e3:.2f} mrad\n")

    # Plots — 4 GeV runs
    plot_distributions(theta_nt, n_up_nt, target_data, sigma_beam, OUTDIR)
    plot_kinematic_map(sigma_beam, OUTDIR)

    # Plots — all runs combined (4 GeV + 3 GeV), saved separately
    all_outdir = OUTDIR / "all_runs"
    all_outdir.mkdir(exist_ok=True)
    plot_distributions(theta_nt, n_up_nt, target_data + target_data_3gev,
                       sigma_beam, all_outdir)

    # Chi-square tests
    print(f"Chi-square test vs no-target baseline  (theta > {sigma_beam:.1f} mrad, expected > 5):")
    print("  -- 4 GeV runs --")
    for theta, n_up, label, _ in target_data:
        chi2_test(theta_nt, n_up_nt, theta, n_up, sigma_beam, label)
    print("  -- 3 GeV runs --")
    for theta, n_up, label, _ in target_data_3gev:
        chi2_test(theta_nt, n_up_nt, theta, n_up, sigma_beam, label)


if __name__ == "__main__":
    main()
