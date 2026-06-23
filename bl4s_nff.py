#!/usr/bin/env python3
import sys
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist


# Z Positions in cm
zmimosa = np.array([35.0, 38.6, 41.4, 48.5, 51.5, 53.65])
ztarget = 44.95 # estimated as midpoint of MIMOSA positions
zbrmc = 211.5
zbrmin = zbrmc - 75 # since length of BRM is 150 cm
zbrmout = zbrmc + 75
zdwc = 339.75
zmm = np.array([384.5, 515.0])
zbrmlen = 150.0

# Masses in GeV
mproton = 938.272*pow(10,-3)
melectron = 0.511*pow(10,-3)
mpion = 139.570*pow(10,-3)
mdelta = 1232.0*pow(10,-3)

# Useful constants
nrg_beam = 4.0 # GeV
pelectron = np.sqrt(nrg_beam**2 - melectron**2) #GeV/c
GeV_factor = 0.299792 # for dealing with magnetic field in GeV
h_bar = 6.582*pow(10,-25) #GeVs
delta_lifetime = 5.6*pow(10,-24) #s
Gamma = h_bar/delta_lifetime #for BW distro


# Access data from file reco branch if valid and readable 
def open_tree(path):
    f = uproot.open(path)
    reco = f["RECOdata"]
    readable = []
    for branch in reco.keys():
        try:
            arr = reco[branch].array(entry_stop=1)
            if len(arr) > 0:
                readable.append(branch)
        except Exception as e:
            print(f"Warning: could not read branch {branch}: {e}")
    if not readable:
        raise RuntimeError(f"No readable branches found in {path}")
    return reco

# Isolate valid runs. Check if mimosas were hit. Gives an array of booleans for each event
# Since a particle is apoint it can hit 2 x positons at once and 2 y at once, so bound it
def upstream_valid(reco):
    nx0 = reco["NMimosa0_xPosition"].array(library="np")
    ny0 = reco["NMimosa0_yPosition"].array(library="np")
    nx1 = reco["NMimosa1_xPosition"].array(library="np")
    ny1 = reco["NMimosa1_yPosition"].array(library="np")
    nx2 = reco["NMimosa2_xPosition"].array(library="np")
    ny2 = reco["NMimosa2_yPosition"].array(library="np")
    return (nx0 >= 1) & (nx0 <= 2) & (ny0 >= 1) & (ny0 <= 2) & \
           (nx1 >= 1) & (nx1 <= 2) & (ny1 >= 1) & (ny1 <= 2) & \
           (nx2 >= 1) & (nx2 <= 2) & (ny2 >= 1) & (ny2 <= 2)

def downstream_valid(reco):
    nx3 = reco["NMimosa3_xPosition"].array(library="np")
    ny3 = reco["NMimosa3_yPosition"].array(library="np")
    nx4 = reco["NMimosa4_xPosition"].array(library="np")
    ny4 = reco["NMimosa4_yPosition"].array(library="np")
    nx5 = reco["NMimosa5_xPosition"].array(library="np")
    ny5 = reco["NMimosa5_yPosition"].array(library="np")
    return (nx3 >= 1) & (nx3 <= 2) & (ny3 >= 1) & (ny3 <= 2) & \
           (nx4 >= 1) & (nx4 <= 2) & (ny4 >= 1) & (ny4 <= 2) & \
           (nx5 >= 1) & (nx5 <= 2) & (ny5 >= 1) & (ny5 <= 2)

#print(f"proportion of events passing upstream cut: {upstream_valid(open_tree(path)).sum()} / {len(upstream_valid(open_tree(path)))}") #% of events passing upstream cut
#print(f"proportion of events with valid downstream: {downstream_valid(open_tree(path)).sum()} / {len(downstream_valid(open_tree(path)))}") #% of events passing downstream cut

# Track construction: run an ols on the upstram detectors
# Since the positions are jagged, flatten first
def upstream_track(reco, up_valid, align_x=None, align_y=None):
    z_up = zmimosa[:3]
    x_up = 0.1 * np.stack([ak.to_numpy(ak.mean(reco["Mimosa0_xPosition"].array()[up_valid], axis=1)),
                     ak.to_numpy(ak.mean(reco["Mimosa1_xPosition"].array()[up_valid], axis=1)),
                     ak.to_numpy(ak.mean(reco["Mimosa2_xPosition"].array()[up_valid], axis=1))])
    y_up = 0.1 * np.stack([ak.to_numpy(ak.mean(reco["Mimosa0_yPosition"].array()[up_valid], axis=1)),
                     ak.to_numpy(ak.mean(reco["Mimosa1_yPosition"].array()[up_valid], axis=1)),
                     ak.to_numpy(ak.mean(reco["Mimosa2_yPosition"].array()[up_valid], axis=1))])
    if align_x is not None:
        x_up = x_up - align_x[:3, np.newaxis]
    if align_y is not None:
        y_up = y_up - align_y[:3, np.newaxis]
    b1_upx, b0_upx = np.polyfit(z_up, x_up, 1)
    b1_upy, b0_upy = np.polyfit(z_up, y_up, 1)
    track_up = namedtuple("track_up", ["b1_upx", "b0_upx", "b1_upy", "b0_upy"])
    return track_up(b1_upx, b0_upx, b1_upy, b0_upy)

def downstream_track(reco, dn_valid, align_x=None, align_y=None):
    z_dn = zmimosa[3:]
    x_dn = 0.1 * np.stack([ak.to_numpy(ak.mean(reco["Mimosa3_xPosition"].array()[dn_valid], axis=1)),
                     ak.to_numpy(ak.mean(reco["Mimosa4_xPosition"].array()[dn_valid], axis=1)),
                     ak.to_numpy(ak.mean(reco["Mimosa5_xPosition"].array()[dn_valid], axis=1))])
    y_dn = 0.1 * np.stack([ak.to_numpy(ak.mean(reco["Mimosa3_yPosition"].array()[dn_valid], axis=1)),
                     ak.to_numpy(ak.mean(reco["Mimosa4_yPosition"].array()[dn_valid], axis=1)),
                     ak.to_numpy(ak.mean(reco["Mimosa5_yPosition"].array()[dn_valid], axis=1))])
    if align_x is not None:
        x_dn = x_dn - align_x[3:, np.newaxis]
    if align_y is not None:
        y_dn = y_dn - align_y[3:, np.newaxis]
    b1_dnx, b0_dnx = np.polyfit(z_dn, x_dn, 1)
    b1_dny, b0_dny = np.polyfit(z_dn, y_dn, 1)
    track_dn = namedtuple("track_dn", ["b1_dnx", "b0_dnx", "b1_dny", "b0_dny"])
    return track_dn(b1_dnx, b0_dnx, b1_dny, b0_dny)


# Since we only need the partial derivatives, we use the slopes
def scatter_angle(b1_upx, b1_upy, b1_dnx, b1_dny):
    theta_x = np.arctan(b1_dnx) - np.arctan(b1_upx)
    theta_y = np.arctan(b1_dny) - np.arctan(b1_upy)
    return np.sqrt(theta_x**2 + theta_y**2)

def post_BRM_valid(reco, up_valid):
    nmm1 = reco["NChamber1_position"].array(library="np")[up_valid]
    nmm2 = reco["NChamber2_position"].array(library="np")[up_valid]
    return (nmm1 >= 1) & (nmm1 <= 2) & (nmm2 >= 1) & (nmm2 <= 2)


# Magnetic deflection: compare MM slope to upstream slope.
# Upstream is used as reference rather than downstream: the downstream track is fit over only
# ~5 cm (Mimosa 3-4-5), giving a noisier slope that smears phi and hence W more than the
# small scatter-angle contamination it would remove. Upstream phi is lower-noise in practice.
def mag_def_angle(reco, track_up, up_valid, post_valid):
    pos1 = ak.to_numpy(ak.mean(reco["Chamber1_position"].array()[up_valid][post_valid], axis=1)) * 0.1
    pos2 = ak.to_numpy(ak.mean(reco["Chamber2_position"].array()[up_valid][post_valid], axis=1)) * 0.1
    detected_pos = np.stack([pos1, pos2])
    b1_detected, _ = np.polyfit(zmm, detected_pos, 1)
    return np.arctan(b1_detected) - np.arctan(track_up.b1_upx[post_valid])

# B*L from magnet spec: DESY type MD, integrated field length 1 m, max 1.35 T at 1400 A
def BL_from_current(current_A):
    return (current_A / 1400.0) * 1.35 * 1.0  # T*m

# B*L from angle subtraction on the 700A no-target beam-through run (bl_beamthrough.py), scaled to 550A.
# Angle subtraction is preferred over position residuals: it compares upstream slope to MM slope
# directly and makes no thin-lens assumption about where the kick occurs inside the magnet.
BL_CALIBRATED = 0.07509  # T*m

# Data-driven B*L from a single run's phi distribution (diagnostic use)
def mag_field_calibration(phi):
    return np.sin(np.nanmedian(phi)) * pelectron / GeV_factor

def momentum_track(phi):
    return GeV_factor * BL_CALIBRATED / np.sin(phi)

# Recovering the mass of proton post collision
# Use Lorentz invariance in terms of masses and energies
# Look at momenta of concerned particles before and after
def inv_mass_rec(track_up, track_dn, phi, up_valid, dn_valid, post_valid):
    both = up_valid & dn_valid
    up_in_both = both[up_valid]    # mask into track_up arrays
    dn_in_both = both[dn_valid]    # mask into track_dn arrays

    post_in_both = post_valid[up_in_both]
    pelectron_new = momentum_track(phi)[up_in_both[post_valid]]
    theta = scatter_angle(
        track_up.b1_upx[up_in_both][post_in_both],
        track_up.b1_upy[up_in_both][post_in_both],
        track_dn.b1_dnx[dn_in_both][post_in_both],
        track_dn.b1_dny[dn_in_both][post_in_both],
    )
    W = np.sqrt((mproton + np.sqrt(melectron**2 + pelectron**2)
                    - np.sqrt(melectron**2 + pelectron_new**2))**2
                    - (pelectron**2 + pelectron_new**2 - 2*pelectron*pelectron_new*np.cos(theta)))
    finite = np.isfinite(W)
    return W[finite], theta[finite]

# Plotting histogram with events and recovered masses (display only)
def plot_inv_mass(W):
    plt.figure()
    plt.hist(W, bins=50, range=(0.5, 2))
    plt.axvline(mdelta, color='r', linestyle='--', label=f'$\\Delta^+(1232)$ = {mdelta:.3f} GeV')
    plt.xlabel('W (GeV/c^2)')
    plt.ylabel('Events')
    plt.legend()
    plt.show()

def plot_phi(phi, label, outpath):
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(phi * 1e3, bins=60, color="steelblue", alpha=0.8)
    ax.axvline(np.nanmedian(phi) * 1e3, color='r', ls='--',
               label=f'median = {np.nanmedian(phi)*1e3:.2f} mrad')
    ax.set_xlabel("φ  [mrad]")
    ax.set_ylabel("Events / bin")
    ax.set_title(f"Magnetic deflection angle — {label}")
    ax.legend()
    plt.tight_layout()
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

def plot_theta(theta, label, outpath):
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(theta * 1e3, bins=60, color="darkorange", alpha=0.8)
    ax.axvline(np.nanmedian(theta) * 1e3, color='r', ls='--',
               label=f'median = {np.nanmedian(theta)*1e3:.2f} mrad')
    ax.set_xlabel("θ  [mrad]")
    ax.set_ylabel("Events / bin")
    ax.set_title(f"Scattering angle — {label}")
    ax.legend()
    plt.tight_layout()
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

def plot_W_distribution(W, label, outpath, chi2_results=None):
    """Save W distribution with BW+bg fit overlay to outpath."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")

    W_plot = W[(W > 0.5) & (W < 2.0)]
    bins   = np.linspace(1.0, 1.7, 36)   # same as Chi_sq
    counts, edges = np.histogram(W[(W > 1.0) & (W < 1.7)], bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    errors  = np.sqrt(np.maximum(counts, 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Hadronic invariant mass W  —  {label}\n"
                 f"B*L = {BL_CALIBRATED:.5f} T*m (calibrated),  "
                 f"E_beam = {nrg_beam:.1f} GeV,  N_W = {len(W):,}", fontsize=11)

    # Left: full range
    ax = axes[0]
    ax.hist(W_plot, bins=40, range=(0.5, 2.0), color="steelblue", alpha=0.8)
    ax.axvline(mdelta,   color="red",  lw=2, ls="--", label=f"$\\Delta^+$(1232) = {mdelta:.3f} GeV")
    ax.axvline(mproton,  color="grey", lw=1, ls=":",  label=f"p = {mproton:.3f} GeV")
    ax.set_xlabel("W  [GeV/c$^2$]")
    ax.set_ylabel("Events / bin")
    ax.set_title("Full range (0.5 – 2.0 GeV)")
    ax.legend(fontsize=9)

    # Right: fit range with BW overlay if fit available
    ax2 = axes[1]
    ax2.errorbar(centres, counts, yerr=errors, fmt="o", color="steelblue",
                 ms=4, label="Data")
    ax2.axvline(mdelta,  color="red",  lw=1.5, ls="--", label=f"$\\Delta^+$ PDG")
    ax2.axvline(mproton, color="grey", lw=1,   ls=":",  label=f"p threshold")

    if chi2_results is not None:
        (chi2_bg, df_bg, popt_bg), (chi2_sig, df_sig, popt_sig) = chi2_results
        W_dense = np.linspace(1.0, 1.7, 400)

        bg_line  = popt_bg[0] + popt_bg[1] * (W_dense - mdelta)
        bw_total = BW_plus_bg(W_dense, *popt_sig)
        bw_bg    = popt_sig[3] + popt_sig[4] * (W_dense - mdelta) if len(popt_sig) > 4 \
                   else popt_sig[1] + popt_sig[2] * (W_dense - mdelta)

        ax2.plot(W_dense, bg_line,  "k--", lw=1.5, label=f"BG only  chi2/df={chi2_bg:.1f}/{df_bg}")
        ax2.plot(W_dense, bw_total, "r-",  lw=2,
                 label=f"BW+BG  chi2/df={chi2_sig:.1f}/{df_sig},  A={popt_sig[0]:.3f}")

    ax2.set_xlabel("W  [GeV/c$^2$]")
    ax2.set_ylabel("Events / bin")
    ax2.set_title("Fit range (1.0 – 1.7 GeV)")
    ax2.set_xlim(1.0, 1.7)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# Relativistic Breit-Wigner distribution: Due to Heisenberg uncert. princip. mass has a spread.
def BW_distro(W, A):
    return A*W*mdelta*Gamma/((W**2 - mdelta**2)**2 + (mdelta*Gamma)**2)

# BW signal on top of a linear background (non-resonant scattering)
def BW_plus_bg(W, A, a0, a1):
    return BW_distro(W, A) + a0 + a1*(W - mdelta)

# We want masses >1 since above the pion production threshold, <1.7 to account for tails
# We fit both a pure background and BW+background to compare
# df = bins used - free parameters
def Chi_sq(W):
    W_fit = W[(W>1) & (W<1.7)]
    bins = np.linspace(1, 1.7, 36)
    counts, edges = np.histogram(W_fit, bins)
    centers = (edges[:-1]+edges[1:])/2
    error = np.sqrt(np.maximum(counts,1)) #to avoid div0
    use = counts > 0

    # Fit 1: background only (linear), 2 free params
    def bg_only(W, a0, a1):
        return a0 + a1*(W - mdelta)
    popt_bg, _ = curve_fit(bg_only, centers[use], counts[use],
                           p0=[counts.mean(), 0.0],
                           sigma=error[use], absolute_sigma=True)
    expected_bg = bg_only(centers, *popt_bg)
    chi2_bg = np.sum(((counts[use] - expected_bg[use]) / error[use])**2)
    df_bg = use.sum() - 2

    # Fit 2: BW + background, 3 free params
    popt_sig, _ = curve_fit(BW_plus_bg, centers[use], counts[use],
                            p0=[counts.max(), counts.mean(), 0.0],
                            sigma=error[use], absolute_sigma=True,
                            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
    expected_sig = BW_plus_bg(centers, *popt_sig)
    chi2_sig = np.sum(((counts[use] - expected_sig[use]) / error[use])**2)
    df_sig = use.sum() - 3

    return (chi2_bg, df_bg, popt_bg), (chi2_sig, df_sig, popt_sig)

NOTARGET_PATH = "Runs/No Target Runs/1601727557.root"

TARGET_RUNS = [
    ("Runs/Target Runs/1601730086.root", "1mm W",   550),
    ("Runs/Target Runs/1601742188.root", "0.5mm W", 550),
    ("Runs/Target Runs/1601747831.root", "2cm H2O", 550),
]

def run_analysis(reco, label, align_x, align_y, current_A, outdir="Runs/Target Runs"):
    from pathlib import Path
    up_valid   = upstream_valid(reco)
    dn_valid   = downstream_valid(reco)
    post_valid = post_BRM_valid(reco, up_valid)

    track_up = upstream_track(reco, up_valid, align_x, align_y)
    track_dn = downstream_track(reco, dn_valid, align_x, align_y)

    phi      = mag_def_angle(reco, track_up, up_valid, post_valid)
    BL_run   = mag_field_calibration(phi)

    print(f"\n{'='*55}")
    print(f"  {label}  (BRM {current_A}A)")
    print(f"{'='*55}")
    print(f"  B*L calibrated : {BL_CALIBRATED:.5f} T*m  (run estimate: {BL_run:.5f} T*m)")
    print(f"  Post-BRM events: {post_valid.sum():,}")
    print(f"  phi median/std : {np.nanmedian(phi)*1e3:.3f} / {np.nanstd(phi)*1e3:.3f} mrad")

    W, theta = inv_mass_rec(track_up, track_dn, phi, up_valid, dn_valid, post_valid)
    print(f"  Valid W values : {len(W):,}")

    slug = label.replace(' ', '_').replace('/', '')
    plot_phi(phi,     label, Path(outdir) / f"phi_{slug}.png")
    plot_theta(theta, label, Path(outdir) / f"theta_{slug}.png")

    if len(W[(W > 1) & (W < 1.7)]) < 10:
        print("  Too few events in fit range for chi-square test")
        plot_W_distribution(W, label, Path(outdir) / f"W_{slug}.png")
        return

    chi2_results = Chi_sq(W)
    (chi2_bg, df_bg, popt_bg), (chi2_sig, df_sig, popt_sig) = chi2_results
    p_bg  = 1 - chi2_dist.cdf(chi2_bg,  df_bg)
    p_sig = 1 - chi2_dist.cdf(chi2_sig, df_sig)

    print(f"\n  Background only : chi2/df = {chi2_bg:.2f}/{df_bg},  p = {p_bg:.4f}")
    print(f"  BW + background : chi2/df = {chi2_sig:.2f}/{df_sig},  p = {p_sig:.4f}")
    print(f"  BW amplitude    : {popt_sig[0]:.4f}")

    plot_W_distribution(W, label, Path(outdir) / f"W_{slug}.png",
                        chi2_results=chi2_results)

def main():
    from mimosa_residuals import compute_alignment
    align_x, align_y = compute_alignment(NOTARGET_PATH)
    print(f"Using B*L = {BL_CALIBRATED:.5f} T*m  (calibrated from data, 4 GeV/550A runs)")

    for path, label, current_A in TARGET_RUNS:
        reco = open_tree(path)
        run_analysis(reco, label, align_x, align_y, current_A)

if __name__ == "__main__":
    main()