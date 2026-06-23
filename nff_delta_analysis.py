#!/usr/bin/env python3
"""
NFF (Nations Flying Foxes) BL4S-2020 — Delta+(1232) search.

Physics: electrons hit a target (Al/W/H2O); we look for events consistent
         with Delta+ production and decay.
           e- + p  ->  e- + Delta+(1232)  ->  e- + p + pi0
                                           ->  e- + n + pi+

Detector layout (z positions from Sep 29 logbook):
   MIMOSA 0  :   0.0 mm  (upstream tracking plane 1)
   CMS pixel :  19.0 mm
   MIMOSA 1  :  36.0 mm  (upstream tracking plane 2)
   MIMOSA 2  :  64.5 mm  (upstream tracking plane 3)
   --- target ---
   MIMOSA 3  : 135.0 mm  (downstream tracking plane 1)
   MIMOSA 4  : 161.0 mm  (downstream tracking plane 2)
   MIMOSA 5  : 186.5 mm  (downstream tracking plane 3)
   MM1 (Ch2) :  downstream
   MM2 (Ch3) :  downstream
   DWC0, DWC1: downstream (connected Oct 1, run 130+)

Run notes (NFF logbook):
   Sep 29 run 104  : first Al target; MM timing finally works
   Sep 29 run 105  : 5 mm tungsten target, 2 GeV e-
   Sep 29 run 117  : ~3 mm Al target, 2 GeV e-, recalibrated MM
   Oct  2 run 141  : ~4 mm Al target, 5 GeV e-, all detectors (6M+CMS+2MM+DWC)
   Oct  3 run 173  : 2 cm water target, 4.6 GeV e-, bending magnet 700 A
                     -> enables momentum measurement for invariant mass!

Usage:
   python nff_delta_analysis.py <run.root> [beam_energy_GeV]

   beam_energy_GeV defaults to 4.6 (run 173 water-target configuration).
"""

import sys
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")          # no display required
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from scipy.optimize import curve_fit
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ─── physical constants (MeV) ─────────────────────────────────────────────────
M_PROTON = 938.272
M_PION_C = 139.570   # pi+/-
M_DELTA  = 1232.0    # Delta+(1232) PDG value

# ─── detector geometry (cm from end of collimator, Sep 29 logbook) ────────────
# MIMOSA positions: RECOdata x/y are in cm  (±1.06 cm = ±10.6 mm = MIMOSA26 active area)
# Chamber positions: RECOdata is in mm      (±23 mm = Micromegas active strip range)
Z_MIMOSA = np.array([35.0, 38.6, 41.4, 48.5, 51.5, 53.65])  # cm, planes 0–5
MIMOSA_X_OFFSET = np.array([0.0, 0.00372, -0.002625, -0.01239, -0.009682, -0.005116])  # measured residuals relative to M0 [cm], x only
Z_CMS    = 36.9    # cm
Z_TARGET = 44.95   # cm  (midpoint of Mimosa2–Mimosa3 gap; actual not measured)
Z_MAG_CTR = 211.5  # cm  (bending magnet centre)
Z_MAG_IN  = 136.5  # cm  (magnet entrance: 211.5 - 75)
Z_MAG_OUT = 286.5  # cm  (magnet exit:     211.5 + 75)
Z_MAG_LEN = 150.0  # cm
Z_MM1     = 384.5  # cm
Z_MM2     = 515.0  # cm
MM_TO_CM  = 0.1    # Chamber branches are in mm; multiply by this for cm

# Deflection lever arms for thin-lens approximation: Δx_MM = δθ * lever
# Derivation: Δx = δθ * (L/2 + z_MM - z_mag_out)
LEVER_MM1 = Z_MAG_LEN / 2 + Z_MM1 - Z_MAG_OUT   # = 173.0 cm
LEVER_MM2 = Z_MAG_LEN / 2 + Z_MM2 - Z_MAG_OUT   # = 303.5 cm


# ════════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ════════════════════════════════════════════════════════════════════════════════

def open_trees(path):
    """
    Open ROOT file and return (RAWdata tree or None, RECOdata tree).
    Handles multi-cycle files (e.g. 1601730086.root) by falling back to
    the lowest readable cycle when the latest is incomplete.
    """
    f = uproot.open(path)
    all_keys = f.keys(cycle=True)

    # --- RECOdata: try cycles from highest to lowest ---
    reco = None
    for key in sorted([k for k in all_keys if k.startswith("RECOdata")], reverse=True):
        try:
            t = f[key]
            # quick read test on a lightweight branch
            t["NChamber1_position" if "NChamber1_position" in t.keys()
              else "NChamber2_position"].array(library="np", entry_stop=10)
            reco = t
            break
        except Exception:
            continue
    if reco is None:
        raise RuntimeError(f"No readable RECOdata cycle found in {path}")

    # --- RAWdata: try cycles from highest to lowest (may not exist) ---
    raw = None
    for key in sorted([k for k in all_keys if k.startswith("RAWdata")], reverse=True):
        try:
            t = f[key]
            t["QDC0_ch0"].array(library="np", entry_stop=10)
            raw = t
            break
        except Exception:
            continue

    return raw, reco


def first_hit(jagged_array):
    """
    Reduce a jagged per-event array to one float per event:
    take the first element, or NaN for empty events.
    """
    return ak.to_numpy(ak.fill_none(ak.firsts(jagged_array), np.nan))


# ════════════════════════════════════════════════════════════════════════════════
# Beam quality selection
# ════════════════════════════════════════════════════════════════════════════════

# DWC fiducial cut — beam spot should be inside ±4 mm on both axes
DWC_FIDUCIAL_MM = 4.0

# Trigger scintillator QDC channel (ch3 from Sep 29 logbook entry on
# "scintilator input channels to 3&5") and minimum signal above pedestal
TRIG_QDC_CH = 3
TRIG_QDC_MIN = 200    # counts above typical pedestal (~90-140)


def _has_dwc(reco):
    """Return True if the DWC branch is present and has actual data."""
    if "NDWC0_xPosition" not in reco.keys():
        return False
    return int(reco["NDWC0_xPosition"].array(library="np").sum()) > 0


def beam_quality_mask(raw, reco):
    """
    Select clean single-beam-particle events.

    When DWC is available (earlier runs, TDAQ < ~1601715484):
      - Exactly 1 DWC0 hit on each wire plane within ±4 mm fiducial
      - Trigger scintillator QDC above pedestal

    When DWC is absent (Oct 3+ runs, DWC physically removed):
      - All three upstream MIMOSA planes (0, 1, 2) have at least 1 cluster
        — this confirms the beam particle traversed the full upstream arm
      - Trigger QDC cut is applied only if RAWdata is readable

    Returns bool array [n_events].
    """
    n_events = reco.num_entries

    if _has_dwc(reco):
        # DWC-based beam quality (early runs)
        n_dwc_x = reco["NDWC0_xPosition"].array(library="np")
        n_dwc_y = reco["NDWC0_yPosition"].array(library="np")
        single_dwc = (n_dwc_x == 1) & (n_dwc_y == 1)

        dwc_x = first_hit(reco["DWC0_xPosition"].array())
        dwc_y = first_hit(reco["DWC0_yPosition"].array())
        in_fiducial = (np.abs(dwc_x) < DWC_FIDUCIAL_MM) & (np.abs(dwc_y) < DWC_FIDUCIAL_MM)

        mask = single_dwc & in_fiducial
    else:
        # MIMOSA-based beam quality: all three upstream planes must have a hit
        n0 = reco["NMimosa0_xPosition"].array(library="np")
        n1 = reco["NMimosa1_xPosition"].array(library="np")
        n2 = reco["NMimosa2_xPosition"].array(library="np")
        mask = (n0 >= 1) & (n1 >= 1) & (n2 >= 1)

    # Add QDC trigger cut only when RAWdata is readable
    if raw is not None:
        qdc_trig = raw[f"QDC0_ch{TRIG_QDC_CH}"].array(library="np") > TRIG_QDC_MIN
        mask = mask & qdc_trig

    return mask


# ════════════════════════════════════════════════════════════════════════════════
# Track multiplicity observables
# ════════════════════════════════════════════════════════════════════════════════

def _chamber_branches(reco):
    """
    Detect which Chamber naming convention the file uses:
      - Old calibration files: Chamber2, Chamber3
      - 2020 physics files:    Chamber1, Chamber2
    Returns list of 'N' branch names for the two Micromegas chambers.
    """
    keys = reco.keys()
    if "NChamber1_position" in keys:
        return ["NChamber1_position", "NChamber2_position"]
    return ["NChamber2_position", "NChamber3_position"]


def _has_downstream_mimosa(reco):
    """True when the full 6-plane MIMOSA telescope is present (Oct 2020 physics files)."""
    return "NMimosa3_xPosition" in reco.keys()


def upstream_nhits(reco):
    """
    Total reconstructed cluster positions across the three upstream MIMOSA
    planes (0, 1, 2). For a single through-going particle this should be ~3
    (one cluster per plane); more indicates beam-halo or secondaries.
    Returns int array [n_events].
    """
    n0 = reco["NMimosa0_xPosition"].array(library="np")
    n1 = reco["NMimosa1_xPosition"].array(library="np")
    n2 = reco["NMimosa2_xPosition"].array(library="np")
    return n0 + n1 + n2


def downstream_nhits(reco):
    """
    Total downstream charged-particle hits, combining:
      - Downstream MIMOSA planes (3, 4, 5) when present (2020 physics files)
      - The two Micromegas chambers (always present)
    A non-interacting beam contributes ~1 hit per detector; extra hits
    beyond that signal secondary particles from a target interaction.
    Returns int array [n_events].
    """
    mm_branches = _chamber_branches(reco)
    n_mm = sum(reco[b].array(library="np") for b in mm_branches)

    if _has_downstream_mimosa(reco):
        # Add downstream MIMOSA planes for better track counting
        n3 = reco["NMimosa3_xPosition"].array(library="np")
        n4 = reco["NMimosa4_xPosition"].array(library="np")
        n5 = reco["NMimosa5_xPosition"].array(library="np")
        return n_mm + n3 + n4 + n5

    return n_mm


def interaction_mask(reco):
    """
    Select candidate target-interaction events.

    Without downstream MIMOSA (calibration files):
      Require >= 2 Micromegas hits (secondary track).

    With downstream MIMOSA (2020 physics files, 6-plane telescope):
      Require the downstream MIMOSA sum > upstream sum + 2, i.e. noticeably
      more hits after the target than a single pass-through would give.
      This flags events where extra secondary particles were produced.

    Returns bool array [n_events].
    """
    n_up = upstream_nhits(reco)

    if _has_downstream_mimosa(reco):
        n3 = reco["NMimosa3_xPosition"].array(library="np")
        n4 = reco["NMimosa4_xPosition"].array(library="np")
        n5 = reco["NMimosa5_xPosition"].array(library="np")
        n_dn_mimosa = n3 + n4 + n5
        mm_branches = _chamber_branches(reco)
        n_mm = sum(reco[b].array(library="np") for b in mm_branches)
        # More downstream clusters than upstream (extra secondaries) OR 2+ MM hits
        return (n_up >= 3) & ((n_dn_mimosa > n_up + 2) | (n_mm >= 2))
    else:
        mm_branches = _chamber_branches(reco)
        n_mm = sum(reco[b].array(library="np") for b in mm_branches)
        return (n_up >= 3) & (n_mm >= 2)


# ════════════════════════════════════════════════════════════════════════════════
# Track reconstruction from MIMOSA telescope
# ════════════════════════════════════════════════════════════════════════════════

def _single_hit_position(reco, plane, axis, x_align=None, y_align=None):
    """
    Return per-event first cluster position (cm) for a given MIMOSA plane and
    axis ('x' or 'y').  Events with no hit give NaN; events with >1 cluster
    also give NaN so only unambiguous single-track events are used downstream.
    """
    n   = reco[f"NMimosa{plane}_{axis}Position"].array(library="np")
    pos = reco[f"Mimosa{plane}_{axis}Position"].array()  # jagged, cm
    result = np.full(len(n), np.nan)
    single = n == 1
    if single.any():
        result[single] = ak.to_numpy(ak.flatten(pos[single]))
    if axis == "x":
        result += MIMOSA_X_OFFSET[plane]
        if x_align is not None:
            result -= x_align[plane]
    if axis == "y" and y_align is not None:
        result -= y_align[plane]
    return result


def fit_track_segment(reco, planes, axis, x_align=None, y_align=None):
    """
    Least-squares straight-line fit through the given MIMOSA planes.

    Selects events where every requested plane has exactly 1 cluster (clean
    single track), then fits  x = x0 + slope * z  using the known z positions.

    Returns:
        intercept : float array [n_events], x at z=0 (cm); NaN if fit failed
        slope     : float array [n_events], dx/dz (radians ≈ tan θ); NaN if failed
        clean_mask: bool array [n_events], True where the fit succeeded
    """
    z_vals = Z_MIMOSA[planes]                          # z positions for these planes [cm]
    x_per_plane = np.column_stack(                     # shape (n_events, n_planes)
        [_single_hit_position(reco, p, axis, x_align, y_align) for p in planes]
    )
    # Require all planes to have exactly one hit
    clean = np.all(np.isfinite(x_per_plane), axis=1)

    n_ev = len(clean)
    intercept = np.full(n_ev, np.nan)
    slope     = np.full(n_ev, np.nan)

    if clean.sum() >= 2:
        # Vectorised least-squares for a line:  x ≈ intercept + slope * z
        z   = z_vals                                   # shape (n_planes,)
        x   = x_per_plane[clean]                       # shape (n_clean, n_planes)
        z_m = z.mean()
        x_m = x.mean(axis=1, keepdims=True)
        slope_vals     = ((x - x_m) * (z - z_m)).sum(axis=1) / ((z - z_m)**2).sum()
        intercept_vals = x_m[:, 0] - slope_vals * z_m
        slope[clean]     = slope_vals
        intercept[clean] = intercept_vals

    return intercept, slope, clean


def upstream_track(reco, x_align=None, y_align=None):
    """
    Fit straight-line tracks through MIMOSA planes 0–2 (upstream of target).
    Returns (x0, sx, y0, sy, mask) — intercepts and slopes in x and y, and
    the boolean mask of events where the fit succeeded on both axes.
    """
    x0, sx, mx = fit_track_segment(reco, [0, 1, 2], "x", x_align, y_align)
    y0, sy, my = fit_track_segment(reco, [0, 1, 2], "y", x_align, y_align)
    mask = mx & my
    return x0, sx, y0, sy, mask


def downstream_track(reco, x_align=None, y_align=None):
    """
    Fit straight-line tracks through MIMOSA planes 3–5 (downstream of target).
    Only available in 2020 physics files with the full 6-plane telescope.
    """
    if not _has_downstream_mimosa(reco):
        n = reco.num_entries
        nan = np.full(n, np.nan)
        return nan, nan, nan, nan, np.zeros(n, dtype=bool)
    x0, sx, mx = fit_track_segment(reco, [3, 4, 5], "x", x_align, y_align)
    y0, sy, my = fit_track_segment(reco, [3, 4, 5], "y", x_align, y_align)
    mask = mx & my
    return x0, sx, y0, sy, mask


def extrapolate(x0, slope, z_target):
    """Straight-line extrapolation: x = x0 + slope * z_target  [cm]."""
    return x0 + slope * z_target


def scattering_angles(reco, x_align=None, y_align=None):
    """
    Compute the x- and y-scattering angles at the target for each event:
        θ_scatter = arctan(slope_downstream) - arctan(slope_upstream)
    Both in radians.  Returns (theta_x, theta_y, valid_mask).
    """
    _, sx_up, _, sy_up, m_up = upstream_track(reco, x_align, y_align)
    _, sx_dn, _, sy_dn, m_dn = downstream_track(reco, x_align, y_align)
    valid = m_up & m_dn
    th_x = np.where(valid, np.arctan(sx_dn) - np.arctan(sx_up), np.nan)
    th_y = np.where(valid, np.arctan(sy_dn) - np.arctan(sy_up), np.nan)
    return th_x, th_y, valid


# ════════════════════════════════════════════════════════════════════════════════
# Momentum reconstruction using the bending magnet
# ════════════════════════════════════════════════════════════════════════════════

def magnetic_deflection(reco, x0_up, sx_up, mm_branch):
    """
    Compare extrapolated upstream track with actual MM position.

    x_extrap = x0_up + sx_up * z_MM   [cm]
    x_actual = Chamber_position * MM_TO_CM  [mm → cm]
    deflection = x_actual - x_extrap    [cm]

    Returns per-event deflection (cm); NaN where no MM hit or no upstream track.
    """
    z_MM = Z_MM1 if "1" in mm_branch else Z_MM2
    lever = LEVER_MM1 if "1" in mm_branch else LEVER_MM2

    n_mm  = reco[f"N{mm_branch}"].array(library="np")
    pos_mm = reco[mm_branch].array()

    # Only use events with exactly 1 MM hit and a valid upstream track
    single_mm = n_mm == 1
    has_up    = np.isfinite(x0_up)
    good = single_mm & has_up

    deflection = np.full(len(n_mm), np.nan)
    if good.any():
        x_act  = ak.to_numpy(ak.flatten(pos_mm[good])) * MM_TO_CM   # mm → cm
        x_ext  = extrapolate(x0_up[good], sx_up[good], z_MM)
        deflection[good] = x_act - x_ext

    return deflection


def calibrate_BL(defl_mm1, defl_mm2, E_beam_GeV):
    """
    Estimate B*L [T·m] from the deflection of beam-momentum electrons.

    Beam electrons dominate the deflection distribution and produce a sharp
    peak. We take the median of the central ±5 cm deflection window as the
    beam peak (robust against tails from scattered secondaries).

    For a particle of momentum p:
        deflection = δθ * lever
        δθ = 0.3 * B*L / p   (p in GeV/c, B*L in T·m, δθ in rad)
    → B*L = p * deflection / (0.3 * lever)
    """
    # Use MM1 for calibration (longer lever = larger deflection = more precise)
    valid = np.isfinite(defl_mm1) & (np.abs(defl_mm1) < 5.0)
    if valid.sum() < 100:
        return np.nan
    defl_beam = np.median(defl_mm1[valid])   # cm — beam electron peak

    BL = (E_beam_GeV * defl_beam) / (0.3 * LEVER_MM1)   # T·m
    return BL


def track_momentum(deflection_cm, lever_cm, BL_Tm):
    """
    Recover momentum [GeV/c] from magnetic deflection using:
        p = 0.3 * B*L [T·m] / δθ [rad]
        δθ = deflection [cm] / lever [cm]
    Sign of deflection tells the charge sign.
    Returns NaN where deflection is zero or undefined.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_theta = deflection_cm / lever_cm   # rad
        p = 0.3 * BL_Tm / delta_theta            # GeV/c  (signed)
    return p


# ════════════════════════════════════════════════════════════════════════════════
# Micromegas hit-level energy deposit (dE/dx proxy for PID)
# ════════════════════════════════════════════════════════════════════════════════

def mm_pdo_spectrum(raw, mask, board="02"):
    """
    Return flattened PDO (Pulse-height Digital Output) values from the
    Micromegas board. PDO is proportional to dE/dx in the gas gap.
    Heavier / slower particles (protons vs pions) deposit more energy,
    so the PDO spectrum can separate particle species.
    """
    pdo_jagged = raw[f"MMFE8_{board}_hitPDO"].array()
    return ak.to_numpy(ak.flatten(pdo_jagged[mask]))


# ════════════════════════════════════════════════════════════════════════════════
# Kinematic reconstruction (requires bending magnet — runs 173+ with 700 A)
# ════════════════════════════════════════════════════════════════════════════════

def hadronic_inv_mass(E_beam_MeV, E_e_prime_MeV, theta_e_rad):
    """
    Invariant mass W of the hadronic recoil system, from scattered
    electron kinematics (quasi-real photoproduction, Q2 ≠ 0 general case):

       W^2 = M_p^2 + 2*M_p*nu - Q^2
       Q^2 = 2*E_beam*E_e'*(1 - cos(theta_e))
       nu  = E_beam - E_e'

    Delta+(1232) should show as a peak at W ~ 1232 MeV.
    Returns W in MeV/c^2 (NaN where W^2 < 0).
    """
    nu = E_beam_MeV - E_e_prime_MeV                               # virtual photon energy
    Q2 = 2 * E_beam_MeV * E_e_prime_MeV * (1 - np.cos(theta_e_rad))  # photon virtuality
    W2 = M_PROTON**2 + 2 * M_PROTON * nu - Q2
    return np.where(W2 >= 0, np.sqrt(W2), np.nan)


def track_momentum_from_sagitta(sagitta_mm, L_mm, B_T):
    """
    Approximate track momentum from sagitta (deflection midpoint of arc)
    in a uniform magnetic field, using:

       p [GeV/c] = 0.3 * B[T] * L^2[m] / (8 * sagitta[m])

    sagitta_mm : measured transverse deflection at mid-arc [mm]
    L_mm       : arc length (distance across the magnet region) [mm]
    B_T        : magnetic field strength [T] (700 A bending magnet)
    """
    p_GeV = 0.3 * B_T * (L_mm * 1e-3)**2 / (8 * sagitta_mm * 1e-3)
    return p_GeV * 1e3   # return in MeV/c


# ════════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════════

def plot_multiplicity(reco, mask_beam, mask_interaction, run_name, outdir):
    """Track hit multiplicity before and after the target selection cut."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Track multiplicity — {run_name}")

    n_up = upstream_nhits(reco)[mask_beam]
    n_dn = downstream_nhits(reco)[mask_beam]
    is_int = mask_interaction[mask_beam]

    # Upstream (MIMOSA 0-2): expect ~3 for clean single-particle events
    axes[0].hist(n_up, bins=np.arange(-0.5, 20.5), color="steelblue",
                 alpha=0.7, label=f"beam-quality (N={mask_beam.sum():,})")
    axes[0].set_xlabel("Upstream hit clusters (MIMOSA 0–2 sum)")
    axes[0].set_ylabel("Events / bin")
    axes[0].set_title("Upstream multiplicity (beam particle)")
    axes[0].legend()

    # Downstream (MM Chambers 2+3): cut at >=2 selects interaction candidates
    axes[1].hist(n_dn, bins=np.arange(-0.5, 8.5), color="steelblue",
                 alpha=0.5, label="beam-quality")
    axes[1].hist(n_dn[is_int], bins=np.arange(-0.5, 8.5), color="tomato",
                 alpha=0.8, label=f"interaction candidates (N={is_int.sum():,})")
    axes[1].axvline(1.5, color="black", ls="--", lw=1.5, label="cut: ≥ 2")
    axes[1].set_xlabel("Downstream hits (MM Chamber2 + Chamber3)")
    axes[1].set_ylabel("Events / bin")
    axes[1].set_title("Downstream multiplicity  (≥2 → Delta+ candidates)")
    axes[1].legend()

    plt.tight_layout()
    out = Path(outdir) / "multiplicity.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_chamber_positions(reco, mask_beam, mask_interaction, run_name, outdir):
    """
    MM chamber hit positions for beam-quality vs interaction-candidate events.
    A second peak or wider distribution in the interaction sample signals
    secondary tracks from target scattering.
    """
    ch_names = _chamber_branches(reco)
    # Strip the leading 'N' and trailing '_position' to get position branch names
    pos_branches = [b[1:] for b in ch_names]   # e.g. "Chamber1_position"
    labels = [f"MM1 ({b})" for b in pos_branches]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Micromegas hit positions — {run_name}")

    for ax, pos_br, label in zip(axes, pos_branches, labels):
        pos_beam = ak.to_numpy(ak.flatten(reco[pos_br].array()[mask_beam]))
        pos_sel  = ak.to_numpy(ak.flatten(reco[pos_br].array()[mask_interaction]))

        ax.hist(pos_beam, bins=60, color="steelblue", alpha=0.5, label="beam-quality")
        ax.hist(pos_sel,  bins=60, color="tomato",    alpha=0.8, label="interaction candidates")
        ax.set_xlabel("Position [mm]")
        ax.set_ylabel("Hits / bin")
        ax.set_title(label)
        ax.legend()

    plt.tight_layout()
    out = Path(outdir) / "chamber_positions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_mimosa_telescope(reco, mask_beam, mask_interaction, run_name, outdir):
    """
    Show the x-position distributions across all 6 MIMOSA planes for beam
    vs interaction-candidate events.  Planes 0-2 are upstream of the target,
    planes 3-5 are downstream.  A wider or double-peaked distribution in
    planes 3-5 for interaction candidates signals secondary tracks fanning
    out from the target vertex.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"MIMOSA telescope x-positions — {run_name}")

    for i, ax in enumerate(axes.flat):
        br = f"Mimosa{i}_xPosition"
        if br not in reco.keys():
            ax.set_visible(False)
            continue
        x_beam = ak.to_numpy(ak.flatten(reco[br].array()[mask_beam]))
        x_sel  = ak.to_numpy(ak.flatten(reco[br].array()[mask_interaction]))

        ax.hist(x_beam, bins=60, color="steelblue", alpha=0.5,
                density=True, label="beam-quality")
        ax.hist(x_sel,  bins=60, color="tomato",    alpha=0.7,
                density=True, label="interaction candidates")
        side = "upstream" if i < 3 else "downstream"
        ax.set_title(f"MIMOSA {i}  (z={Z_MIMOSA[i]:.2f} cm, {side})")
        ax.set_xlabel("x position [cm]")
        ax.set_ylabel("Norm. counts")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = Path(outdir) / "mimosa_telescope.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_scattering_angle(reco, mask_beam, run_name, outdir, x_align=None, y_align=None):
    """
    Scatter-angle distribution (x and y) from the 6-plane MIMOSA telescope.
    The angular trigger pre-selects events with θ > some minimum, so we
    expect a hole near zero.  The tails are inelastically scattered tracks.
    """
    th_x, th_y, valid = scattering_angles(reco, x_align, y_align)
    mask = mask_beam & valid
    if mask.sum() < 10:
        print("  Skipping scattering-angle plot (too few events with both track segments)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Scattering angles (MIMOSA 0–2 vs 3–5) — {run_name}")

    for ax, th, label in zip(axes, [th_x[mask]*1e3, th_y[mask]*1e3], ["θ_x", "θ_y"]):
        ax.hist(th, bins=80, color="darkorange", alpha=0.8)
        ax.axvline(0, color="grey", ls=":", lw=1)
        ax.set_xlabel(f"{label}  [mrad]")
        ax.set_ylabel("Events / bin")
        ax.set_title(f"{label} scattering  (N={mask.sum():,})")

    plt.tight_layout()
    out = Path(outdir) / "scattering_angles.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_momentum_spectrum(defl1, defl2, mask_beam, mask_sel,
                           BL_Tm, E_beam_GeV, run_name, outdir):
    """
    Momentum spectrum from magnetic deflection at MM1 and MM2.
    The beam electron peak (p = E_beam, negative deflection for e-)
    is used for calibration; the positive-deflection (or low-p) tail
    contains secondary hadrons from the target interaction.
    """
    p1 = track_momentum(defl1, LEVER_MM1, BL_Tm)  # GeV/c, signed
    p2 = track_momentum(defl2, LEVER_MM2, BL_Tm)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Track momentum spectrum (B·L = {BL_Tm:.3f} T·m) — {run_name}")

    for ax, p, label in zip(axes, [p1, p2], ["MM1 (Chamber1)", "MM2 (Chamber2)"]):
        p_beam = p[mask_beam]
        p_sel  = p[mask_sel]
        valid_b = np.isfinite(p_beam)
        valid_s = np.isfinite(p_sel)

        prange = (-8, 8)
        ax.hist(p_beam[valid_b], bins=80, range=prange, color="steelblue",
                alpha=0.5, density=True, label="beam-quality")
        ax.hist(p_sel[valid_s],  bins=80, range=prange, color="tomato",
                alpha=0.7, density=True, label="interaction candidates")
        ax.axvline(-E_beam_GeV, color="navy",  ls="--", lw=1.5,
                   label=f"−{E_beam_GeV} GeV beam e⁻")
        ax.axvline( E_beam_GeV, color="grey",  ls=":",  lw=1.0,
                   label=f"+{E_beam_GeV} GeV (if +q)")
        ax.set_xlabel("Momentum  p  [GeV/c]  (sign = charge × deflection direction)")
        ax.set_ylabel("Norm. counts / bin")
        ax.set_title(label)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = Path(outdir) / "momentum_spectrum.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_beam_profile(reco, mask_beam, mask_interaction, run_name, outdir):
    """
    DWC0 beam position profile at target: compare beam-quality vs interaction
    candidates to check whether interactions cluster at the beam centre.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Beam profile at DWC0 — {run_name}")

    dwc_x_b = first_hit(reco["DWC0_xPosition"].array())[mask_beam]
    dwc_y_b = first_hit(reco["DWC0_yPosition"].array())[mask_beam]
    dwc_x_s = first_hit(reco["DWC0_xPosition"].array())[mask_interaction]
    dwc_y_s = first_hit(reco["DWC0_yPosition"].array())[mask_interaction]

    def valid(x, y):
        ok = ~np.isnan(x) & ~np.isnan(y)
        return x[ok], y[ok]

    xb, yb = valid(dwc_x_b, dwc_y_b)
    xs, ys = valid(dwc_x_s, dwc_y_s)

    h = axes[0].hist2d(xb, yb, bins=40, range=[[-5, 5], [-5, 5]], cmap="Blues")
    plt.colorbar(h[3], ax=axes[0])
    axes[0].set_xlabel("DWC0 x [mm]")
    axes[0].set_ylabel("DWC0 y [mm]")
    axes[0].set_title("Beam profile (beam-quality events)")

    h2 = axes[1].hist2d(xs, ys, bins=40, range=[[-5, 5], [-5, 5]], cmap="Reds")
    plt.colorbar(h2[3], ax=axes[1])
    axes[1].set_xlabel("DWC0 x [mm]")
    axes[1].set_ylabel("DWC0 y [mm]")
    axes[1].set_title("Beam profile (interaction candidates)")

    plt.tight_layout()
    out = Path(outdir) / "beam_profile.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_qdc_spectra(raw, mask_beam, mask_interaction, run_name, outdir):
    """
    QDC spectra for the first 6 channels.
    Compares beam-quality events vs interaction candidates.
    The trigger/scintillator channels (ch3, ch5) should differ between
    samples if secondary particles fire additional detectors.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"QDC0 spectra — {run_name}")

    for ch, ax in enumerate(axes.flat):
        vals_b = raw[f"QDC0_ch{ch}"].array(library="np")[mask_beam]
        vals_s = raw[f"QDC0_ch{ch}"].array(library="np")[mask_interaction]
        ax.hist(vals_b, bins=60, range=(0, 2000), color="steelblue",
                alpha=0.5, label="beam-quality", density=True)
        ax.hist(vals_s, bins=60, range=(0, 2000), color="tomato",
                alpha=0.7, label="interaction", density=True)
        ax.set_xlabel("QDC counts")
        ax.set_title(f"QDC0 ch{ch}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = Path(outdir) / "qdc_spectra.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_mm_pdo(raw, mask_beam, mask_interaction, run_name, outdir):
    """
    Micromegas PDO (Pulse-height Digital Output) for interaction candidates.
    PDO is proportional to dE/dx in the gas — heavier particles (protons)
    deposit more energy than pions, giving a higher PDO.
    Looking for a bimodal distribution would hint at p+pi from Delta+ decay.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Micromegas dE/dx (PDO) — {run_name}")

    for ax, board, label in zip(axes, ["02", "03"], ["MM1 (MMFE8_02)", "MM2 (MMFE8_03)"]):
        pdo_b = mm_pdo_spectrum(raw, mask_beam,        board=board)
        pdo_s = mm_pdo_spectrum(raw, mask_interaction, board=board)

        bins = np.linspace(0, 1200, 80)
        ax.hist(pdo_b, bins=bins, color="steelblue", alpha=0.5,
                density=True, label="beam-quality")
        ax.hist(pdo_s, bins=bins, color="tomato",    alpha=0.7,
                density=True, label="interaction candidates")
        ax.set_xlabel("PDO (pulse height, proportional to dE/dx)")
        ax.set_ylabel("Normalised counts / bin")
        ax.set_title(label)
        ax.legend()

    plt.tight_layout()
    out = Path(outdir) / "mm_pdo.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_w_proper(reco, defl1, BL_Tm, E_beam_MeV, mask_beam, run_name, outdir, x_align=None, y_align=None):
    """
    Proper hadronic invariant mass W using both magnet and MIMOSA telescope:

      E_e'  from MM1 magnetic deflection (calibrated B·L)  →  momentum magnitude
      θ_e   from MIMOSA 3–5 vs 0–2 straight-line angle difference

      W² = M_p² + 2·M_p·(E_beam − E_e') − 2·E_beam·E_e'·(1 − cos θ_e)

    A Delta+(1232) signal appears as a peak at W = 1232 MeV above a smooth
    background.  We assume the downstream MIMOSA track and the MM hit both
    belong to the scattered electron (valid for Δ⁺→p+π⁰ where π⁰ is invisible
    and the proton stops before MM1).
    """
    if np.isnan(BL_Tm):
        print("  Skipping proper W plot (no B·L calibration)")
        return None
    if not _has_downstream_mimosa(reco):
        print("  Skipping proper W plot (no downstream MIMOSA in this file)")
        return None

    # Momentum from MM1 deflection — take the absolute value (we want magnitude E_e')
    p_GeV    = track_momentum(defl1, LEVER_MM1, BL_Tm)
    E_e_MeV  = np.abs(p_GeV) * 1e3      # massless approximation fine at GeV scale

    # Scattering angle from MIMOSA telescope (upstream vs downstream segment)
    th_x, th_y, valid_angle = scattering_angles(reco, x_align, y_align)
    theta_e  = np.sqrt(th_x**2 + th_y**2)   # total angle [rad]

    # Event selection: beam-quality, valid angle, valid momentum, E_e' < E_beam
    valid_p = np.isfinite(E_e_MeV) & (E_e_MeV < E_beam_MeV * 0.999) & (E_e_MeV > 100)
    good    = mask_beam & valid_angle & valid_p

    if good.sum() < 10:
        print(f"  Skipping proper W plot (only {good.sum()} events pass all cuts)")
        return None

    W = hadronic_inv_mass(E_beam_MeV, E_e_MeV[good], theta_e[good])
    W = W[np.isfinite(W) & (W > M_PROTON)]
    print(f"  Proper W reconstruction: {good.sum():,} events -> {len(W):,} with valid W")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Hadronic invariant mass W — {run_name}\n"
        f"E_e' from MM1 deflection · θ_e from MIMOSA 0–2 vs 3–5  (N = {len(W):,})"
    )

    # Full range
    axes[0].hist(W, bins=60, range=(M_PROTON, 3000), color="tomato", alpha=0.8)
    axes[0].axvline(M_DELTA,  color="black", ls="--", lw=2,
                    label=f"Δ⁺(1232) = {M_DELTA:.0f} MeV")
    axes[0].axvline(M_PROTON, color="grey",  ls=":",  lw=1.5,
                    label=f"p = {M_PROTON:.0f} MeV")
    axes[0].set_xlabel("W  [MeV/c²]")
    axes[0].set_ylabel("Events / bin")
    axes[0].set_title("Full range")
    axes[0].legend()

    # Zoom on Delta+ region
    axes[1].hist(W, bins=40, range=(900, 1800), color="tomato", alpha=0.8)
    axes[1].axvline(M_DELTA,  color="black", ls="--", lw=2,
                    label=f"Δ⁺(1232) = {M_DELTA:.0f} MeV")
    axes[1].axvline(M_PROTON, color="grey",  ls=":",  lw=1.5,
                    label=f"p threshold")
    axes[1].set_xlabel("W  [MeV/c²]")
    axes[1].set_ylabel("Events / bin")
    axes[1].set_title("Zoom: Δ⁺ region (900–1800 MeV)")
    axes[1].legend()

    plt.tight_layout()
    out = Path(outdir) / "w_distribution_proper.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")
    return W      # hand back to caller for fitting


def _bw_plus_poly(W, M, Gamma, A_sig, a0, a1, a2):
    """Relativistic Breit-Wigner + quadratic polynomial background."""
    bw = A_sig * (Gamma / 2)**2 / ((W - M)**2 + (Gamma / 2)**2)
    bg = a0 + a1 * (W - 1232) + a2 * (W - 1232)**2
    return bw + bg


def plot_w_fit(W_vals, run_name, outdir):
    """
    Fit the W distribution with a Breit-Wigner for the Delta+(1232) signal
    on top of a smooth polynomial background, then show signal significance.

    Bins: ~24 MeV/bin across 938–1800 MeV (comparable to MM momentum resolution).
    Gamma is fixed to the PDG width of 117 MeV; M and amplitude float.
    """
    if not _SCIPY:
        print("  scipy not found — skipping W fit  (pip install scipy)")
        return

    W_fit = W_vals[(W_vals > 938) & (W_vals < 1800)]
    if len(W_fit) < 30:
        print(f"  Too few events for W fit ({len(W_fit)}), skipping")
        return

    bins = np.linspace(938, 1800, 37)           # 36 bins, ~24 MeV each
    counts, edges = np.histogram(W_fit, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    errors  = np.sqrt(np.maximum(counts, 1))    # Poisson; floor at 1 to avoid /0

    use = counts > 0

    # Initial guesses: M near PDG, Gamma fixed at 117, amplitude from counts
    p0     = [1232, 117, float(counts.max()) * 20, float(counts.mean()), -0.05, 0.0]
    bounds = ([1100, 116.9,    0, -1e4, -5, -0.05],
              [1400, 117.1, 1e6,  1e4,  5,  0.05])

    try:
        popt, pcov = curve_fit(
            _bw_plus_poly, centers[use], counts[use],
            p0=p0, sigma=errors[use], absolute_sigma=True,
            bounds=bounds, maxfev=20000
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as exc:
        print(f"  W fit did not converge: {exc}")
        return

    M_fit, G_fit, A_fit = popt[0], popt[1], popt[2]
    M_err               = perr[0]

    # Background alone in the signal window (M ± 2Γ)
    win_lo, win_hi = M_fit - 2 * G_fit, M_fit + 2 * G_fit
    in_win   = (centers >= win_lo) & (centers <= win_hi)
    bg_counts = popt[3] + popt[4]*(centers[in_win]-1232) + popt[5]*(centers[in_win]-1232)**2
    B_win    = float(bg_counts.sum())
    S_win    = float(counts[in_win].sum()) - B_win
    sig      = S_win / np.sqrt(B_win) if B_win > 0 else 0.0

    # Chi^2
    resid    = counts[use] - _bw_plus_poly(centers[use], *popt)
    chi2_val = float(np.sum((resid / errors[use])**2))
    ndf      = int(use.sum()) - len(popt)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"W distribution fit — {run_name}  "
                 f"(Breit-Wigner + poly background,  chi2/ndf = {chi2_val:.1f}/{ndf})")

    W_dense = np.linspace(938, 1800, 600)
    y_total  = _bw_plus_poly(W_dense, *popt)
    y_bg     = popt[3] + popt[4]*(W_dense-1232) + popt[5]*(W_dense-1232)**2
    y_signal = y_total - y_bg

    for ax, xlim, title in zip(
        axes,
        [(938, 1800), (1000, 1600)],
        ["Full fit range (938–1800 MeV)", f"Zoom on Delta+(1232) region"]
    ):
        ax.errorbar(centers, counts, yerr=errors,
                    fmt='o', color='tomato', ms=4, label='Data')
        ax.plot(W_dense, y_total,  color='black', lw=2,
                label=f'Fit: M={M_fit:.0f}+/-{M_err:.0f} MeV')
        ax.plot(W_dense, y_bg,    color='steelblue', lw=1.5, ls='--',
                label='Background (poly)')
        ax.fill_between(W_dense, y_bg, y_total,
                        where=y_signal > 0, alpha=0.35, color='gold',
                        label=f'BW signal (S={S_win:.0f}, S/sqrt(B)={sig:.1f})')
        ax.axvline(M_DELTA,  color='black', ls=':', lw=1.2,
                   label=f'PDG: {M_DELTA:.0f} MeV')
        ax.axvline(M_PROTON, color='grey',  ls=':', lw=1.0)
        ax.set_xlim(xlim)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('W  [MeV/c^2]')
        ax.set_ylabel('Events / bin  (~24 MeV)')
        ax.set_title(title)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = Path(outdir) / "w_fit.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")
    print(f"  Delta+ mass  : {M_fit:.1f} +/- {M_err:.1f} MeV  (PDG: 1232 MeV)")
    print(f"  Signal events: {S_win:.0f} in M +/- 2*Gamma window")
    print(f"  Significance : S/sqrt(B) = {sig:.1f} sigma")
    print(f"  chi^2/ndf    : {chi2_val:.1f} / {ndf}")


# ════════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python nff_delta_analysis.py <run.root> [beam_energy_GeV]")
        sys.exit(1)

    path = sys.argv[1]
    E_beam_GeV = float(sys.argv[2]) if len(sys.argv) > 2 else 4.6
    E_beam_MeV = E_beam_GeV * 1e3

    run_name = Path(path).stem
    outdir   = Path(path).parent

    print(f"\n{'='*60}")
    print(f"  NFF Delta+ analysis")
    print(f"  File       : {path}")
    print(f"  Beam energy: {E_beam_GeV:.1f} GeV")
    print(f"{'='*60}\n")

    # Load the two TTrees from the converted ROOT file
    raw, reco = open_trees(path)
    n_total = reco.num_entries   # use RECOdata as the authoritative event count
    if raw is None:
        print(f"Events in file: {n_total:,}  (RAWdata unreadable — reco-only mode)")
    else:
        print(f"Events in file: {n_total:,}")

    # Apply beam-quality selection (DWC fiducial + trigger scintillator)
    mask_beam = beam_quality_mask(raw, reco)
    n_beam = mask_beam.sum()
    print(f"Beam-quality events: {n_beam:,}  ({100*n_beam/n_total:.1f}%)")

    # Select inelastic target-interaction candidates (>=2 downstream MM tracks)
    mask_int = interaction_mask(reco)
    mask_sel = mask_beam & mask_int
    n_sel = mask_sel.sum()
    print(f"Interaction candidates: {n_sel:,}  ({100*n_sel/n_beam:.2f}% of beam events)\n")

    # ── Alignment correction from no-target run ───────────────────────────────
    from mimosa_residuals import compute_alignment
    align_x, align_y = compute_alignment("Runs/1601727557.root")

    # ── Upstream track fitting (MIMOSA 0–2) ──────────────────────────────────
    x0_up, sx_up, y0_up, sy_up, m_up = upstream_track(reco, align_x, align_y)
    n_up_track = m_up.sum()
    print(f"Upstream tracks fitted  : {n_up_track:,}  ({100*n_up_track/n_total:.1f}%)")

    # ── Downstream track fitting (MIMOSA 3–5, if present) ────────────────────
    _, sx_dn, _, sy_dn, m_dn = downstream_track(reco, align_x, align_y)
    n_dn_track = m_dn.sum()
    if _has_downstream_mimosa(reco):
        print(f"Downstream tracks fitted: {n_dn_track:,}  ({100*n_dn_track/n_total:.1f}%)")

    # ── Magnetic deflection and momentum ─────────────────────────────────────
    ch_pos_branches = [b[1:] for b in _chamber_branches(reco)]  # strip leading 'N'
    defl1 = magnetic_deflection(reco, x0_up, sx_up, ch_pos_branches[0])
    defl2 = magnetic_deflection(reco, x0_up, sx_up, ch_pos_branches[1])

    BL_Tm = calibrate_BL(defl1[mask_beam], defl2[mask_beam], E_beam_GeV)
    if not np.isnan(BL_Tm):
        print(f"Calibrated B·L         : {BL_Tm:.4f} T·m")
    else:
        print("  B·L calibration failed (too few hits at MM); momentum plots skipped")

    # Generate output plots
    print("\nGenerating plots...")
    plot_multiplicity(reco, mask_beam, mask_sel, run_name, outdir)
    plot_chamber_positions(reco, mask_beam, mask_sel, run_name, outdir)
    plot_mimosa_telescope(reco, mask_beam, mask_sel, run_name, outdir)

    if _has_dwc(reco):
        plot_beam_profile(reco, mask_beam, mask_sel, run_name, outdir)
    else:
        print("  Skipping beam profile (no DWC in this run)")

    if _has_downstream_mimosa(reco):
        plot_scattering_angle(reco, mask_beam, run_name, outdir, align_x, align_y)

    if not np.isnan(BL_Tm):
        plot_momentum_spectrum(defl1, defl2, mask_beam, mask_sel,
                               BL_Tm, E_beam_GeV, run_name, outdir)

    if raw is not None:
        plot_qdc_spectra(raw, mask_beam, mask_sel, run_name, outdir)
        plot_mm_pdo(raw, mask_beam, mask_sel, run_name, outdir)
    else:
        print("  Skipping QDC/PDO plots (RAWdata not readable in this file)")

    W_vals = plot_w_proper(reco, defl1, BL_Tm, E_beam_MeV, mask_beam, run_name, outdir, align_x, align_y)
    if W_vals is not None and len(W_vals) > 30:
        plot_w_fit(W_vals, run_name, outdir)

    # Print summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  Total events          : {n_total:>10,}")
    print(f"  Beam-quality events   : {n_beam:>10,}  ({100*n_beam/n_total:.1f}%)")
    print(f"  Interaction candidates: {n_sel:>10,}  ({100*n_sel/n_beam:.2f}% of beam)")
    print(f"  Upstream tracks       : {n_up_track:>10,}  ({100*n_up_track/n_total:.1f}%)")
    if _has_downstream_mimosa(reco):
        print(f"  Downstream tracks     : {n_dn_track:>10,}  ({100*n_dn_track/n_total:.1f}%)")
    if not np.isnan(BL_Tm):
        print(f"  Calibrated B·L        : {BL_Tm:>10.4f} T·m")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()