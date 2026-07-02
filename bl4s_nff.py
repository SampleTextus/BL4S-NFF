#!/usr/bin/env python3
import uproot
import numpy as np
import awkward as ak
from pathlib import Path
from collections import namedtuple
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import chi2 as chi2_dist


# Z positions in cm
zmimosa   = np.array([35.0, 38.6, 41.4, 48.5, 51.5, 53.65])
zcmspixel = 36.9    # between Mimosa0 (35.0) and Mimosa1 (38.6)
ztarget   = 44.95
zbrmc     = 211.5
zbrmin    = zbrmc - 75
zbrmout   = zbrmc + 75
zdwc      = 339.75
zmm       = np.array([384.5, 515.0])
zbrmlen   = 150.0

# Masses in GeV
mproton   = 938.272e-3
melectron = 0.511e-3
mpion     = 139.570e-3
mdelta    = 1232.0e-3

# Beam / analysis constants
nrg_beam       = 4.0
pelectron      = np.sqrt(nrg_beam**2 - melectron**2)
GeV_factor     = 0.299792
h_bar          = 6.582e-25   # GeV·s
delta_lifetime = 5.6e-24     # s
Gamma          = h_bar / delta_lifetime

MAX_RESIDUAL_CM = 0.1        # 1 mm max absolute track residual per plane
PHI_NSIGMA      = 3.0        # phi cut: ±3 robust-sigma of the 600A reference run

MIMOSA_PITCH_CM       = 0.00184   # 18.4 µm Mimosa26/CMSPixel sensor pitch (physical spec)
# Target-intersection pairing tolerance is NOT set to MIMOSA_PITCH_CM directly: measured on
# the 600A no-target reference run, the achieved upstream/downstream position-matching
# resolution at the target is ~140-400 µm (extrapolation over a short lever arm amplifies the
# pixel-level hit resolution well past the raw pitch), ~20x looser than one pixel pitch. The
# tolerance is instead calibrated per-dataset via compute_target_pair_reference, mirroring the
# existing IQR-based sigma methodology already used for PHI_NSIGMA.
TARGET_PAIR_NSIGMA    = 3.0       # pairing cut: median + N*robust-sigma of the 600A reference run
# Seeding-search caps match upstream_valid/downstream_valid's 1-2 hit window: relaxing this to
# admit busier events was tried and reverted -- verified empirically to flood the sample with
# combinatorial "ghost" tracks (busy events find a spuriously-consistent line by chance at the
# 1mm residual tolerance) that dilute the reconstructed W signal rather than recovering it.
MAX_HITS_PER_PLANE_UP = 2         # seeding-search cap for M0/CMSPixel/M1/M2
MAX_HITS_PER_PLANE_DN = 2         # seeding-search cap for M3/M4/M5

W_LO, W_HI, W_MID = 1.0, 1.7, 1.35
_C_BG = W_HI - W_LO         # normalisation of linear BG over [W_LO, W_HI]

def _bw_unnorm(W):
    return W * mdelta * Gamma / ((W**2 - mdelta**2)**2 + (mdelta * Gamma)**2)

_C_BW, _ = quad(_bw_unnorm, W_LO, W_HI)


def open_tree(path):
    f    = uproot.open(path)
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


def upstream_valid(reco):
    """Events with 1-2 hits in Mimosa 0, 1, 2 AND CMSPixel.
    This is the input mask for upstream_track's seeding algorithm as well as alignment.py's
    and plots.py's raw per-plane ak.mean diagnostics. The 1-2 cap is deliberate: at higher
    multiplicity, a fixed 1mm residual tolerance over these short lever arms lets busy/pile-up
    events find a spuriously-consistent straight line by combinatorial luck ("ghost" tracks)
    -- verified empirically to dilute the reconstructed W signal even though it increases raw
    statistics, so the cap stays tight rather than admitting those events."""
    nx0 = reco["NMimosa0_xPosition"].array(library="np")
    ny0 = reco["NMimosa0_yPosition"].array(library="np")
    nx1 = reco["NMimosa1_xPosition"].array(library="np")
    ny1 = reco["NMimosa1_yPosition"].array(library="np")
    nx2 = reco["NMimosa2_xPosition"].array(library="np")
    ny2 = reco["NMimosa2_yPosition"].array(library="np")
    ncx = reco["NCMSPixel0_xPosition"].array(library="np")
    ncy = reco["NCMSPixel0_yPosition"].array(library="np")
    return ((nx0 >= 1) & (nx0 <= 2) & (ny0 >= 1) & (ny0 <= 2) &
            (nx1 >= 1) & (nx1 <= 2) & (ny1 >= 1) & (ny1 <= 2) &
            (nx2 >= 1) & (nx2 <= 2) & (ny2 >= 1) & (ny2 <= 2) &
            (ncx >= 1) & (ncx <= 2) & (ncy >= 1) & (ncy <= 2))


def downstream_valid(reco):
    """Events with 1-2 hits in Mimosa 3, 4, 5.
    This is the input mask for downstream_track's seeding algorithm as well as alignment.py's
    and plots.py's raw per-plane ak.mean diagnostics -- see upstream_valid docstring for why
    the cap stays tight."""
    nx3 = reco["NMimosa3_xPosition"].array(library="np")
    ny3 = reco["NMimosa3_yPosition"].array(library="np")
    nx4 = reco["NMimosa4_xPosition"].array(library="np")
    ny4 = reco["NMimosa4_yPosition"].array(library="np")
    nx5 = reco["NMimosa5_xPosition"].array(library="np")
    ny5 = reco["NMimosa5_yPosition"].array(library="np")
    return ((nx3 >= 1) & (nx3 <= 2) & (ny3 >= 1) & (ny3 <= 2) &
            (nx4 >= 1) & (nx4 <= 2) & (ny4 >= 1) & (ny4 <= 2) &
            (nx5 >= 1) & (nx5 <= 2) & (ny5 >= 1) & (ny5 <= 2))


def _pad_hits(arr_ak, align_offset, max_hits):
    """
    Pad a ragged per-event hit array to a dense (n_events, K) NumPy array, alignment-corrected.
    Missing hits (beyond that event's actual hit count) become NaN.
    K = min(actual max multiplicity in this array, max_hits) -- the cap only exists to bound
    the seeding search below; RECOdata hits are already distinct-particle centroids (verified:
    no charge/PDO branch exists for Mimosa/CMSPixel, and RAWdata shows clustering already
    happened upstream), so there is never a physical reason to discard a hit outright.
    """
    n_max = int(ak.max(ak.num(arr_ak))) if len(arr_ak) > 0 else 1
    K = max(1, min(n_max, max_hits))
    padded = ak.pad_none(arr_ak, K, clip=True)
    dense = ak.to_numpy(ak.fill_none(padded, np.nan))
    return dense - align_offset


def _seed_fit_1d(z, hit_grids, anchor_idx, return_idx=False):
    """
    Vectorized seed-and-extend track finder for one coordinate (x or y) across a detector arm.

    z         : (P,) z-positions of the P planes, track order.
    hit_grids : list of P dense (n_events, K_p) arrays, already alignment-corrected
                (NaN = no hit at that padding slot).
    anchor_idx: (a1, a2) indices of the two mandatory seed-pair planes.

    For every event, enumerates every (hit in plane a1) x (hit in plane a2) seed line,
    extrapolates it to every other plane in the arm and attaches the nearest available hit
    there, scores each full P-plane candidate by the max absolute residual of a joint
    least-squares refit through all P chosen points, and keeps the lowest-scoring candidate.
    No averaging ever occurs -- every 2+-hit case is resolved by genuine track consistency.

    A candidate built from a padding slot (event has fewer hits than K_a1/K_a2 at the anchor
    planes) is NaN at that anchor by construction; np.polyfit propagates that NaN to every
    coefficient for that candidate's column (verified: a NaN entry in one column of a
    multi-column np.polyfit call only invalidates that column, leaving others unaffected), so
    a padding-only candidate can never win the argmin unless every candidate for that event is
    invalid, i.e. no consistent track exists across the arm.

    Returns (P, n_events) selected positions. A NaN column means no consistent candidate was
    found for that event -- this flows through track_max_residual as an automatic reject in
    run_analysis/compute_phi_reference, with no extra plumbing required.
    """
    a1, a2 = anchor_idx
    P = len(z)
    match_idx = [i for i in range(P) if i not in anchor_idx]
    n_events = hit_grids[0].shape[0]

    ha1, ha2 = hit_grids[a1], hit_grids[a2]                      # (N, K1), (N, K2)
    za1, za2 = z[a1], z[a2]

    with np.errstate(invalid="ignore"):
        slope = (ha2[:, None, :] - ha1[:, :, None]) / (za2 - za1)   # (N, K1, K2)
        icept = ha1[:, :, None] - slope * za1

        chosen = {a1: np.broadcast_to(ha1[:, :, None], slope.shape).copy(),
                  a2: np.broadcast_to(ha2[:, None, :], slope.shape).copy()}

        best_k_all = {}                                             # per match plane, for idx
        for m in match_idx:
            pred = slope * z[m] + icept                              # (N, K1, K2)
            hits_m = hit_grids[m]                                    # (N, Km)
            diff = np.abs(pred[..., None] - hits_m[:, None, None, :])  # (N, K1, K2, Km)
            diff = np.where(np.isnan(diff), np.inf, diff)
            best_k = np.argmin(diff, axis=-1)                        # (N, K1, K2)
            best_k_all[m] = best_k
            hits_m_b = np.broadcast_to(hits_m[:, None, None, :], diff.shape)
            chosen[m] = np.take_along_axis(hits_m_b, best_k[..., None], axis=-1)[..., 0]

        shape = slope.shape                                          # (N, K1, K2)
        stacked = np.stack([chosen[i].reshape(-1) for i in range(P)])  # (P, N*K1*K2)
        coeff = np.polyfit(z, stacked, 1)                             # (2, N*K1*K2)
        resid = stacked - (coeff[0][None, :] * z[:, None] + coeff[1][None, :])
        max_resid = np.max(np.abs(resid), axis=0).reshape(shape)
        max_resid = np.where(np.isnan(max_resid), np.inf, max_resid)

    flat = max_resid.reshape(n_events, -1)
    best_flat = np.argmin(flat, axis=1)
    i_best, j_best = np.unravel_index(best_flat, shape[1:])
    ev = np.arange(n_events)
    finite = np.isfinite(flat[ev, best_flat])

    sel = np.full((P, n_events), np.nan)
    for i in range(P):
        sel[i, finite] = chosen[i][ev, i_best, j_best][finite]

    if not return_idx:
        return sel

    # chosen hit index per plane for the winning candidate (NaN where no valid candidate)
    idx = np.full((P, n_events), np.nan)
    idx[a1, finite] = i_best[finite]
    idx[a2, finite] = j_best[finite]
    for m in match_idx:
        idx[m, finite] = best_k_all[m][ev, i_best, j_best][finite]
    return sel, idx


def _apply_exclude(grids, ex):
    """NaN out, per plane, the hit indices in `ex` (P, N) — used to hide candidate-1's chosen
    hits so a re-run of the same tracker finds the next-best (second-particle) track."""
    for p, g in enumerate(grids):
        m = np.isfinite(ex[p])
        rows = np.where(m)[0]
        cols = ex[p][m].astype(int)
        g[rows, cols] = np.nan


def upstream_track(reco, up_valid, align_x=None, align_y=None, rot_x=None, rot_y=None,
                   cms_align=None, cms_rot=None, exclude=None, return_idx=False, max_hits=None):
    """
    4-plane straight-line fit: M0(35.0) → CMS(36.9) → M1(38.6) → M2(41.4).
    Hit selection uses a seeding track finder (see _seed_fit_1d): the seed line is built from
    (CMSPixel, Mimosa2) -- CMSPixel is a mandatory anchor (physically pileup-resistant per its
    faster readout, even though no explicit timestamp exists in this data) paired with the
    plane giving the longest lever arm from it. Mimosa0 and Mimosa1 are each matched to the
    nearest hit consistent with the extrapolated seed line, and the best-scoring full-arm
    candidate is kept per event -- no averaging ever occurs. up_valid should be
    upstream_valid(reco): a 1-2 hit input population, not a higher cap (verified empirically
    that relaxing it lets busy events pass by combinatorial luck rather than genuine tracking).
    Returns namedtuple with slopes, intercepts, and per-plane residuals r_x/r_y (shape 4×N).
    """
    z_up = np.array([zmimosa[0], zcmspixel, zmimosa[1], zmimosa[2]])

    # Alignment offsets in track order: [M0, CMS, M1, M2]
    ax = np.array([
        align_x[0]   if align_x   is not None else 0.0,
        cms_align[0] if cms_align is not None else 0.0,
        align_x[1]   if align_x   is not None else 0.0,
        align_x[2]   if align_x   is not None else 0.0,
    ])
    ay = np.array([
        align_y[0]   if align_y   is not None else 0.0,
        cms_align[1] if cms_align is not None else 0.0,
        align_y[1]   if align_y   is not None else 0.0,
        align_y[2]   if align_y   is not None else 0.0,
    ])

    raw_x = [
        reco["Mimosa0_xPosition"].array()[up_valid],
        reco["CMSPixel0_xPosition"].array()[up_valid],
        reco["Mimosa1_xPosition"].array()[up_valid],
        reco["Mimosa2_xPosition"].array()[up_valid],
    ]
    raw_y = [
        reco["Mimosa0_yPosition"].array()[up_valid],
        reco["CMSPixel0_yPosition"].array()[up_valid],
        reco["Mimosa1_yPosition"].array()[up_valid],
        reco["Mimosa2_yPosition"].array()[up_valid],
    ]

    mh = MAX_HITS_PER_PLANE_UP if max_hits is None else max_hits
    grids_x = [_pad_hits(r, a, mh) for r, a in zip(raw_x, ax)]
    grids_y = [_pad_hits(r, a, mh) for r, a in zip(raw_y, ay)]
    for ex in (exclude or []):        # exclude = list of (x_idx, y_idx) candidate hit-index pairs
        _apply_exclude(grids_x, ex[0])
        _apply_exclude(grids_y, ex[1])

    # Seed anchors: CMSPixel (index 1), Mimosa2 (index 3) -- longest lever arm from CMSPixel
    x_sel, x_idx = _seed_fit_1d(z_up, grids_x, anchor_idx=(1, 3), return_idx=True)
    y_sel, y_idx = _seed_fit_1d(z_up, grids_y, anchor_idx=(1, 3), return_idx=True)

    # Tilt corrections — track order [M0, CMS, M1, M2]; Mimosa rot indices [0, 1, 2]
    rx = np.zeros(4)
    ry = np.zeros(4)
    if rot_x is not None:
        rx[[0, 2, 3]] = rot_x[[0, 1, 2]]
    if rot_y is not None:
        ry[[0, 2, 3]] = rot_y[[0, 1, 2]]
    if cms_rot is not None:
        rx[1] = cms_rot[0]
        ry[1] = cms_rot[1]

    x0, y0 = x_sel.copy(), y_sel.copy()
    x_up = x0 - rx[:, np.newaxis] * y0
    y_up = y0 - ry[:, np.newaxis] * x0

    with np.errstate(invalid="ignore"):
        b1_upx, b0_upx = np.polyfit(z_up, x_up, 1)
        b1_upy, b0_upy = np.polyfit(z_up, y_up, 1)
    r_x = x_up - (b1_upx[np.newaxis, :] * z_up[:, np.newaxis] + b0_upx[np.newaxis, :])
    r_y = y_up - (b1_upy[np.newaxis, :] * z_up[:, np.newaxis] + b0_upy[np.newaxis, :])
    track_up = namedtuple("track_up", ["b1_upx", "b0_upx", "b1_upy", "b0_upy", "r_x", "r_y"])
    trk = track_up(b1_upx, b0_upx, b1_upy, b0_upy, r_x, r_y)
    return (trk, (x_idx, y_idx)) if return_idx else trk


def downstream_track(reco, dn_valid, align_x=None, align_y=None, rot_x=None, rot_y=None,
                     exclude=None, return_idx=False, max_hits=None):
    """
    3-plane straight-line fit: M3(48.5) → M4(51.5) → M5(53.65).
    Hit selection uses the same seeding track finder as upstream_track (see _seed_fit_1d):
    the seed line is built from (Mimosa3, Mimosa5) -- the outermost pair, longest lever arm
    available on this arm (no CMSPixel-equivalent anchor exists downstream). Mimosa4 is
    matched to the nearest hit consistent with the extrapolated line. dn_valid should be
    downstream_valid(reco) -- see upstream_track docstring for why the 1-2 hit cap stays.
    """
    z_dn = zmimosa[3:]   # [48.5, 51.5, 53.65]

    ax = np.array([
        align_x[3] if align_x is not None else 0.0,
        align_x[4] if align_x is not None else 0.0,
        align_x[5] if align_x is not None else 0.0,
    ])
    ay = np.array([
        align_y[3] if align_y is not None else 0.0,
        align_y[4] if align_y is not None else 0.0,
        align_y[5] if align_y is not None else 0.0,
    ])

    raw_x = [
        reco["Mimosa3_xPosition"].array()[dn_valid],
        reco["Mimosa4_xPosition"].array()[dn_valid],
        reco["Mimosa5_xPosition"].array()[dn_valid],
    ]
    raw_y = [
        reco["Mimosa3_yPosition"].array()[dn_valid],
        reco["Mimosa4_yPosition"].array()[dn_valid],
        reco["Mimosa5_yPosition"].array()[dn_valid],
    ]

    mh = MAX_HITS_PER_PLANE_DN if max_hits is None else max_hits
    grids_x = [_pad_hits(r, a, mh) for r, a in zip(raw_x, ax)]
    grids_y = [_pad_hits(r, a, mh) for r, a in zip(raw_y, ay)]
    for ex in (exclude or []):        # exclude = list of (x_idx, y_idx) candidate hit-index pairs
        _apply_exclude(grids_x, ex[0])
        _apply_exclude(grids_y, ex[1])

    # Seed anchors: Mimosa3 (index 0), Mimosa5 (index 2) -- outermost pair
    x_sel, x_idx = _seed_fit_1d(z_dn, grids_x, anchor_idx=(0, 2), return_idx=True)
    y_sel, y_idx = _seed_fit_1d(z_dn, grids_y, anchor_idx=(0, 2), return_idx=True)

    rx = rot_x[3:] if rot_x is not None else np.zeros(3)
    ry = rot_y[3:] if rot_y is not None else np.zeros(3)

    x0, y0 = x_sel.copy(), y_sel.copy()
    x_dn = x0 - rx[:, np.newaxis] * y0
    y_dn = y0 - ry[:, np.newaxis] * x0

    with np.errstate(invalid="ignore"):
        b1_dnx, b0_dnx = np.polyfit(z_dn, x_dn, 1)
        b1_dny, b0_dny = np.polyfit(z_dn, y_dn, 1)
    r_x = x_dn - (b1_dnx[np.newaxis, :] * z_dn[:, np.newaxis] + b0_dnx[np.newaxis, :])
    r_y = y_dn - (b1_dny[np.newaxis, :] * z_dn[:, np.newaxis] + b0_dny[np.newaxis, :])
    track_dn = namedtuple("track_dn", ["b1_dnx", "b0_dnx", "b1_dny", "b0_dny", "r_x", "r_y"])
    trk = track_dn(b1_dnx, b0_dnx, b1_dny, b0_dny, r_x, r_y)
    return (trk, (x_idx, y_idx)) if return_idx else trk


def track_max_residual(r_x, r_y):
    """
    Max absolute residual across all planes and both coordinates per event.
    r_x, r_y: shape (n_planes, N_events). Returns shape (N_events,).
    A value > MAX_RESIDUAL_CM means at least one hit is > 1 mm from the fitted line.
    """
    return np.max(np.abs(np.vstack([r_x, r_y])), axis=0)


def scatter_angle(b1_upx, b1_upy, b1_dnx, b1_dny):
    theta_x = np.arctan(b1_dnx) - np.arctan(b1_upx)
    theta_y = np.arctan(b1_dny) - np.arctan(b1_upy)
    return np.sqrt(theta_x**2 + theta_y**2)


def target_pair_distance(track_up, track_dn):
    """
    Extrapolate the upstream and downstream tracks to ztarget and return the Euclidean
    distance between the two extrapolated (x, y) positions, per event.
    A small distance means the upstream and downstream tracks intersect at (approximately)
    the same physical point at the target -- i.e. they plausibly belong to the same particle.
    This is the pairing criterion that replaces the old implicit assumption that sharing a
    boolean event mask (`both = up_valid & dn_valid`) was enough to treat the two arms as
    belonging to the same particle, which pile-up events could silently violate.
    """
    x_up_t = track_up.b1_upx * ztarget + track_up.b0_upx
    y_up_t = track_up.b1_upy * ztarget + track_up.b0_upy
    x_dn_t = track_dn.b1_dnx * ztarget + track_dn.b0_dnx
    y_dn_t = track_dn.b1_dny * ztarget + track_dn.b0_dny
    return np.sqrt((x_up_t - x_dn_t)**2 + (y_up_t - y_dn_t)**2)


def post_BRM_valid(reco, up_valid):
    nmm1 = reco["NChamber1_position"].array(library="np")[up_valid]
    nmm2 = reco["NChamber2_position"].array(library="np")[up_valid]
    return (nmm1 >= 1) & (nmm1 <= 2) & (nmm2 >= 1) & (nmm2 <= 2)


def mag_def_angle(reco, track_up, up_valid, post_valid, mm_align=None, track_dn=None):
    pos1 = ak.to_numpy(ak.mean(reco["Chamber1_position"].array()[up_valid][post_valid], axis=1))
    pos2 = ak.to_numpy(ak.mean(reco["Chamber2_position"].array()[up_valid][post_valid], axis=1))
    if mm_align is not None:
        pos1 = pos1 - mm_align[0]
        pos2 = pos2 - mm_align[1]
    detected_pos  = np.stack([pos1, pos2])
    b1_detected, _ = np.polyfit(zmm, detected_pos, 1)
    ref_slope = track_dn.b1_dnx[post_valid] if track_dn is not None else track_up.b1_upx[post_valid]
    return np.arctan(b1_detected) - np.arctan(ref_slope)


def BL_from_current(current_A):
    return (current_A / 1400.0) * 1.35 * 1.0  # T·m

BL_CALIBRATED = BL_from_current(550)

def mag_field_calibration(phi):
    return np.sin(np.nanmedian(phi)) * pelectron / GeV_factor

def momentum_track(phi):
    return GeV_factor * BL_CALIBRATED / np.sin(phi)


def inv_mass_rec(track_up, track_dn, phi, up_valid, dn_valid, post_valid):
    both         = up_valid & dn_valid
    up_in_both   = both[up_valid]
    dn_in_both   = both[dn_valid]
    post_in_both = post_valid[up_in_both]
    pelectron_new = momentum_track(phi)[up_in_both[post_valid]]
    theta = scatter_angle(
        track_up.b1_upx[up_in_both][post_in_both],
        track_up.b1_upy[up_in_both][post_in_both],
        track_dn.b1_dnx[dn_in_both][post_in_both],
        track_dn.b1_dny[dn_in_both][post_in_both],
    )
    W = np.sqrt(
        (mproton + np.sqrt(melectron**2 + pelectron**2)
                 - np.sqrt(melectron**2 + pelectron_new**2))**2
        - (pelectron**2 + pelectron_new**2
           - 2 * pelectron * pelectron_new * np.cos(theta))
    )
    finite = np.isfinite(W)
    return W[finite], theta[finite]


def BW_distro(W, A):
    return A * W * mdelta * Gamma / ((W**2 - mdelta**2)**2 + (mdelta * Gamma)**2)

def BW_plus_bg(W, A, a0, a1):
    return BW_distro(W, A) + a0 + a1 * (W - mdelta)


def unbinned_lrt(W):
    """
    Unbinned maximum likelihood ratio test: BW signal over a linear background.
    t = -2*log(L_null / L_alt) ~ chi2(1 dof) by Wilks' theorem.
    Returns a dict with fit results and plot callables, or None if too few events.
    """
    W_fit = W[(W > W_LO) & (W < W_HI)]
    if len(W_fit) < 10:
        return None

    bw_vals = _bw_unnorm(W_fit) / _C_BW

    def nll_null(params):
        m, = params
        pdf = (1.0 + m * (W_fit - W_MID)) / _C_BG
        if np.any(pdf <= 0):
            return 1e10
        return -np.sum(np.log(pdf))

    def nll_alt(params):
        f_s, m = params
        bg  = (1.0 + m * (W_fit - W_MID)) / _C_BG
        pdf = f_s * bw_vals + (1.0 - f_s) * bg
        if np.any(pdf <= 0):
            return 1e10
        return -np.sum(np.log(pdf))

    res_null = minimize(nll_null, x0=[0.0],       method='L-BFGS-B', bounds=[(-2.85, 2.85)])
    res_alt  = minimize(nll_alt,  x0=[0.01, 0.0], method='L-BFGS-B',
                        bounds=[(0.0, 1.0), (-2.85, 2.85)])

    f_s    = res_alt.x[0]
    m_alt  = res_alt.x[1]
    m_null = res_null.x[0]
    t      = max(2.0 * (res_null.fun - res_alt.fun), 0.0)

    N_fit = len(W_fit)
    bin_w = (W_HI - W_LO) / 35

    def bg_curve(W_arr):
        return (1.0 + m_null * (W_arr - W_MID)) / _C_BG * N_fit * bin_w

    def total_curve(W_arr):
        bw = _bw_unnorm(W_arr) / _C_BW
        bg = (1.0 + m_alt  * (W_arr - W_MID)) / _C_BG
        return (f_s * bw + (1.0 - f_s) * bg) * N_fit * bin_w

    return {
        'f_s': f_s, 'm': m_alt, 't': t,
        'significance': np.sqrt(t),
        'bg_curve': bg_curve, 'total_curve': total_curve, 'N': N_fit,
    }


NOTARGET_PATH = "Runs/No Target Runs/1601728111.root"
NOTARGET_600A = "Runs/No Target Runs/1601727557.root"

TARGET_RUNS = [
    ("Runs/Target Runs/1601730086.root", "1mm W",   550),
    ("Runs/Target Runs/1601742188.root", "0.5mm W", 550),
    ("Runs/Target Runs/1601747831.root", "2cm H2O", 550),
]


def compute_target_pair_reference(align_x, align_y, rot_x, rot_y,
                                   cms_align=None, cms_rot=None, path=NOTARGET_600A):
    """
    Measure the upstream/downstream target-intersection distance distribution on the 600A
    no-target run (no target scattering, so this isolates the achieved track-fit/extrapolation
    resolution) after the residual quality cut only -- no pairing cut, since this is what
    calibrates one. Returns (dist_median, dist_sigma) where dist_sigma = IQR/1.35 (robust
    sigma), mirroring compute_phi_reference's methodology. The tolerance used everywhere else
    is dist_median + TARGET_PAIR_NSIGMA * dist_sigma (one-sided, since distance is >= 0).
    """
    reco = open_tree(path)
    both = upstream_valid(reco) & downstream_valid(reco)

    track_up = upstream_track(reco, both, align_x, align_y, rot_x, rot_y, cms_align, cms_rot)
    track_dn = downstream_track(reco, both, align_x, align_y, rot_x, rot_y)
    res_mask = ((track_max_residual(track_up.r_x, track_up.r_y) < MAX_RESIDUAL_CM) &
                (track_max_residual(track_dn.r_x, track_dn.r_y) < MAX_RESIDUAL_CM))
    both_new       = both.copy()
    both_new[both] = res_mask
    both           = both_new
    track_up = upstream_track(reco, both, align_x, align_y, rot_x, rot_y, cms_align, cms_rot)
    track_dn = downstream_track(reco, both, align_x, align_y, rot_x, rot_y)

    dist        = target_pair_distance(track_up, track_dn)
    dist_median = np.nanmedian(dist)
    q25, q75    = np.nanpercentile(dist, [25, 75])
    dist_sigma  = (q75 - q25) / 1.35
    return dist_median, dist_sigma


def compute_phi_reference(align_x, align_y, mm_align, rot_x, rot_y,
                           cms_align=None, cms_rot=None, path=NOTARGET_600A, pair_tol=None):
    """
    Measure the phi distribution from the 600A no-target run after track cleaning.
    Returns (phi_center, phi_sigma) where phi_sigma = IQR/1.35 (robust sigma).
    The sigma is used as the width of the phi acceptance window for target runs.
    pair_tol: target-intersection pairing tolerance (cm), from compute_target_pair_reference.
    If given, events failing target_pair_distance(...) < pair_tol are rejected before phi
    is computed, so the phi reference isn't contaminated by mismatched upstream/downstream
    pile-up pairs either.
    """
    reco = open_tree(path)
    both = upstream_valid(reco) & downstream_valid(reco)

    track_up = upstream_track(reco, both, align_x, align_y, rot_x, rot_y, cms_align, cms_rot)
    track_dn = downstream_track(reco, both, align_x, align_y, rot_x, rot_y)
    res_mask = ((track_max_residual(track_up.r_x, track_up.r_y) < MAX_RESIDUAL_CM) &
                (track_max_residual(track_dn.r_x, track_dn.r_y) < MAX_RESIDUAL_CM))
    both_new       = both.copy()
    both_new[both] = res_mask
    both           = both_new
    track_up = upstream_track(reco, both, align_x, align_y, rot_x, rot_y, cms_align, cms_rot)
    track_dn = downstream_track(reco, both, align_x, align_y, rot_x, rot_y)

    if pair_tol is not None:
        pair_mask      = target_pair_distance(track_up, track_dn) < pair_tol
        both_new       = both.copy()
        both_new[both] = pair_mask
        both           = both_new
        track_up = track_up._replace(
            b1_upx=track_up.b1_upx[pair_mask], b0_upx=track_up.b0_upx[pair_mask],
            b1_upy=track_up.b1_upy[pair_mask], b0_upy=track_up.b0_upy[pair_mask],
            r_x=track_up.r_x[:, pair_mask], r_y=track_up.r_y[:, pair_mask])
        track_dn = track_dn._replace(
            b1_dnx=track_dn.b1_dnx[pair_mask], b0_dnx=track_dn.b0_dnx[pair_mask],
            b1_dny=track_dn.b1_dny[pair_mask], b0_dny=track_dn.b0_dny[pair_mask],
            r_x=track_dn.r_x[:, pair_mask], r_y=track_dn.r_y[:, pair_mask])

    post_v = post_BRM_valid(reco, both)
    phi    = mag_def_angle(reco, track_up, both, post_v, mm_align, track_dn=track_dn)

    phi_center = np.nanmedian(phi)
    q25, q75   = np.nanpercentile(phi, [25, 75])
    phi_sigma  = (q75 - q25) / 1.35
    return phi_center, phi_sigma


def phi_sanity_check(align_x, align_y, mm_align, rot_x, rot_y,
                      cms_align=None, cms_rot=None,
                      path=NOTARGET_600A, current_A=600, beam_energy=4.6):
    reco   = open_tree(path)
    both   = upstream_valid(reco) & downstream_valid(reco)
    post_v = post_BRM_valid(reco, both)

    track_up = upstream_track(reco, both, align_x, align_y, rot_x, rot_y, cms_align, cms_rot)
    track_dn = downstream_track(reco, both, align_x, align_y, rot_x, rot_y)
    phi      = mag_def_angle(reco, track_up, both, post_v, mm_align, track_dn=track_dn)

    p_beam   = np.sqrt(beam_energy**2 - melectron**2)
    BL_check = BL_from_current(current_A)
    phi_pred = np.arcsin(GeV_factor * BL_check / p_beam)
    phi_med  = np.nanmedian(phi)
    phi_std  = np.nanstd(phi)
    delta    = abs(phi_med - phi_pred) * 1e3

    print(f"\n  Phi sanity check -- {current_A}A no-target run (E = {beam_energy} GeV)")
    print(f"  Predicted phi : {phi_pred*1e3:.2f} mrad  (BL = {BL_check:.4f} T*m)")
    print(f"  Measured  phi : {phi_med*1e3:.2f} +/- {phi_std*1e3:.2f} mrad  (median +/- std)")
    print(f"  Residual      : {delta:.2f} mrad  ({delta / (phi_pred*1e3) * 100:.1f}% of prediction)")

    return phi, phi_pred, BL_check


def run_analysis(reco, label, align_x, align_y, current_A, mm_align=None,
                 rot_x=None, rot_y=None, cms_align=None, cms_rot=None, phi_ref=None,
                 pair_tol=None):
    """
    Full analysis pipeline for one target run:
    1. Select events valid in all upstream planes (M0/CMS/M1/M2) and downstream (M3-5).
    2. Fit tracks via seeding (see upstream_track/downstream_track); reject events where any
       plane has residual > 1 mm.
    3. Pairing cut: reject events where the extrapolated upstream/downstream tracks don't
       meet at the target within pair_tol (see target_pair_distance,
       compute_target_pair_reference). Skipped entirely if pair_tol is None.
    4. Compute phi; cut events outside ±PHI_NSIGMA * sigma_ref of this run's median.
    5. Reconstruct W; run unbinned LRT.
    Returns (phi, W, theta, lrt_results).
    """
    up_valid = upstream_valid(reco)
    dn_valid = downstream_valid(reco)
    both     = up_valid & dn_valid

    track_up = upstream_track(reco, both, align_x, align_y, rot_x, rot_y, cms_align, cms_rot)
    track_dn = downstream_track(reco, both, align_x, align_y, rot_x, rot_y)

    # Max-residual track quality cut: reject if any hit > 1 mm from fitted line
    res_mask       = ((track_max_residual(track_up.r_x, track_up.r_y) < MAX_RESIDUAL_CM) &
                      (track_max_residual(track_dn.r_x, track_dn.r_y) < MAX_RESIDUAL_CM))
    n_before, n_keep = res_mask.size, int(res_mask.sum())
    both_new        = both.copy()
    both_new[both]  = res_mask
    both            = both_new
    track_up = upstream_track(reco, both, align_x, align_y, rot_x, rot_y, cms_align, cms_rot)
    track_dn = downstream_track(reco, both, align_x, align_y, rot_x, rot_y)
    print(f"  Max-residual cut (<{MAX_RESIDUAL_CM*10:.0f}mm): kept {n_keep}/{n_before} "
          f"({100*n_keep/n_before:.1f}%)")

    # Target-intersection pairing cut: reject events where the extrapolated upstream and
    # downstream tracks don't meet at the target within tolerance (see target_pair_distance)
    if pair_tol is not None:
        pair_mask         = target_pair_distance(track_up, track_dn) < pair_tol
        n_pair_before, n_pair_keep = pair_mask.size, int(pair_mask.sum())
        both_new          = both.copy()
        both_new[both]    = pair_mask
        both              = both_new
        track_up = track_up._replace(
            b1_upx=track_up.b1_upx[pair_mask], b0_upx=track_up.b0_upx[pair_mask],
            b1_upy=track_up.b1_upy[pair_mask], b0_upy=track_up.b0_upy[pair_mask],
            r_x=track_up.r_x[:, pair_mask], r_y=track_up.r_y[:, pair_mask])
        track_dn = track_dn._replace(
            b1_dnx=track_dn.b1_dnx[pair_mask], b0_dnx=track_dn.b0_dnx[pair_mask],
            b1_dny=track_dn.b1_dny[pair_mask], b0_dny=track_dn.b0_dny[pair_mask],
            r_x=track_dn.r_x[:, pair_mask], r_y=track_dn.r_y[:, pair_mask])
        print(f"  Target-pairing cut (<{pair_tol*1e4:.1f}um at z={ztarget}): "
              f"kept {n_pair_keep}/{n_pair_before} ({100*n_pair_keep/n_pair_before:.1f}%)")

    post_valid = post_BRM_valid(reco, both)
    phi        = mag_def_angle(reco, track_up, both, post_valid, mm_align, track_dn=track_dn)

    # Phi cut: keep events within ±PHI_NSIGMA robust-sigma of this run's phi median
    if phi_ref is not None:
        _, phi_sigma_ref = phi_ref
        phi_center       = np.nanmedian(phi)
        phi_cut_mask     = np.abs(phi - phi_center) < PHI_NSIGMA * phi_sigma_ref
        n_post_before    = int(post_valid.sum())
        n_post_after     = int(phi_cut_mask.sum())
        post_valid_new            = post_valid.copy()
        post_valid_new[post_valid] = phi_cut_mask
        post_valid = post_valid_new
        phi        = phi[phi_cut_mask]
        print(f"  Phi cut (+/-{PHI_NSIGMA}*sigma_ref = +/-{PHI_NSIGMA*phi_sigma_ref*1e3:.1f}mrad): "
              f"kept {n_post_after}/{n_post_before} ({100*n_post_after/n_post_before:.1f}%)")

    BL_run = mag_field_calibration(phi)

    print(f"\n{'='*55}")
    print(f"  {label}  (BRM {current_A}A)")
    print(f"{'='*55}")
    print(f"  B*L calibrated : {BL_CALIBRATED:.5f} T*m  (run estimate: {BL_run:.5f} T*m)")
    print(f"  Post-BRM events: {post_valid.sum():,}")
    print(f"  phi median/std : {np.nanmedian(phi)*1e3:.3f} / {np.nanstd(phi)*1e3:.3f} mrad")

    W, theta = inv_mass_rec(track_up, track_dn, phi, both, both, post_valid)
    print(f"  Valid W values : {len(W):,}")

    if len(W[(W > W_LO) & (W < W_HI)]) < 10:
        print("  Too few events in fit range")
        return phi, W, theta, None

    lrt = unbinned_lrt(W)
    if lrt is None:
        return phi, W, theta, None

    p_val = 0.5 * (1 - chi2_dist.cdf(lrt['t'], 1)) if lrt['t'] > 0 else 0.5
    print(f"\n  Unbinned LRT : t = {lrt['t']:.3f},  significance = {lrt['significance']:.2f} sigma")
    print(f"  Signal fraction: f_s = {lrt['f_s']:.4f}")
    print(f"  p-value (1-sided): {p_val:.4f}")

    return phi, W, theta, lrt


def main():
    from alignment import (compute_alignment, compute_tilt_alignment,
                           compute_mm_alignment_track, compute_cmspixel_alignment)

    align_x, align_y             = compute_alignment()
    rot_x, rot_y                  = compute_tilt_alignment(align_x=align_x, align_y=align_y)
    mm_align, _, _, mm_rot, _, _ = compute_mm_alignment_track(
        align_x=align_x, align_y=align_y, rot_x=rot_x, rot_y=rot_y)
    cms_align, cms_rot           = compute_cmspixel_alignment(
        align_x=align_x, align_y=align_y, rot_x=rot_x, rot_y=rot_y)
    pair_median, pair_sigma      = compute_target_pair_reference(
        align_x, align_y, rot_x, rot_y, cms_align, cms_rot)
    pair_tol                     = pair_median + TARGET_PAIR_NSIGMA * pair_sigma
    phi_ref                      = compute_phi_reference(
        align_x, align_y, mm_align, rot_x, rot_y, cms_align, cms_rot, pair_tol=pair_tol)

    print(f"\nUsing B*L = {BL_CALIBRATED:.5f} T*m  (spec: BL_from_current at 550 A)")
    print(f"MM alignment (track): MM1 = {mm_align[0]*10:.2f} mm, MM2 = {mm_align[1]*10:.2f} mm")
    print(f"MM tilt [mrad]: MM1 = {mm_rot[0]*1e3:.2f}, MM2 = {mm_rot[1]*1e3:.2f}  [not applied]")
    print(f"CMSPixel alignment: x = {cms_align[0]*10:.2f} mm, y = {cms_align[1]*10:.2f} mm")
    print(f"CMSPixel tilt: rot_x = {cms_rot[0]*1e3:.2f} mrad, rot_y = {cms_rot[1]*1e3:.2f} mrad")
    print(f"Target-pair reference (600A): median = {pair_median*1e4:.1f} um, "
          f"sigma_robust = {pair_sigma*1e4:.1f} um -> tolerance = {pair_tol*1e4:.1f} um "
          f"(+/-{TARGET_PAIR_NSIGMA}*sigma)")
    print(f"Phi reference (600A): center = {phi_ref[0]*1e3:.2f} mrad, "
          f"sigma_robust = {phi_ref[1]*1e3:.2f} mrad")
    print(f"Tilt rot_x [mrad]: {np.round(rot_x*1e3, 1)}")
    print(f"Tilt rot_y [mrad]: {np.round(rot_y*1e3, 1)}")

    phi_sanity_check(align_x, align_y, mm_align, rot_x, rot_y, cms_align, cms_rot)

    for path, label, current_A in TARGET_RUNS:
        reco = open_tree(path)
        run_analysis(reco, label, align_x, align_y, current_A, mm_align, rot_x, rot_y,
                     cms_align=cms_align, cms_rot=cms_rot, phi_ref=phi_ref, pair_tol=pair_tol)

    print("\nRun plots.py to generate all figures.")


if __name__ == "__main__":
    main()
