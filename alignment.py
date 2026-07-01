#!/usr/bin/env python3
"""
Detector alignment using the 0A BRM no-target run (1601728111, 4.6 GeV).
With B=0 the beam is undeflected through all planes, giving geometrically valid
alignment for all six Mimosa planes and both MM chambers in one pass.

Alignment order (must be respected by callers):
  1. compute_alignment()          — Mimosa translational (per-plane mean)
  2. compute_tilt_alignment()     — Mimosa in-plane rotation (leave-one-out, about z)
  3. compute_mm_alignment_track() — MM translational via Mimosa track extrapolation
"""

import numpy as np
import awkward as ak
from bl4s_nff import (
    open_tree, upstream_valid, downstream_valid, post_BRM_valid,
    zmimosa, zmm, zcmspixel,
)

ALIGN_PATH = "Runs/No Target Runs/1601728111.root"


def compute_alignment(path=ALIGN_PATH, n_iterations=20, tol=1e-7):
    """
    Mimosa translational alignment via iterative track-based residual correction (0A run).
    Per iteration: fit a straight line through all 6 planes per event, compute mean residual
    (hit − track prediction) at each plane, subtract as correction. Plane 0 is fixed as the
    reference (align_x[0] = align_y[0] = 0 always) to remove the global-translation weak mode.
    Returns (align_x, align_y): 6-element arrays in cm.
    """
    reco   = open_tree(path)
    both_v = upstream_valid(reco) & downstream_valid(reco)

    planes_x = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_xPosition"].array()[both_v], axis=1))
        for i in range(6)
    ])
    planes_y = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_yPosition"].array()[both_v], axis=1))
        for i in range(6)
    ])

    align_x = np.zeros(6)
    align_y = np.zeros(6)

    for it in range(n_iterations):
        cx = planes_x - align_x[:, np.newaxis]
        cy = planes_y - align_y[:, np.newaxis]

        coeff_x = np.polyfit(zmimosa, cx, 1)   # shape (2, n_events)
        coeff_y = np.polyfit(zmimosa, cy, 1)

        delta_x = np.array([
            np.mean(cx[i] - (coeff_x[0] * zmimosa[i] + coeff_x[1]))
            for i in range(6)
        ])
        delta_y = np.array([
            np.mean(cy[i] - (coeff_y[0] * zmimosa[i] + coeff_y[1]))
            for i in range(6)
        ])

        delta_x -= delta_x[0]   # fix plane 0 as reference
        delta_y -= delta_y[0]

        align_x += delta_x
        align_y += delta_y

        if np.max(np.abs(delta_x)) < tol and np.max(np.abs(delta_y)) < tol:
            print(f"  Translational alignment converged after {it + 1} iteration(s)")
            break

    return align_x, align_y



def compute_mm_alignment_track(path=ALIGN_PATH, align_x=None, align_y=None,
                                rot_x=None, rot_y=None):
    """
    MM translational alignment via Mimosa track extrapolation (0A run).
    Fits a 6-plane Mimosa straight-line track per event (with full Mimosa alignment
    and tilt correction applied), extrapolates to each MM z-position, and computes
    mean(MM_hit - predicted) as the alignment offset.

    Also computes an MM in-plane rotation estimate from the slope of
    (x_residual vs y_pred) at each MM — the same diagnostic used for Mimosa tilt.

    Must be called AFTER compute_alignment and compute_tilt_alignment.
    Returns (mm_align, res1, res2, mm_rot, y_pred1, y_pred2):
        mm_align  : [MM1, MM2] translational offsets in cm
        res1, res2: per-event x-residuals (MM_hit - x_pred) in cm
        mm_rot    : [MM1, MM2] rotation coefficients (from res_x vs y_pred)
        y_pred1/2 : per-event y predictions at MM1/MM2 positions from Mimosa track
    """
    reco   = open_tree(path)
    both_v = upstream_valid(reco) & downstream_valid(reco)
    post_v = post_BRM_valid(reco, both_v)

    planes_x = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_xPosition"].array()[both_v], axis=1))
        for i in range(6)
    ])
    planes_y = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_yPosition"].array()[both_v], axis=1))
        for i in range(6)
    ])

    if align_x is not None:
        planes_x = planes_x - align_x[:, np.newaxis]
    if align_y is not None:
        planes_y = planes_y - align_y[:, np.newaxis]

    if rot_x is not None and rot_y is not None:
        x0, y0 = planes_x.copy(), planes_y.copy()
        planes_x = x0 - rot_x[:, np.newaxis] * y0
        planes_y = y0 - rot_y[:, np.newaxis] * x0

    coeff_x = np.polyfit(zmimosa, planes_x, 1)
    coeff_y = np.polyfit(zmimosa, planes_y, 1)

    x_pred1 = coeff_x[0][post_v] * zmm[0] + coeff_x[1][post_v]
    x_pred2 = coeff_x[0][post_v] * zmm[1] + coeff_x[1][post_v]
    y_pred1 = coeff_y[0][post_v] * zmm[0] + coeff_y[1][post_v]
    y_pred2 = coeff_y[0][post_v] * zmm[1] + coeff_y[1][post_v]

    pos1 = ak.to_numpy(ak.mean(reco["Chamber1_position"].array()[both_v][post_v], axis=1))
    pos2 = ak.to_numpy(ak.mean(reco["Chamber2_position"].array()[both_v][post_v], axis=1))

    res1 = pos1 - x_pred1
    res2 = pos2 - x_pred2
    mm_align = np.array([np.mean(res1), np.mean(res2)])

    res1_corr = res1 - mm_align[0]
    res2_corr = res2 - mm_align[1]
    mm_rot = np.array([
        np.polyfit(y_pred1, res1_corr, 1)[0],
        np.polyfit(y_pred2, res2_corr, 1)[0],
    ])

    return mm_align, res1, res2, mm_rot, y_pred1, y_pred2


def compute_tilt_alignment(path=ALIGN_PATH, align_x=None, align_y=None):
    """
    In-plane rotation coefficients for Mimosa planes 0-5 (leave-one-out, 0A run).
    For each plane: fit a straight track through the other 5, compute residuals,
    then fit residual_x vs y_pred and residual_y vs x_pred.
    These slopes are the in-plane rotation angles (about z) for each plane.
    Returns (rot_x, rot_y): 6-element arrays.
    Apply as: x_corr = x_aligned - rot_x[i]*y_aligned
              y_corr = y_aligned - rot_y[i]*x_aligned
    """
    if align_x is None or align_y is None:
        align_x, align_y = compute_alignment(path)

    reco   = open_tree(path)
    both_v = upstream_valid(reco) & downstream_valid(reco)

    planes_x = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_xPosition"].array()[both_v], axis=1))
        for i in range(6)
    ]) - align_x[:, np.newaxis]

    planes_y = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_yPosition"].array()[both_v], axis=1))
        for i in range(6)
    ]) - align_y[:, np.newaxis]

    rot_x = np.zeros(6)
    rot_y = np.zeros(6)

    for i in range(6):
        others = [j for j in range(6) if j != i]
        z_oth  = zmimosa[others]
        cx = np.polyfit(z_oth, planes_x[others], 1)
        cy = np.polyfit(z_oth, planes_y[others], 1)
        x_pred = cx[0] * zmimosa[i] + cx[1]
        y_pred = cy[0] * zmimosa[i] + cy[1]
        rot_x[i] = np.polyfit(y_pred, planes_x[i] - x_pred, 1)[0]
        rot_y[i] = np.polyfit(x_pred, planes_y[i] - y_pred, 1)[0]

    return rot_x, rot_y


def compute_cmspixel_alignment(path=ALIGN_PATH, align_x=None, align_y=None,
                               rot_x=None, rot_y=None):
    """
    CMSPixel translational and tilt alignment via 6-plane Mimosa track interpolation (0A run).
    CMSPixel is at z=36.9 cm, between M0 and M1 — well inside the Mimosa arm so this is
    interpolation (not extrapolation), giving a precise prediction.
    Returns (cms_align, cms_rot):
      cms_align: [x_offset, y_offset] in cm — subtract from raw CMSPixel positions
      cms_rot:   [rot_x, rot_y] in-plane rotation angles (dimensionless, radians)
    """
    if align_x is None:
        align_x, align_y = compute_alignment(path)
    if rot_x is None:
        rot_x, rot_y = compute_tilt_alignment(path, align_x=align_x, align_y=align_y)

    reco   = open_tree(path)
    both_v = upstream_valid(reco) & downstream_valid(reco)

    planes_x = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_xPosition"].array()[both_v], axis=1))
        for i in range(6)
    ]) - align_x[:, np.newaxis]
    planes_y = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_yPosition"].array()[both_v], axis=1))
        for i in range(6)
    ]) - align_y[:, np.newaxis]

    x0, y0   = planes_x.copy(), planes_y.copy()
    planes_x = x0 - rot_x[:, np.newaxis] * y0
    planes_y = y0 - rot_y[:, np.newaxis] * x0

    coeff_x = np.polyfit(zmimosa, planes_x, 1)   # (2, N)
    coeff_y = np.polyfit(zmimosa, planes_y, 1)
    x_pred  = coeff_x[0] * zcmspixel + coeff_x[1]
    y_pred  = coeff_y[0] * zcmspixel + coeff_y[1]

    x_cms = ak.to_numpy(ak.mean(reco["CMSPixel0_xPosition"].array()[both_v], axis=1))
    y_cms = ak.to_numpy(ak.mean(reco["CMSPixel0_yPosition"].array()[both_v], axis=1))

    res_x   = x_cms - x_pred
    res_y   = y_cms - y_pred
    cms_align = np.array([np.mean(res_x), np.mean(res_y)])

    res_x_c = res_x - cms_align[0]
    res_y_c = res_y - cms_align[1]
    cms_rot = np.array([
        np.polyfit(y_pred, res_x_c, 1)[0],
        np.polyfit(x_pred, res_y_c, 1)[0],
    ])

    return cms_align, cms_rot


def compute_track_resolution(path=ALIGN_PATH, align_x=None, align_y=None,
                              rot_x=None, rot_y=None):
    """
    Estimate per-plane position resolution (sigma) from 3-plane arm fits on the 0A run.
    Fits the upstream arm (planes 0-2) and downstream arm (planes 3-5) independently,
    then returns the RMS of polyfit residuals per plane.
    Must be called AFTER compute_alignment and compute_tilt_alignment.
    Returns (sigma_x, sigma_y): 6-element arrays in cm.
    """
    if align_x is None:
        align_x, align_y = compute_alignment(path)
    if rot_x is None:
        rot_x, rot_y = compute_tilt_alignment(path, align_x=align_x, align_y=align_y)

    reco   = open_tree(path)
    both_v = upstream_valid(reco) & downstream_valid(reco)

    planes_x = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_xPosition"].array()[both_v], axis=1))
        for i in range(6)
    ]) - align_x[:, np.newaxis]
    planes_y = np.array([
        ak.to_numpy(ak.mean(reco[f"Mimosa{i}_yPosition"].array()[both_v], axis=1))
        for i in range(6)
    ]) - align_y[:, np.newaxis]

    x0, y0   = planes_x.copy(), planes_y.copy()
    planes_x = x0 - rot_x[:, np.newaxis] * y0
    planes_y = y0 - rot_y[:, np.newaxis] * x0

    sigma_x = np.zeros(6)
    sigma_y = np.zeros(6)

    for arm_planes, z_arm in [([0, 1, 2], zmimosa[:3]), ([3, 4, 5], zmimosa[3:])]:
        px = planes_x[arm_planes]   # shape (3, N)
        py = planes_y[arm_planes]
        cx = np.polyfit(z_arm, px, 1)   # (2, N): slopes and intercepts per event
        cy = np.polyfit(z_arm, py, 1)
        for k, i in enumerate(arm_planes):
            r_x = px[k] - (cx[0] * z_arm[k] + cx[1])
            r_y = py[k] - (cy[0] * z_arm[k] + cy[1])
            sigma_x[i] = np.std(r_x)
            sigma_y[i] = np.std(r_y)

    return sigma_x, sigma_y


def main():
    align_x, align_y                         = compute_alignment()
    rot_x, rot_y                              = compute_tilt_alignment(align_x=align_x, align_y=align_y)
    mm_align, _, _, mm_rot, _, _             = compute_mm_alignment_track(
        align_x=align_x, align_y=align_y, rot_x=rot_x, rot_y=rot_y)
    cms_align, cms_rot                        = compute_cmspixel_alignment(
        align_x=align_x, align_y=align_y, rot_x=rot_x, rot_y=rot_y)

    print("\n" + "="*70)
    print("  Detector alignment  --  0A BRM, 4.6 GeV, no target (1601728111)")
    print("="*70)
    print(f"  {'Plane':<12} {'align_x (cm)':>14} {'align_y (cm)':>14} "
          f"{'rot_x (mrad)':>14} {'rot_y (mrad)':>14}")
    print(f"  {'-'*60}")
    for i in range(6):
        print(f"  {'Mimosa '+str(i):<12} {align_x[i]:>14.4f} {align_y[i]:>14.4f} "
              f"{rot_x[i]*1e3:>14.2f} {rot_y[i]*1e3:>14.2f}")
    print(f"  {'CMSPixel':<12} {cms_align[0]:>14.4f} {cms_align[1]:>14.4f} "
          f"{cms_rot[0]*1e3:>14.2f} {cms_rot[1]*1e3:>14.2f}")
    print(f"  {'-'*60}")
    print(f"  {'MM1':<12} {mm_align[0]:>14.4f} cm  "
          f"rot = {mm_rot[0]*1e3:.2f} mrad  (x_res vs y_pred)")
    print(f"  {'MM2':<12} {mm_align[1]:>14.4f} cm  "
          f"rot = {mm_rot[1]*1e3:.2f} mrad  (x_res vs y_pred)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
