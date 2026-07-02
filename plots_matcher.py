#!/usr/bin/env python3
"""
Regenerate the W / phi / theta plots using the new arm-re-pairing method (matcher.py),
saved with a `_matcher` suffix alongside the baseline plots from plots.py so the two can be
compared directly (e.g. Runs/Plots/W_1mm_W.png vs W_1mm_W_matcher.png).

Uses the validated freeze-sample configuration: baseline event selection + baseline phi cut,
but Phase-B (best-pair) kinematics — i.e. the defensible 2.73 sigma re-pairing result, not the
phi-cut-inflated 2.89. See METHOD_COMPARISON.md.
"""
import numpy as np
import awkward as ak
import matcher as M
import bl4s_nff as B
from bl4s_nff import (unbinned_lrt, momentum_track, scatter_angle,
                      pelectron, mproton, melectron)
from plots import plot_W, plot_phi, plot_theta
from alignment import (compute_alignment, compute_tilt_alignment,
                       compute_mm_alignment_track, compute_cmspixel_alignment)


def _pair(reco, mcand, ax, ay, rx, ry, ca, cr, tol):
    v = M.relaxed_valid(reco, 2)
    N = int(v.sum())
    uc = M._extract(B.upstream_track, v, (reco,), (ax, ay, rx, ry, ca, cr), mcand, 2)
    dc = M._extract(B.downstream_track, v, (reco,), (ax, ay, rx, ry), mcand, 2)
    up, dn, mt = M.match_multi(uc, dc, tol, N)
    return v, up, dn, mt


def main():
    ax, ay = compute_alignment()
    rx, ry = compute_tilt_alignment(align_x=ax, align_y=ay)
    mm, *_ = compute_mm_alignment_track(align_x=ax, align_y=ay, rot_x=rx, rot_y=ry)
    ca, cr = compute_cmspixel_alignment(align_x=ax, align_y=ay, rot_x=rx, rot_y=ry)
    pm, ps = B.compute_target_pair_reference(ax, ay, rx, ry, ca, cr)
    tol = pm + B.TARGET_PAIR_NSIGMA * ps
    sig_ref = B.compute_phi_reference(ax, ay, mm, rx, ry, ca, cr, pair_tol=tol)[1]
    z1, z2 = B.zmm[0], B.zmm[1]

    # frozen phi-cut center from the 1mm baseline
    reco0 = B.open_tree(B.TARGET_RUNS[0][0])
    v0, u0, d0, m0 = _pair(reco0, 1, ax, ay, rx, ry, ca, cr, tol)
    _, phi_c, _ = M.reconstruct(reco0, v0, u0, d0, m0, mm, sig_ref)

    for path, label, _cur in B.TARGET_RUNS:
        reco = B.open_tree(path)
        v, up0, dn0, mb = _pair(reco, 1, ax, ay, rx, ry, ca, cr, tol)   # baseline pairing
        _, up, dn, _mp = _pair(reco, 2, ax, ay, rx, ry, ca, cr, tol)    # Phase-B pairing
        both0 = v.copy(); both0[v] = mb
        post0 = B.post_BRM_valid(reco, both0)
        c1 = reco["Chamber1_position"].array()[both0][post0]
        c2 = reco["Chamber2_position"].array()[both0][post0]
        mms = ((ak.to_numpy(ak.mean(c2, axis=1)) - mm[1])
               - (ak.to_numpy(ak.mean(c1, axis=1)) - mm[0])) / (z2 - z1)
        dn0x = dn0["b1x"][mb][post0]
        phi_b = np.arctan(mms) - np.arctan(dn0x)
        keep = np.abs(phi_b - phi_c) < B.PHI_NSIGMA * sig_ref            # frozen sample

        dnpx = dn["b1x"][mb][post0]; dnpy = dn["b1y"][mb][post0]
        upx = up["b1x"][mb][post0];  upy = up["b1y"][mb][post0]
        phi = (np.arctan(mms) - np.arctan(dnpx))[keep]
        theta = scatter_angle(upx[keep], upy[keep], dnpx[keep], dnpy[keep])
        pe = pelectron; pn = momentum_track(phi)
        with np.errstate(invalid="ignore"):
            W = np.sqrt((mproton + np.sqrt(melectron**2 + pe**2)
                         - np.sqrt(melectron**2 + pn**2))**2
                        - (pe**2 + pn**2 - 2 * pe * pn * np.cos(theta)))
        fin = np.isfinite(W); W = W[fin]
        lrt = unbinned_lrt(W)
        slug = label.replace(" ", "_").replace("/", "") + "_matcher"
        tag = label + "  (seeding + arm re-pairing)"
        plot_W(W, tag, slug, lrt)
        plot_phi(phi[fin], tag, slug)
        plot_theta(theta[fin], tag, slug)
        print(f"{label}: N_W={len(W)}, significance={lrt['significance']:.2f}, f_s={lrt['f_s']:.4f}")


if __name__ == "__main__":
    main()
