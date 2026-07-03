"""
matcher_stats.py -- statistical results on the arm re-pairing (matcher) selection.

Selection: the freeze-sample Phase-B configuration validated in METHOD_COMPARISON.md and
used by plots_matcher.py -- baseline cap=2 event selection with the phi cut frozen on the
baseline (mcand=1) pairing, kinematics from best-pair (mcand=2) re-pairing of the same
events. This is the defensible 2.73 sigma variant; cap relaxation was tested and rejected
(recovered busy events carry no signal), so it is NOT revisited here.

On that selection this script runs:
  1. sigma_W propagation at the Delta peak (analytic, per-event; instrumental resolutions
     from the 600A no-target run, see sigma_w_propagation.py),
  2. the unbinned LRT with both templates: bare Breit-Wigner (must reproduce the 2.73 sigma
     of METHOD_COMPARISON.md -- regression gate) and the resolution-smeared (Voigt-like)
     template with the sigma_W measured in step 1,
  3. toy-MC validation of the significances (background-only pseudo-experiments through the
     identical LRT; Wilks is unreliable here because f_s sits on its >= 0 boundary and the
     smeared template is nearly degenerate with the linear background over the window).
"""
import numpy as np
import awkward as ak

import bl4s_nff as B
import matcher as M
from sigma_w_propagation import analytic_sigma_w, theta_resolution_notarget, PEAK_HALF_WINDOW

N_TOYS = 1000


def _pair(reco, mcand, ax, ay, rx, ry, ca, cr, tol):
    """Baseline-population candidate extraction + best-pair matching (== plots_matcher._pair)."""
    v = M.relaxed_valid(reco, 2)
    N = int(v.sum())
    uc = M._extract(B.upstream_track, v, (reco,), (ax, ay, rx, ry, ca, cr), mcand, 2)
    dc = M._extract(B.downstream_track, v, (reco,), (ax, ay, rx, ry), mcand, 2)
    up, dn, mt = M.match_multi(uc, dc, tol, N)
    return v, up, dn, mt


def phase_b_events(reco, ax, ay, rx, ry, ca, cr, mm, tol, sig_ref, phi_c):
    """Freeze-sample Phase-B reconstruction (mirrors plots_matcher.main's inner loop):
    event set and phi cut from the baseline (mcand=1) pairing, slopes/phi from the
    re-paired (mcand=2) kinematics on the same events. Returns per-event
    (W, phi, theta_x, theta_y, theta) with a common finite mask."""
    z1, z2 = B.zmm[0], B.zmm[1]
    v, up0, dn0, mb = _pair(reco, 1, ax, ay, rx, ry, ca, cr, tol)   # baseline pairing
    _, up, dn, _mp  = _pair(reco, 2, ax, ay, rx, ry, ca, cr, tol)   # Phase-B pairing
    both0 = v.copy(); both0[v] = mb
    post0 = B.post_BRM_valid(reco, both0)
    c1 = reco["Chamber1_position"].array()[both0][post0]
    c2 = reco["Chamber2_position"].array()[both0][post0]
    mms = ((ak.to_numpy(ak.mean(c2, axis=1)) - mm[1])
           - (ak.to_numpy(ak.mean(c1, axis=1)) - mm[0])) / (z2 - z1)
    phi_b = np.arctan(mms) - np.arctan(dn0["b1x"][mb][post0])
    keep  = np.abs(phi_b - phi_c) < B.PHI_NSIGMA * sig_ref           # frozen sample

    dnpx = dn["b1x"][mb][post0]; dnpy = dn["b1y"][mb][post0]
    upx  = up["b1x"][mb][post0]; upy  = up["b1y"][mb][post0]
    phi  = (np.arctan(mms) - np.arctan(dnpx))[keep]
    thx  = np.arctan(dnpx[keep]) - np.arctan(upx[keep])
    thy  = np.arctan(dnpy[keep]) - np.arctan(upy[keep])
    theta = np.sqrt(thx**2 + thy**2)

    pe = B.pelectron
    pn = B.momentum_track(phi)
    with np.errstate(invalid="ignore"):
        W = np.sqrt((B.mproton + np.sqrt(B.melectron**2 + pe**2)
                     - np.sqrt(B.melectron**2 + pn**2))**2
                    - (pe**2 + pn**2 - 2 * pe * pn * np.cos(theta)))
    fin = np.isfinite(W)
    return W[fin], phi[fin], thx[fin], thy[fin], theta[fin]


def main():
    from alignment import (compute_alignment, compute_tilt_alignment,
                           compute_mm_alignment_track, compute_cmspixel_alignment)
    ax, ay = compute_alignment()
    rx, ry = compute_tilt_alignment(align_x=ax, align_y=ay)
    mm, *_ = compute_mm_alignment_track(align_x=ax, align_y=ay, rot_x=rx, rot_y=ry)
    ca, cr = compute_cmspixel_alignment(align_x=ax, align_y=ay, rot_x=rx, rot_y=ry)
    pm, ps = B.compute_target_pair_reference(ax, ay, rx, ry, ca, cr)
    tol = pm + B.TARGET_PAIR_NSIGMA * ps
    phi_ref = B.compute_phi_reference(ax, ay, mm, rx, ry, ca, cr, pair_tol=tol)
    sig_ref = phi_ref[1]

    sig_phi = phi_ref[1]
    sig_tx, sig_ty, _n = theta_resolution_notarget(ax, ay, rx, ry, ca, cr, tol)
    print(f"\nInstrumental resolutions (600A no-target): sigma_phi = {sig_phi*1e3:.2f} mrad, "
          f"sigma_theta_x/y = {sig_tx*1e3:.2f}/{sig_ty*1e3:.2f} mrad")

    # frozen phi-cut center from the 1mm baseline (mcand=1) pairing
    reco0 = B.open_tree(B.TARGET_RUNS[0][0])
    v0, u0, d0, m0 = _pair(reco0, 1, ax, ay, rx, ry, ca, cr, tol)
    _, phi_c, _ang = M.reconstruct(reco0, v0, u0, d0, m0, mm, sig_ref)
    print(f"Frozen phi-cut center (baseline 1mm pairing): {phi_c*1e3:.2f} mrad")

    for path, label, _cur in B.TARGET_RUNS:
        reco = B.open_tree(path)
        W, phi, thx, thy, theta = phase_b_events(
            reco, ax, ay, rx, ry, ca, cr, mm, tol, sig_ref, phi_c)

        print(f"\n{'='*66}")
        print(f"  {label}  (arm re-pairing, freeze-sample Phase-B, N_W = {len(W)})")
        print(f"{'='*66}")

        # 1. sigma_W at the peak, on this selection's own events
        pk = np.abs(W - B.mdelta) < PEAK_HALF_WINDOW
        if pk.sum() >= 5:
            sigW_all, c_phi, c_th = analytic_sigma_w(
                phi[pk], thx[pk], thy[pk], theta[pk], W[pk], sig_phi, sig_tx, sig_ty)
            sigW = float(np.nanmedian(sigW_all))
            print(f"  sigma_W (analytic, {int(pk.sum())} peak events): {sigW*1e3:.0f} MeV  "
                  f"= {sigW/B.Gamma:.2f} * Gamma  "
                  f"(phi: {np.nanmedian(c_phi)*1e3:.0f} MeV, theta: {np.nanmedian(c_th)*1e3:.1f} MeV)")
        else:
            sigW = 0.295
            print(f"  Too few peak events; using sigma_W = {sigW*1e3:.0f} MeV from baseline study")

        # 2 + 3. both templates, each with its toy-MC p-value
        for name, sres in [("bare BW", None), ("smeared", sigW)]:
            lrt = B.unbinned_lrt(W, sigma_res=sres)
            if lrt is None:
                print(f"  {name:8s}: too few events in fit range")
                continue
            line = (f"  {name:8s}: t = {lrt['t']:7.3f}, Wilks = {lrt['significance']:5.2f} sigma, "
                    f"f_s = {lrt['f_s']:.4f}")
            toy = B.toy_mc_pvalue(W, sigma_res=sres, n_toys=N_TOYS)
            if toy is not None:
                line += (f" | toys: p = {toy['p']:.4f} (Wilks p = {toy['p_wilks']:.4f})"
                         f" -> Z = {toy['z']:5.2f} sigma")
            print(line)


if __name__ == "__main__":
    main()
