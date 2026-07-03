"""
run_10k_toys.py -- high-statistics (10k) toy-MC validation on the 1mm W run.

Three fits, same toy machinery, 10,000 background-only pseudo-experiments each
(p floor 1/10001 -> max measurable ~3.7 sigma; at p ~ 0.005 the toy-statistics
error on Z shrinks to ~+/-0.05 sigma):

  1. OLD selection (single-pair baseline), smeared template -- the Wilks 2.96 sigma
     from the sigma_w study, never toy-validated; directly tests how much of it
     was Wilks bias vs sample difference.
  2. MATCHER selection (freeze-sample Phase-B), bare BW template  (Wilks 2.73).
  3. MATCHER selection, smeared template                          (Wilks 2.50).
"""
import numpy as np

import bl4s_nff as B
from sigma_w_propagation import peak_events, analytic_sigma_w, theta_resolution_notarget
from matcher_stats import phase_b_events, _pair
import matcher as M

N_TOYS = 10_000
PATH_1MM = B.TARGET_RUNS[0][0]


def med_sigma_w(phi, thx, thy, theta, W, sig_phi, sig_tx, sig_ty):
    pk = np.abs(W - B.mdelta) < B.Gamma
    s, _, _ = analytic_sigma_w(phi[pk], thx[pk], thy[pk], theta[pk], W[pk],
                               sig_phi, sig_tx, sig_ty)
    return float(np.nanmedian(s))


def report(tag, W, sigma_res):
    lrt = B.unbinned_lrt(W, sigma_res=sigma_res)
    toy = B.toy_mc_pvalue(W, sigma_res=sigma_res, n_toys=N_TOYS)
    print(f"  {tag:34s}: Wilks = {lrt['significance']:5.2f} | "
          f"toys p = {toy['p']:.5f} -> Z = {toy['z']:5.2f} sigma "
          f"({int(round(toy['p']*(toy['n_toys']+1)-1))} toys >= t_obs)", flush=True)


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
    sig_phi = phi_ref[1]
    sig_tx, sig_ty, _ = theta_resolution_notarget(ax, ay, rx, ry, ca, cr, tol)

    # OLD selection, 1mm W (peak_events already returns peak-only angle arrays)
    phi_o, thx_o, thy_o, th_o, W_pk, W_old = peak_events(
        PATH_1MM, ax, ay, rx, ry, ca, cr, mm, phi_ref, tol)
    s, _, _ = analytic_sigma_w(phi_o, thx_o, thy_o, th_o, W_pk, sig_phi, sig_tx, sig_ty)
    sigW_old = float(np.nanmedian(s))

    # MATCHER selection, 1mm W
    reco0 = B.open_tree(PATH_1MM)
    v0, u0, d0, m0 = _pair(reco0, 1, ax, ay, rx, ry, ca, cr, tol)
    _, phi_c, _ = M.reconstruct(reco0, v0, u0, d0, m0, mm, sig_phi)
    W_m, phi_m, thx_m, thy_m, th_m = phase_b_events(
        reco0, ax, ay, rx, ry, ca, cr, mm, tol, sig_phi, phi_c)
    sigW_m = med_sigma_w(phi_m, thx_m, thy_m, th_m, W_m, sig_phi, sig_tx, sig_ty)

    print(f"\n1mm W, {N_TOYS:,} toys per fit "
          f"(sigma_W old = {sigW_old*1e3:.0f} MeV, matcher = {sigW_m*1e3:.0f} MeV):\n",
          flush=True)
    report("old selection, smeared", W_old, sigW_old)
    report("matcher (Phase-B), bare BW", W_m, None)
    report("matcher (Phase-B), smeared", W_m, sigW_m)


if __name__ == "__main__":
    main()
