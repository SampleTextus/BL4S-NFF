#!/usr/bin/env python3
"""
STEP 2: relax the 1-2 hit cap and let the matcher reject the ghosts that flooded the sample
the last time the cap was relaxed (2.4->0.2 sigma). Defenses: (1) each candidate must be
straight (<1mm residual); (2) up/dn candidates must meet at the target within pair_tol;
(3) best-pair-per-event picks one genuine pair, discarding accidental combinations.

Built entirely on the WRAPPED bl4s_nff tracker (baseline untouched, remains the control):
 - relaxed_valid(cap): admit 1..cap hits/plane (cap=2 == baseline population).
 - iterative candidate extraction: re-run the tracker with previous candidates' hits excluded.
 - phi-cut center is FROZEN to the cap=2 baseline median (advisor: the self-referential cut
   would otherwise slide as the sample grows with cap, masquerading as signal).

ACCEPTANCE: controls (0.5mm, H2O) must stay ~0 sigma as cap rises. If they climb, ghosts are
leaking and the cap must stop there. Signal significance is watched but NEVER tuned on.
"""
import numpy as np
import awkward as ak

import bl4s_nff as B


def _ok(trk):
    return B.track_max_residual(trk.r_x, trk.r_y) < B.MAX_RESIDUAL_CM


def relaxed_valid(reco, cap):
    def ok(p):
        nx = reco[f"N{p}_xPosition"].array(library="np")
        ny = reco[f"N{p}_yPosition"].array(library="np")
        return (nx >= 1) & (nx <= cap) & (ny >= 1) & (ny <= cap)
    up = ok("Mimosa0") & ok("Mimosa1") & ok("Mimosa2") & ok("CMSPixel0")
    dn = ok("Mimosa3") & ok("Mimosa4") & ok("Mimosa5")
    return up & dn


def _extract(track_fn, valid, args, kw, max_cand, max_hits):
    """Iteratively pull up to max_cand candidate tracks, excluding prior candidates' hits."""
    cands, excl = [], []
    for _ in range(max_cand):
        trk, idx = track_fn(*args, valid, *kw, exclude=excl, return_idx=True, max_hits=max_hits)
        cands.append(trk)
        excl = excl + [idx]
    return cands


def match_multi(up_cands, dn_cands, pair_tol, N):
    combos, dists = [], []
    with np.errstate(invalid="ignore"):
        for ui, tu in enumerate(up_cands):
            uok = _ok(tu)
            for di, td in enumerate(dn_cands):
                d = B.target_pair_distance(tu, td)
                combos.append((ui, di))
                dists.append(np.where(uok & _ok(td) & (d < pair_tol), d, np.inf))
    D = np.stack(dists)
    best = np.argmin(D, axis=0)
    matched = np.isfinite(D[best, np.arange(N)])
    up = {k: np.full(N, np.nan) for k in ("b1x", "b0x", "b1y", "b0y")}
    dn = {k: np.full(N, np.nan) for k in ("b1x", "b0x", "b1y", "b0y")}
    for c, (ui, di) in enumerate(combos):
        sel = matched & (best == c)
        tu, td = up_cands[ui], dn_cands[di]
        for k, v in zip(up, (tu.b1_upx, tu.b0_upx, tu.b1_upy, tu.b0_upy)):
            up[k][sel] = v[sel]
        for k, v in zip(dn, (td.b1_dnx, td.b0_dnx, td.b1_dny, td.b0_dny)):
            dn[k][sel] = v[sel]
    return up, dn, matched


def mm_select(c1, c2, dn_b1x, dn_b0x, mm_align):
    """Conservative MM hit selection. Default per chamber = ak.mean (== baseline). For events
    with MM pile-up, pick the (MM1,MM2) hit pair whose line back-extrapolates to the magnet
    centre nearest the downstream track's forward extrapolation there (kink-at-centre model),
    BUT only override the mean when that choice beats the runner-up by > 2*eps_mm. eps_mm and
    the thin-lens sagitta offset are self-calibrated from this run's unambiguous single-hit
    events. Returns (h1, h2, n_overridden)."""
    z1, z2, zc = B.zmm[0], B.zmm[1], B.zbrmc
    n1 = ak.to_numpy(ak.num(c1)); n2 = ak.to_numpy(ak.num(c2))
    h1 = ak.to_numpy(ak.mean(c1, axis=1)) - mm_align[0]
    h2 = ak.to_numpy(ak.mean(c2, axis=1)) - mm_align[1]
    x_dn = dn_b1x * zc + dn_b0x
    one = (n1 == 1) & (n2 == 1)
    xmm1 = h1[one] + (h2[one] - h1[one]) / (z2 - z1) * (zc - z1)
    mism = x_dn[one] - xmm1
    med = np.nanmedian(mism); q25, q75 = np.nanpercentile(mism, [25, 75]); eps = (q75 - q25) / 1.35
    c1l, c2l = ak.to_list(c1), ak.to_list(c2)
    n_over = 0
    for i in np.where((n1 >= 2) | (n2 >= 2))[0]:
        a = np.array(c1l[i]) - mm_align[0]; b = np.array(c2l[i]) - mm_align[1]
        pairs = [(x, y) for x in a for y in b]
        xmm = np.array([x + (y - x) / (z2 - z1) * (zc - z1) for x, y in pairs])
        cost = np.abs((x_dn[i] - xmm) - med)
        o = np.argsort(cost)
        if len(cost) >= 2 and (cost[o[1]] - cost[o[0]]) > 2 * eps:
            h1[i], h2[i] = pairs[o[0]]
            n_over += 1
    return h1, h2, n_over


def reconstruct(reco, valid, up, dn, matched, mm_align, phi_sigma_ref, phi_center=None,
                mm_match=False):
    both = valid.copy(); both[valid] = matched
    post = B.post_BRM_valid(reco, both)
    c1 = reco["Chamber1_position"].array()[both][post]
    c2 = reco["Chamber2_position"].array()[both][post]
    if mm_match:
        pos1, pos2, _ = mm_select(c1, c2, dn["b1x"][matched][post], dn["b0x"][matched][post], mm_align)
    else:
        pos1 = ak.to_numpy(ak.mean(c1, axis=1)) - mm_align[0]
        pos2 = ak.to_numpy(ak.mean(c2, axis=1)) - mm_align[1]
    b1_det, _ = np.polyfit(B.zmm, np.stack([pos1, pos2]), 1)
    phi = np.arctan(b1_det) - np.arctan(dn["b1x"][matched][post])
    phi_med = np.nanmedian(phi)
    center = phi_med if phi_center is None else phi_center
    keep = np.abs(phi - center) < B.PHI_NSIGMA * phi_sigma_ref
    phi = phi[keep]
    pe = B.pelectron
    pe_new = B.momentum_track(phi)
    ux = up["b1x"][matched][post][keep]; uy = up["b1y"][matched][post][keep]
    dx = dn["b1x"][matched][post][keep]; dy = dn["b1y"][matched][post][keep]
    thx = np.arctan(dx) - np.arctan(ux)
    thy = np.arctan(dy) - np.arctan(uy)
    theta = np.sqrt(thx**2 + thy**2)
    with np.errstate(invalid="ignore"):
        W = np.sqrt((B.mproton + np.sqrt(B.melectron**2 + pe**2)
                     - np.sqrt(B.melectron**2 + pe_new**2))**2
                    - (pe**2 + pe_new**2 - 2 * pe * pe_new * np.cos(theta)))
    fin = np.isfinite(W)
    # per-event angles are returned alongside W (same finite mask) so downstream studies
    # (sigma_W propagation, smeared-template fits) can run on the matcher selection
    ang = {"phi": phi[fin], "thx": thx[fin], "thy": thy[fin], "theta": theta[fin]}
    return W[fin], phi_med, ang


def run_cap(reco, cap, max_cand, ax, ay, mm, rx, ry, ca, cr, pair_tol, sig_ref, phi_center):
    valid = relaxed_valid(reco, cap)
    N = int(valid.sum())
    up_c = _extract(B.upstream_track, valid, (reco,), (ax, ay, rx, ry, ca, cr), max_cand, cap)
    dn_c = _extract(B.downstream_track, valid, (reco,), (ax, ay, rx, ry), max_cand, cap)
    up, dn, matched = match_multi(up_c, dn_c, pair_tol, N)
    W, _, ang = reconstruct(reco, valid, up, dn, matched, mm, sig_ref, phi_center=phi_center)
    return N, int(matched.sum()), W, ang


def main():
    from alignment import (compute_alignment, compute_tilt_alignment,
                           compute_mm_alignment_track, compute_cmspixel_alignment)
    ax, ay = compute_alignment()
    rx, ry = compute_tilt_alignment(align_x=ax, align_y=ay)
    mm, *_ = compute_mm_alignment_track(align_x=ax, align_y=ay, rot_x=rx, rot_y=ry)
    ca, cr = compute_cmspixel_alignment(align_x=ax, align_y=ay, rot_x=rx, rot_y=ry)
    pm, ps = B.compute_target_pair_reference(ax, ay, rx, ry, ca, cr)
    pair_tol = pm + B.TARGET_PAIR_NSIGMA * ps
    phi_ref = B.compute_phi_reference(ax, ay, mm, rx, ry, ca, cr, pair_tol=pair_tol)
    sig_ref = phi_ref[1]

    # frozen phi-cut center: cap=2, single candidate (== bl4s_nff baseline pairing)
    reco0 = B.open_tree(B.TARGET_RUNS[0][0])
    v0 = relaxed_valid(reco0, 2)
    u0 = _extract(B.upstream_track, v0, (reco0,), (ax, ay, rx, ry, ca, cr), 1, 2)
    d0 = _extract(B.downstream_track, v0, (reco0,), (ax, ay, rx, ry), 1, 2)
    up0, dn0, m0 = match_multi(u0, d0, pair_tol, int(v0.sum()))
    _, phi_center, _ang0 = reconstruct(reco0, v0, up0, dn0, m0, mm, sig_ref)
    print(f"\npair_tol={pair_tol*1e4:.1f}um | sig_ref={sig_ref*1e3:.2f}mrad | "
          f"frozen phi center={phi_center*1e3:.2f}mrad")
    print("STEP 2 cap sweep (phi center frozen). Watch CONTROLS (0.5mm, H2O) -- must stay ~0:\n")
    print(f"  {'run':8s} {'cap':>3s} {'mcand':>5s} {'valid':>8s} {'matched':>8s} {'W':>7s} {'sigma':>6s}")

    configs = [(2, 1), (2, 2), (3, 3), (4, 3)]
    for path, label, _cur in B.TARGET_RUNS:
        reco = B.open_tree(path)
        for cap, mc in configs:
            N, nm, W, _ang = run_cap(reco, cap, mc, ax, ay, mm, rx, ry, ca, cr,
                                     pair_tol, sig_ref, phi_center)
            lrt = B.unbinned_lrt(W)
            sig = lrt['significance'] if lrt else float('nan')
            tag = " <= baseline" if (cap, mc) == (2, 1) else ""
            print(f"  {label:8s} {cap:>3d} {mc:>5d} {N:>8,} {nm:>8,} {len(W):>7,} {sig:>6.2f}{tag}")
        print()


if __name__ == "__main__":
    main()
