"""
sigma_w_propagation.py -- detector resolution propagated to the W spectrum.

Question this answers: how much does instrumental smearing broaden the reconstructed
W distribution at the Delta(1232) peak, compared to the resonance's natural width
Gamma = hbar/tau ~ 117 MeV?

  - If sigma_W << Gamma : the plain Breit-Wigner signal shape in unbinned_lrt is fine.
  - If sigma_W ~  Gamma : the observed line shape is BW (x) Gaussian = Voigt profile,
                          and fitting an unsmeared BW misattributes signal to background.

W depends on exactly two measured angles (see inv_mass_rec):
  phi   -> scattered momentum p' = GeV_factor * BL / sin(phi)
  theta -> opening angle between upstream and downstream tracks (theta_x, theta_y)

Both resolutions are measured on the 600A no-target reference run, where there is no
target physics, so the entire spread is instrumental (tracking + MM + extrapolation +
in-air multiple scattering):
  sigma_phi      : robust IQR/1.35 width from compute_phi_reference (same number the
                   phi acceptance cut is already built from).
  sigma_theta_x/y: robust IQR/1.35 width of arctan(b1_dn) - arctan(b1_up) per projection
                   (true theta = 0 with no target, so any spread is resolution).

Propagation is done per-event on real 1mm-W-run events near the peak (|W - m_Delta| < Gamma),
both analytically (first-order derivatives of the W formula) and by Monte Carlo smearing
(add Gaussian noise of instrumental size to phi/theta_x/theta_y, recompute W, measure the
spread). The MC does not rely on linearisation, so agreement validates the analytic result.

Known approximations (all first-order, stated so they can be defended):
  - phi and theta_x share the downstream x-slope (mag_def_angle uses b1_dnx as reference),
    so their errors are slightly correlated; treated as independent here.
  - The reference run was taken at 4.6 GeV / 600 A vs 4.0 GeV / 550 A for target runs;
    instrumental resolution is current-independent, multiple scattering scales ~1/p (~15%).
  - Multiple scattering *inside the target* also smears theta for target runs. It is not in
    the no-target reference by construction, so it is estimated separately with the Highland
    formula (average traversal = half the target thickness) and reported as an add-on.
"""

import numpy as np

from bl4s_nff import (
    open_tree, upstream_valid, downstream_valid, upstream_track, downstream_track,
    track_max_residual, target_pair_distance, post_BRM_valid, mag_def_angle,
    momentum_track, compute_target_pair_reference, compute_phi_reference, unbinned_lrt,
    MAX_RESIDUAL_CM, TARGET_PAIR_NSIGMA, PHI_NSIGMA,
    mproton, melectron, pelectron, Gamma, mdelta,
    NOTARGET_600A, TARGET_RUNS,
)

PEAK_HALF_WINDOW = Gamma        # events with |W - m_Delta| < Gamma define "at the peak"
N_TOYS           = 400          # MC smearing repetitions per event
RNG              = np.random.default_rng(20260703)


def robust_sigma(x):
    q25, q75 = np.nanpercentile(x, [25, 75])
    return (q75 - q25) / 1.35


def select_tracks(path, align_x, align_y, rot_x, rot_y, cms_align, cms_rot, pair_tol):
    """Track selection identical to run_analysis/compute_phi_reference:
    validity mask -> seeded track fit -> max-residual cut -> refit -> pairing cut."""
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

    return reco, both, track_up, track_dn


def theta_resolution_notarget(align_x, align_y, rot_x, rot_y, cms_align, cms_rot, pair_tol):
    """Per-projection scatter-angle resolution from the no-target run (true theta = 0)."""
    _, _, track_up, track_dn = select_tracks(
        NOTARGET_600A, align_x, align_y, rot_x, rot_y, cms_align, cms_rot, pair_tol)
    thx = np.arctan(track_dn.b1_dnx) - np.arctan(track_up.b1_upx)
    thy = np.arctan(track_dn.b1_dny) - np.arctan(track_up.b1_upy)
    return robust_sigma(thx), robust_sigma(thy), len(thx)


def W_from_angles(phi, theta):
    """The exact inv_mass_rec formula, as a function of the two measured angles."""
    pnew = momentum_track(phi)
    E_b  = np.sqrt(melectron**2 + pelectron**2)
    E_n  = np.sqrt(melectron**2 + pnew**2)
    W2 = ((mproton + E_b - E_n)**2
          - (pelectron**2 + pnew**2 - 2 * pelectron * pnew * np.cos(theta)))
    return np.sqrt(np.where(W2 > 0, W2, np.nan))


def peak_events(path, align_x, align_y, rot_x, rot_y, cms_align, cms_rot,
                mm_align, phi_ref, pair_tol):
    """Reproduce run_analysis's event selection and return per-event (phi, theta_x,
    theta_y, W) for events reconstructed near the Delta peak."""
    reco, both, track_up, track_dn = select_tracks(
        path, align_x, align_y, rot_x, rot_y, cms_align, cms_rot, pair_tol)

    post_valid = post_BRM_valid(reco, both)
    phi        = mag_def_angle(reco, track_up, both, post_valid, mm_align, track_dn=track_dn)

    _, phi_sigma_ref = phi_ref
    phi_cut_mask = np.abs(phi - np.nanmedian(phi)) < PHI_NSIGMA * phi_sigma_ref
    post_valid_new             = post_valid.copy()
    post_valid_new[post_valid] = phi_cut_mask
    post_valid = post_valid_new
    phi        = phi[phi_cut_mask]

    thx = np.arctan(track_dn.b1_dnx[post_valid]) - np.arctan(track_up.b1_upx[post_valid])
    thy = np.arctan(track_dn.b1_dny[post_valid]) - np.arctan(track_up.b1_upy[post_valid])
    theta = np.sqrt(thx**2 + thy**2)
    W     = W_from_angles(phi, theta)

    keep = np.isfinite(W) & (np.abs(W - mdelta) < PEAK_HALF_WINDOW)
    return phi[keep], thx[keep], thy[keep], theta[keep], W[keep], W[np.isfinite(W)]


def analytic_sigma_w(phi, thx, thy, theta, W, sig_phi, sig_tx, sig_ty):
    """First-order error propagation of the W formula, per event.
    W^2 = S^2 - (p_b^2 + p'^2 - 2 p_b p' cos(theta)),  S = m_p + E_b - E'."""
    pnew = momentum_track(phi)
    E_b  = np.sqrt(melectron**2 + pelectron**2)
    E_n  = np.sqrt(melectron**2 + pnew**2)
    S    = mproton + E_b - E_n

    dW_dp     = (-S * pnew / E_n - pnew + pelectron * np.cos(theta)) / W
    dp_dphi   = -pnew / np.tan(phi)                    # d/dphi [ k / sin(phi) ]
    dW_dphi   = dW_dp * dp_dphi
    dW_dtheta = -pelectron * pnew * np.sin(theta) / W

    # theta = hypot(theta_x, theta_y): project the per-axis resolutions onto theta
    sig_theta = np.sqrt((thx / theta * sig_tx)**2 + (thy / theta * sig_ty)**2)

    contrib_phi   = np.abs(dW_dphi) * sig_phi
    contrib_theta = np.abs(dW_dtheta) * sig_theta
    return np.sqrt(contrib_phi**2 + contrib_theta**2), contrib_phi, contrib_theta


def mc_sigma_w(phi, thx, thy, W, sig_phi, sig_tx, sig_ty):
    """Non-linear cross-check: smear the measured angles with instrumental-size Gaussian
    noise, recompute W, and take the robust spread of (W_smeared - W) per event."""
    n = len(phi)
    phi_s = phi[None, :] + RNG.normal(0.0, sig_phi, (N_TOYS, n))
    thx_s = thx[None, :] + RNG.normal(0.0, sig_tx,  (N_TOYS, n))
    thy_s = thy[None, :] + RNG.normal(0.0, sig_ty,  (N_TOYS, n))
    W_s   = W_from_angles(phi_s, np.sqrt(thx_s**2 + thy_s**2))
    dW    = W_s - W[None, :]
    q25, q75 = np.nanpercentile(dW, [25, 75], axis=0)
    return (q75 - q25) / 1.35


def highland_theta0(thickness_cm, X0_cm, p_GeV):
    """Highland multiple-scattering angle (per projection) for an electron of momentum p."""
    x_over_X0 = thickness_cm / X0_cm
    return (13.6e-3 / p_GeV) * np.sqrt(x_over_X0) * (1 + 0.038 * np.log(x_over_X0))


def main():
    from alignment import (compute_alignment, compute_tilt_alignment,
                           compute_mm_alignment_track, compute_cmspixel_alignment)

    align_x, align_y             = compute_alignment()
    rot_x, rot_y                 = compute_tilt_alignment(align_x=align_x, align_y=align_y)
    mm_align, _, _, _, _, _      = compute_mm_alignment_track(
        align_x=align_x, align_y=align_y, rot_x=rot_x, rot_y=rot_y)
    cms_align, cms_rot           = compute_cmspixel_alignment(
        align_x=align_x, align_y=align_y, rot_x=rot_x, rot_y=rot_y)
    pair_median, pair_sigma      = compute_target_pair_reference(
        align_x, align_y, rot_x, rot_y, cms_align, cms_rot)
    pair_tol                     = pair_median + TARGET_PAIR_NSIGMA * pair_sigma
    phi_ref                      = compute_phi_reference(
        align_x, align_y, mm_align, rot_x, rot_y, cms_align, cms_rot, pair_tol=pair_tol)

    sig_phi = phi_ref[1]
    sig_tx, sig_ty, n_ref = theta_resolution_notarget(
        align_x, align_y, rot_x, rot_y, cms_align, cms_rot, pair_tol)

    print(f"\n{'='*62}")
    print("  Instrumental resolutions (600A no-target reference run)")
    print(f"{'='*62}")
    print(f"  sigma_phi     : {sig_phi*1e3:.3f} mrad   (compute_phi_reference)")
    print(f"  sigma_theta_x : {sig_tx*1e3:.3f} mrad   ({n_ref} events)")
    print(f"  sigma_theta_y : {sig_ty*1e3:.3f} mrad")

    for path, label, _current in TARGET_RUNS:
        phi, thx, thy, theta, W, W_all = peak_events(
            path, align_x, align_y, rot_x, rot_y, cms_align, cms_rot,
            mm_align, phi_ref, pair_tol)

        print(f"\n{'='*62}")
        print(f"  {label}: sigma_W at the Delta peak "
              f"(|W - {mdelta:.3f}| < {PEAK_HALF_WINDOW:.3f} GeV, {len(W)} events)")
        print(f"{'='*62}")
        if len(W) < 5:
            print("  Too few peak events to propagate.")
            continue

        sigW, c_phi, c_theta = analytic_sigma_w(phi, thx, thy, theta, W,
                                                sig_phi, sig_tx, sig_ty)
        sigW_mc = mc_sigma_w(phi, thx, thy, W, sig_phi, sig_tx, sig_ty)

        med = np.nanmedian
        print(f"  analytic  sigma_W : {med(sigW)*1e3:6.1f} MeV   "
              f"[16-84%: {np.nanpercentile(sigW,16)*1e3:.1f} - {np.nanpercentile(sigW,84)*1e3:.1f}]")
        print(f"    from phi  (p')  : {med(c_phi)*1e3:6.1f} MeV")
        print(f"    from theta      : {med(c_theta)*1e3:6.1f} MeV")
        print(f"  MC smear  sigma_W : {med(sigW_mc)*1e3:6.1f} MeV   (validates linearisation)")
        print(f"  Gamma (natural)   : {Gamma*1e3:6.1f} MeV")
        print(f"  sigma_W / Gamma   : {med(sigW)/Gamma:6.2f}")

        fL = Gamma
        fG = 2.355 * med(sigW)
        fwhm_voigt = 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
        print(f"  Observable peak FWHM if Voigt: {fwhm_voigt*1e3:.0f} MeV "
              f"(vs {fL*1e3:.0f} MeV natural)")

        # Step-4 comparison: same unbinned LRT, bare BW template vs the
        # resolution-smeared (Voigt-like) template with the sigma_W measured above.
        lrt_bare  = unbinned_lrt(W_all)
        lrt_voigt = unbinned_lrt(W_all, sigma_res=med(sigW))
        for name, lrt in [("bare BW ", lrt_bare), ("smeared ", lrt_voigt)]:
            if lrt is None:
                print(f"  LRT {name}: too few events")
                continue
            print(f"  LRT {name}: t = {lrt['t']:7.3f}, significance = "
                  f"{lrt['significance']:5.2f} sigma, f_s = {lrt['f_s']:.4f}  "
                  f"(N = {lrt['N']})")

    # Target multiple scattering: not in the no-target reference by construction.
    # Signal electrons scatter somewhere inside the target, so on average traverse
    # half the thickness. X0: W = 0.3504 cm, water = 36.08 cm.
    print(f"\n{'='*62}")
    print("  Add-on: multiple scattering inside the target (Highland, half thickness)")
    print(f"{'='*62}")
    for label, half_cm, X0_cm in [("1mm W",   0.05, 0.3504),
                                  ("0.5mm W", 0.025, 0.3504),
                                  ("2cm H2O", 1.0,  36.08)]:
        th0 = highland_theta0(half_cm, X0_cm, 4.0)
        print(f"  {label:8s}: theta_0 = {th0*1e3:.2f} mrad per projection "
              f"(compare sigma_theta_x/y above)")


if __name__ == "__main__":
    main()
