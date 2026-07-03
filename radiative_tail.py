"""
radiative_tail.py -- does a bremsstrahlung energy-loss tail explain the 1mm W excess?

Motivation (target-scaling test): the observed excess pattern across targets follows
thickness in radiation lengths (1mm W : 0.5mm W : 2cm H2O = 0.285 : 0.143 : 0.055 X0,
i.e. 5.1 : 2.6 : 1), NOT proton areal density (1.0 : 0.5 : 1.43), which is what a real
Delta+ signal would follow. The mundane process with exactly the observed scaling:

  An electron that scatters (quasi-)elastically (true W = m_p) but radiates a collinear
  bremsstrahlung photon of energy dE anywhere in the target arrives at the spectrometer
  with p' reduced by dE. The analysis, which infers W from p', reconstructs

      W_rec = sqrt(m_p^2 + 2*m_p*dE)          (exact in the massless/collinear limit;
                                               radiating before or after the scatter
                                               gives the same measured p', so the full
                                               target thickness applies)

  dE = 0.35 GeV lands exactly on the Delta mass, and such events pass the phi cut.

Model, fully derivable:
  - Bethe-Heitler thick-target straggling: after t radiation lengths the retained
    energy fraction y = E'/E0 satisfies  u = ln(1/y) ~ Gamma(a, 1)  with  a = t/ln2
    (this choice of a reproduces <E'> = E0 * e^-t, the definition of X0).
  - Map u -> dE = E0(1 - e^-u) -> W_rec, then convolve with the measured detector
    resolution sigma_W = 295 MeV (sigma_w_propagation.py). The u ~ 0 part of the Gamma
    (no radiation) automatically models the smeared elastic peak leaking into the
    window from W = 0.938, so one component covers both.
  - The SHAPE per target is fixed by that target's t/X0 -- nothing about it is tuned.
    Only the component's overall fraction f_rad floats in the fit.

Fit per run (unbinned, window [W_LO, W_HI], matcher Phase-B selection):
    pdf = f_s * signal(smeared BW) + f_rad * radiative + (1 - f_s - f_rad) * linear
  null: f_s = 0. The LRT for f_s now asks: is there a peak ON TOP of the correctly
  modeled radiative tail? Validated with background-only toys (tail included in toys).
"""
import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.optimize import minimize

import bl4s_nff as B
import matcher as M
from matcher_stats import phase_b_events, _pair

SIGMA_W = 0.295          # measured W resolution (GeV), see sigma_w_propagation.py
E0      = 4.0            # beam energy (GeV) for target runs
LN2     = np.log(2.0)
N_TOYS  = 1000

# thickness in radiation lengths (X0: W 3.504 mm, water 36.08 cm)
T_X0 = {"1mm W": 0.1 / 0.3504, "0.5mm W": 0.05 / 0.3504, "2cm H2O": 2.0 / 36.08}

_RAD_CACHE = {}


def rad_pdf_W(w, t_x0, sigma_res=SIGMA_W):
    """Radiative-tail template: Bethe-Heitler straggling mapped to W_rec, smeared with
    the detector resolution, normalised to unit integral over [W_LO, W_HI].
    Integration in v = u^a (substitution that flattens the u^(a-1) endpoint singularity:
    the Gamma(a) integrand u^(a-1)e^-u du becomes the smooth e^(-v^(1/a)) dv / Gamma(a+1))."""
    key = (round(float(t_x0), 6), round(float(sigma_res), 6))
    if key not in _RAD_CACHE:
        a  = t_x0 / LN2
        v  = np.linspace(1e-12, 18.0**a, 6000)
        u  = v**(1.0 / a)
        dE = E0 * (1.0 - np.exp(-u))
        Wt = np.sqrt(B.mproton**2 + 2.0 * B.mproton * dE)
        wgt = np.exp(-u) / gamma_fn(a + 1.0)
        dv  = v[1] - v[0]
        wg  = np.linspace(B.W_LO, B.W_HI, 1200)
        G   = np.exp(-0.5 * ((wg[:, None] - Wt[None, :]) / sigma_res)**2)
        pdf = G @ wgt * dv / (np.sqrt(2.0 * np.pi) * sigma_res)
        pdf /= np.trapezoid(pdf, wg)
        _RAD_CACHE[key] = (wg, pdf)
    wg, pdf = _RAD_CACHE[key]
    return np.interp(np.asarray(w, dtype=float), wg, pdf)


def lrt_with_tail(W, t_x0, sigma_res=SIGMA_W):
    """Unbinned LRT for a smeared-BW signal over (radiative tail + linear) background.
    Also reports how much the tail itself improves on the linear-only null."""
    W_fit = W[(W > B.W_LO) & (W < B.W_HI)]
    if len(W_fit) < 10:
        return None
    s = B.signal_pdf(W_fit, sigma_res)
    r = rad_pdf_W(W_fit, t_x0, sigma_res)

    def nll(f_s, f_rad, m):
        if f_s < 0 or f_rad < 0 or f_s + f_rad > 1.0:
            return 1e10
        bg  = (1.0 + m * (W_fit - B.W_MID)) / B._C_BG
        pdf = f_s * s + f_rad * r + (1.0 - f_s - f_rad) * bg
        if np.any(pdf <= 0):
            return 1e10
        return -np.sum(np.log(pdf))

    def best(fun, starts, bounds):
        results = [minimize(fun, x0=x0, method='L-BFGS-B', bounds=bounds) for x0 in starts]
        return min(results, key=lambda res: res.fun)

    bnd_m = (-2.85, 2.85)
    res_lin  = best(lambda p: nll(0.0, 0.0, p[0]), [[0.0]], [bnd_m])
    res_null = best(lambda p: nll(0.0, p[0], p[1]),
                    [[0.1, 0.0], [0.5, 0.0]], [(0.0, 1.0), bnd_m])
    res_alt  = best(lambda p: nll(p[0], p[1], p[2]),
                    [[0.01, 0.1, 0.0], [0.3, 0.1, 0.0], [0.01, 0.5, 0.0]],
                    [(0.0, 1.0), (0.0, 1.0), bnd_m])

    return {
        'N': len(W_fit),
        't_tail_vs_linear': max(2.0 * (res_lin.fun - res_null.fun), 0.0),
        'f_rad_null': res_null.x[0], 'm_null': res_null.x[1],
        't': max(2.0 * (res_null.fun - res_alt.fun), 0.0),
        'f_s': res_alt.x[0], 'f_rad': res_alt.x[1], 'm': res_alt.x[2],
    }


def toy_pvalue_with_tail(W, t_x0, sigma_res=SIGMA_W, n_toys=N_TOYS, seed=20260703):
    """Background-only toys where the background INCLUDES the fitted radiative tail."""
    obs = lrt_with_tail(W, t_x0, sigma_res)
    if obs is None:
        return None
    f_rad, m, N = obs['f_rad_null'], obs['m_null'], obs['N']
    a = t_x0 / LN2
    rng = np.random.default_rng(seed)
    pdf_max = (1.0 + abs(m) * max(B.W_HI - B.W_MID, B.W_MID - B.W_LO)) / B._C_BG

    def sample_component(n, radiative):
        out, need = [], n
        while need > 0:
            k = 2 * need + 16
            if radiative:
                u  = rng.gamma(a, 1.0, k)
                dE = E0 * (1.0 - np.exp(-u))
                w  = np.sqrt(B.mproton**2 + 2 * B.mproton * dE) + rng.normal(0, sigma_res, k)
                acc = w[(w > B.W_LO) & (w < B.W_HI)][:need]
            else:
                w = rng.uniform(B.W_LO, B.W_HI, k)
                q = rng.uniform(0.0, pdf_max, k)
                acc = w[q < (1.0 + m * (w - B.W_MID)) / B._C_BG][:need]
            out.append(acc)
            need -= len(acc)
        return np.concatenate(out)

    t_toys = np.empty(n_toys)
    for i in range(n_toys):
        n_rad = rng.binomial(N, f_rad)
        Wt = np.concatenate([sample_component(n_rad, True),
                             sample_component(N - n_rad, False)])
        t_toys[i] = lrt_with_tail(Wt, t_x0, sigma_res)['t']
    p = (1 + np.sum(t_toys >= obs['t'])) / (n_toys + 1)
    return {'t_obs': obs['t'], 'p': p, 't_toys': t_toys}


def main():
    from alignment import (compute_alignment, compute_tilt_alignment,
                           compute_mm_alignment_track, compute_cmspixel_alignment)
    ax, ay = compute_alignment()
    rx, ry = compute_tilt_alignment(align_x=ax, align_y=ay)
    mm, *_ = compute_mm_alignment_track(align_x=ax, align_y=ay, rot_x=rx, rot_y=ry)
    ca, cr = compute_cmspixel_alignment(align_x=ax, align_y=ay, rot_x=rx, rot_y=ry)
    pm, ps = B.compute_target_pair_reference(ax, ay, rx, ry, ca, cr)
    tol = pm + B.TARGET_PAIR_NSIGMA * ps
    sig_ref = B.compute_phi_reference(ax, ay, mm, rx, ry, ca, cr, pair_tol=tol)[1]

    reco0 = B.open_tree(B.TARGET_RUNS[0][0])
    v0, u0, d0, m0 = _pair(reco0, 1, ax, ay, rx, ry, ca, cr, tol)
    _, phi_c, _ = M.reconstruct(reco0, v0, u0, d0, m0, mm, sig_ref)

    print(f"\nRadiative-tail background test (matcher Phase-B selection, "
          f"sigma_W = {SIGMA_W*1e3:.0f} MeV)", flush=True)

    for path, label, _cur in B.TARGET_RUNS:
        reco = B.open_tree(path)
        W, *_ang = phase_b_events(reco, ax, ay, rx, ry, ca, cr, mm, tol, sig_ref, phi_c)
        t_x0 = T_X0[label]

        res = lrt_with_tail(W, t_x0)
        print(f"\n{'='*66}")
        print(f"  {label}  (t/X0 = {t_x0:.3f}, N_fit = {res['N']})")
        print(f"{'='*66}")
        print(f"  tail vs linear-only null : 2*dNLL = {res['t_tail_vs_linear']:7.2f}  "
              f"(does the data even want the tail?)")
        print(f"  f_rad (null fit)         : {res['f_rad_null']:.4f}")
        print(f"  signal over tail+linear  : t = {res['t']:.3f}, "
              f"Wilks = {np.sqrt(res['t']):.2f} sigma, f_s = {res['f_s']:.4f}, "
              f"f_rad = {res['f_rad']:.4f}", flush=True)

        if res['t'] > 0.01:
            toy = toy_pvalue_with_tail(W, t_x0)
            z = float(B.norm_dist.ppf(1 - toy['p']))
            print(f"  toys ({N_TOYS}): p = {toy['p']:.4f} -> Z = {z:.2f} sigma", flush=True)

    # scaling cross-check: fitted f_rad should rise with t/X0 if the tail is real
    print(f"\n  t/X0 ratios (1mm : 0.5mm : H2O) = "
          f"{T_X0['1mm W']/T_X0['2cm H2O']:.1f} : {T_X0['0.5mm W']/T_X0['2cm H2O']:.1f} : 1")


if __name__ == "__main__":
    main()
