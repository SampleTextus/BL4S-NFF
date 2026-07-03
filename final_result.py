"""
final_result.py -- closes the Delta+(1232) search on the 1mm W run.

With the radiative-tail background in the model the residual signal is 1.0 sigma
(radiative_tail.py), so the search ends in an upper limit:

  1. 90% CL upper limit on the signal fraction f_s -- profile-likelihood scan
     q(f_s) = 2*[NLL(f_s; profiled f_rad, m) - NLL(best)], one-sided. The asymptotic
     crossing (q = 1.64) is then CALIBRATED WITH TOYS: at each scanned f_s, generate
     signal+background pseudo-experiments at that f_s and find where
     P(q_toy >= q_obs) = 10% -- the same distrust-Wilks policy used for the p-values.
  2. The money plot: the W spectrum with the fitted background decomposition
     (linear + radiative tail) and the just-excluded Delta+ signal overlaid, plus
     the limit scan itself.

Selection: matcher Phase-B; signal: resolution-smeared BW (sigma_W = 295 MeV);
background: Bethe-Heitler radiative tail (shape fixed by t/X0) + linear.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, brentq

import bl4s_nff as B
import matcher as M
from matcher_stats import phase_b_events, _pair
from radiative_tail import rad_pdf_W, lrt_with_tail, SIGMA_W, T_X0

OUTDIR       = Path("Runs/Plots")
CL           = 0.90
Q_ASYMPTOTIC = B.norm_dist.ppf(CL)**2          # 1.64: one-sided asymptotic threshold
N_TOYS_SCAN  = 300                             # per bisection point
N_TOYS_FINAL = 600                             # at the final limit
RNG          = np.random.default_rng(20260703)


def make_sampler(pdf_vals_on_grid, wg):
    """Inverse-CDF sampler for a pdf tabulated on wg over the fit window."""
    dw  = np.diff(wg)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf_vals_on_grid[1:] + pdf_vals_on_grid[:-1]) * dw)])
    cdf /= cdf[-1]
    return lambda n: np.interp(RNG.uniform(0.0, 1.0, n), cdf, wg)


def build_fit(W, t_x0):
    """Return the NLL pieces for signal + radiative tail + linear on this dataset."""
    W_fit = W[(W > B.W_LO) & (W < B.W_HI)]
    s = B.signal_pdf(W_fit, SIGMA_W)
    r = rad_pdf_W(W_fit, t_x0, SIGMA_W)

    def nll(f_s, f_rad, m):
        if f_s < 0 or f_rad < 0 or f_s + f_rad > 1.0:
            return 1e10
        bg  = (1.0 + m * (W_fit - B.W_MID)) / B._C_BG
        pdf = f_s * s + f_rad * r + (1.0 - f_s - f_rad) * bg
        if np.any(pdf <= 0):
            return 1e10
        return -np.sum(np.log(pdf))

    def fit_free():
        starts = [[0.01, 0.1, 0.0], [0.3, 0.1, 0.0], [0.01, 0.5, 0.0]]
        res = min((minimize(lambda p: nll(*p), x0=x0, method='L-BFGS-B',
                            bounds=[(0, 1), (0, 1), (-2.85, 2.85)]) for x0 in starts),
                  key=lambda r_: r_.fun)
        return res.fun, res.x

    def fit_fixed(f_s):
        # starting points must satisfy f_s + f_rad <= 1 or L-BFGS-B sticks at the penalty
        starts = [[min(0.1, 0.9 * (1.0 - f_s)), 0.0], [0.5 * (1.0 - f_s), 0.0]]
        res = min((minimize(lambda p: nll(f_s, p[0], p[1]), x0=x0, method='L-BFGS-B',
                            bounds=[(0, 1), (-2.85, 2.85)]) for x0 in starts),
                  key=lambda r_: r_.fun)
        return res.fun, res.x

    return W_fit, nll, fit_free, fit_fixed


def toy_pvalue_at(f_s, fit_ctx, t_x0, n_toys):
    """P(q_toy(f_s) >= q_obs(f_s)) with toys generated at (f_s, profiled nuisances)."""
    W_fit, _nll, fit_free, fit_fixed = fit_ctx
    N = len(W_fit)
    nll_hat, x_hat = fit_free()
    nll_fix, (f_rad_p, m_p) = fit_fixed(f_s)
    q_obs = max(0.0, 2.0 * (nll_fix - nll_hat)) if x_hat[0] <= f_s else 0.0

    wg = np.linspace(B.W_LO, B.W_HI, 2000)
    draw_sig = make_sampler(B.signal_pdf(wg, SIGMA_W), wg)
    draw_rad = make_sampler(rad_pdf_W(wg, t_x0, SIGMA_W), wg)
    draw_lin = make_sampler((1.0 + m_p * (wg - B.W_MID)) / B._C_BG, wg)

    fracs = np.array([f_s, f_rad_p, 1.0 - f_s - f_rad_p])
    n_exceed = 0
    for _ in range(n_toys):
        ns, nr, nl = RNG.multinomial(N, fracs)
        Wt = np.concatenate([draw_sig(ns), draw_rad(nr), draw_lin(nl)])
        _, nll_t, free_t, fixed_t = build_fit(Wt, t_x0)
        fun_hat_t, x_hat_t = free_t()
        fun_fix_t, _ = fixed_t(f_s)
        q_t = max(0.0, 2.0 * (fun_fix_t - fun_hat_t)) if x_hat_t[0] <= f_s else 0.0
        n_exceed += (q_t >= q_obs)
    return n_exceed / n_toys, q_obs


def main():
    from alignment import (compute_alignment, compute_tilt_alignment,
                           compute_mm_alignment_track, compute_cmspixel_alignment)
    ax_, ay = compute_alignment()
    rx, ry = compute_tilt_alignment(align_x=ax_, align_y=ay)
    mm, *_ = compute_mm_alignment_track(align_x=ax_, align_y=ay, rot_x=rx, rot_y=ry)
    ca, cr = compute_cmspixel_alignment(align_x=ax_, align_y=ay, rot_x=rx, rot_y=ry)
    pm, ps = B.compute_target_pair_reference(ax_, ay, rx, ry, ca, cr)
    tol = pm + B.TARGET_PAIR_NSIGMA * ps
    sig_ref = B.compute_phi_reference(ax_, ay, mm, rx, ry, ca, cr, pair_tol=tol)[1]

    reco0 = B.open_tree(B.TARGET_RUNS[0][0])
    v0, u0, d0, m0 = _pair(reco0, 1, ax_, ay, rx, ry, ca, cr, tol)
    _, phi_c, _ = M.reconstruct(reco0, v0, u0, d0, m0, mm, sig_ref)

    label, t_x0 = "1mm W", T_X0["1mm W"]
    W, *_ang = phase_b_events(reco0, ax_, ay, rx, ry, ca, cr, mm, tol, sig_ref, phi_c)
    base = lrt_with_tail(W, t_x0)

    fit_ctx = build_fit(W, t_x0)
    W_fit, _nll, fit_free, fit_fixed = fit_ctx
    nll_hat, (fs_hat, fr_hat, m_hat) = fit_free()
    N = len(W_fit)

    def q_of(fs):
        if fs < fs_hat:
            return 0.0
        return max(0.0, 2.0 * (fit_fixed(fs)[0] - nll_hat))

    # asymptotic 90% upper limit
    fs_hi = 0.99
    fs90_asym = brentq(lambda f: q_of(f) - Q_ASYMPTOTIC, fs_hat + 1e-6, fs_hi, xtol=1e-4)
    print(f"\n{label}: N_fit = {N}, fs_hat = {fs_hat:.4f}, "
          f"asymptotic {CL:.0%} UL on f_s = {fs90_asym:.4f}", flush=True)

    # toy calibration: bisect p(fs) = 1 - CL
    lo, hi = max(fs_hat + 0.01, 0.5 * fs90_asym), min(0.99, 2.0 * fs90_asym)
    for _ in range(4):
        mid = 0.5 * (lo + hi)
        p_mid, _q = toy_pvalue_at(mid, fit_ctx, t_x0, N_TOYS_SCAN)
        print(f"  scan f_s = {mid:.4f}: toy p = {p_mid:.3f}", flush=True)
        if p_mid > 1.0 - CL:
            lo = mid          # limit lies above: data still compatible here
        else:
            hi = mid
    fs90 = 0.5 * (lo + hi)
    p_fin, q_fin = toy_pvalue_at(fs90, fit_ctx, t_x0, N_TOYS_FINAL)
    print(f"  toy-calibrated {CL:.0%} UL: f_s < {fs90:.4f} "
          f"(p = {p_fin:.3f} at limit, q_obs = {q_fin:.2f}) "
          f"-> N_sig < {fs90*N:.0f} events", flush=True)

    # ---- money plot -----------------------------------------------------------
    _, (fr90, m90) = fit_fixed(fs90)
    bins = np.linspace(B.W_LO, B.W_HI, 36)
    bin_w = bins[1] - bins[0]
    counts, edges = np.histogram(W_fit, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    wd = np.linspace(B.W_LO, B.W_HI, 400)
    scale = N * bin_w

    lin_null = (1.0 + base['m_null'] * (wd - B.W_MID)) / B._C_BG
    rad_c    = rad_pdf_W(wd, t_x0, SIGMA_W)
    null_tot = base['f_rad_null'] * rad_c + (1.0 - base['f_rad_null']) * lin_null
    excl_tot = (fs90 * B.signal_pdf(wd, SIGMA_W) + fr90 * rad_c
                + (1.0 - fs90 - fr90) * (1.0 + m90 * (wd - B.W_MID)) / B._C_BG)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"$\\Delta^+$(1232) search, final result  —  {label}, arm re-pairing selection\n"
        f"signal: BW $\\otimes$ Gaussian($\\sigma_W$ = {SIGMA_W*1e3:.0f} MeV);  "
        f"background: Bethe–Heitler radiative tail (t/X$_0$ = {t_x0:.3f}, shape fixed) + linear",
        fontsize=10.5)

    ax = axes[0]
    ax.errorbar(centres, counts, yerr=np.sqrt(np.maximum(counts, 1)), fmt="o",
                color="steelblue", ms=4, label=f"Data (N = {N:,})", zorder=4)
    ax.plot(wd, scale * (1.0 - base['f_rad_null']) * lin_null, color="grey", ls=":",
            lw=1.5, label="linear component")
    ax.plot(wd, scale * base['f_rad_null'] * rad_c, color="darkgoldenrod", ls="-.",
            lw=1.8, label=f"radiative tail (f = {base['f_rad_null']:.2f})")
    ax.plot(wd, scale * null_tot, color="black", lw=2, label="background total (null)")
    ax.plot(wd, scale * excl_tot, color="red", ls="--", lw=2,
            label=f"+ $\\Delta^+$ at 90% CL limit (excluded)")
    ax.axvline(B.mdelta, color="red", lw=1, ls=":", alpha=0.6)
    ax.text(B.mdelta + 0.008, ax.get_ylim()[1] * 0.02, "$m_{\\Delta}$",
            color="red", fontsize=9, alpha=0.8)
    ax.set_xlabel("W  [GeV/c$^2$]")
    ax.set_ylabel(f"Events / {bin_w*1e3:.0f} MeV")
    ax.set_title(f"Residual signal: 1.0$\\sigma$ (toys)  —  "
                 f"f_s < {fs90:.3f} at 90% CL", fontsize=10)
    ax.set_xlim(B.W_LO, B.W_HI)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.5, framealpha=0.9)

    ax2 = axes[1]
    fs_grid = np.linspace(0.0, min(0.99, 1.3 * fs90), 60)
    ax2.plot(fs_grid, [q_of(f) for f in fs_grid], color="black", lw=1.8,
             label="profile likelihood  q(f$_s$)")
    ax2.axhline(Q_ASYMPTOTIC, color="grey", ls=":", lw=1.2,
                label=f"asymptotic 90% ({Q_ASYMPTOTIC:.2f})")
    ax2.axvline(fs90_asym, color="grey", ls=":", lw=1.2)
    ax2.axvline(fs90, color="red", ls="--", lw=1.8,
                label=f"toy-calibrated limit: f$_s$ = {fs90:.3f}")
    ax2.axvline(fs_hat, color="steelblue", ls="-", lw=1.2, alpha=0.7,
                label=f"best fit: f$_s$ = {fs_hat:.3f}")
    ax2.set_xlabel("signal fraction  f$_s$")
    ax2.set_ylabel("q(f$_s$)")
    ax2.set_title("Upper-limit scan", fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    out = OUTDIR / "W_1mm_W_final.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
