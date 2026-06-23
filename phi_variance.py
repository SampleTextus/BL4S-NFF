#!/usr/bin/env python3
"""
Break down the observed phi variance into its contributing components.
If sum ≈ 8 mrad: hardware limited.
If sum ≈ 2 mrad: reconstruction bug exists.
"""
import numpy as np
from bl4s_nff import (open_tree, upstream_valid, downstream_valid,
                       post_BRM_valid, upstream_track, downstream_track,
                       mag_def_angle, zmimosa, zmm, melectron, GeV_factor,
                       BL_CALIBRATED, NOTARGET_PATH)
from mimosa_residuals import compute_alignment

# ── Physical constants ────────────────────────────────────────────────────────
E_BEAM   = 4.0   # GeV
P_BEAM   = np.sqrt(E_BEAM**2 - melectron**2)
M_E_MEV  = 0.511 # MeV

# ── 1. MM detector contribution ───────────────────────────────────────────────
# Two chambers at z1=384.5 cm, z2=515.0 cm → separation L = 130.5 cm
# Chamber position resolution: MMFE8 strips at 0.45 mm pitch → σ_x ≈ pitch/sqrt(12)
# but conservative estimate uses full pitch as uncertainty
mm_pitch_mm  = 0.45       # mm
sigma_x_mm   = mm_pitch_mm / np.sqrt(12)   # mm, single-strip resolution
L_mm_cm      = zmm[1] - zmm[0]             # cm = 130.5 cm
sigma_slope_MM = np.sqrt(2) * (sigma_x_mm * 0.1) / L_mm_cm   # cm/cm = rad
print(f"MM contribution (pitch={mm_pitch_mm} mm, L={L_mm_cm:.1f} cm):")
print(f"  sigma_x_MM     = {sigma_x_mm:.3f} mm")
print(f"  sigma_slope_MM = {sigma_slope_MM*1e3:.3f} mrad")

# ── 2. Upstream telescope contribution ────────────────────────────────────────
# 3 planes at z = 35.0, 38.6, 41.4 cm — MIMOSA resolution ~30 µm
sigma_x_mimosa_cm = 30e-4   # 30 µm in cm
z_up = zmimosa[:3]
z_bar_up = z_up.mean()
S_zz_up  = np.sum((z_up - z_bar_up)**2)
sigma_slope_up = sigma_x_mimosa_cm / np.sqrt(S_zz_up)
print(f"\nUpstream MIMOSA (3 planes, sigma_x={sigma_x_mimosa_cm*1e4:.0f} µm):")
print(f"  z span         = {z_up[-1]-z_up[0]:.1f} cm")
print(f"  sigma_slope_up = {sigma_slope_up*1e3:.3f} mrad")

# ── 3. Downstream telescope contribution ──────────────────────────────────────
z_dn = zmimosa[3:]
z_bar_dn = z_dn.mean()
S_zz_dn  = np.sum((z_dn - z_bar_dn)**2)
sigma_slope_dn = sigma_x_mimosa_cm / np.sqrt(S_zz_dn)
print(f"\nDownstream MIMOSA (3 planes, sigma_x={sigma_x_mimosa_cm*1e4:.0f} µm):")
print(f"  z span         = {z_dn[-1]-z_dn[0]:.1f} cm")
print(f"  sigma_slope_dn = {sigma_slope_dn*1e3:.3f} mrad")

# ── 4. Multiple scattering (Highland formula) ─────────────────────────────────
# θ₀ = (13.6 MeV / pβ) × √(x/X₀) × [1 + 0.038 ln(x/X₀)]
def highland(x_over_X0, p_MeV, beta=1.0):
    log_term = 1 + 0.038 * np.log(x_over_X0)
    return (13.6 / (p_MeV * beta)) * np.sqrt(x_over_X0) * log_term

p_MeV = P_BEAM * 1e3

# MIMOSA26 planes: ~50 µm Si each, X₀(Si) = 93.7 mm
X0_Si_mm = 93.7
t_mimosa_mm = 0.05   # 50 µm per plane
x_X0_mimosa = t_mimosa_mm / X0_Si_mm
theta_MS_per_mimosa = highland(x_X0_mimosa, p_MeV)
theta_MS_6mimosa    = np.sqrt(6) * theta_MS_per_mimosa   # 6 planes in quadrature

# 1mm Tungsten: X₀(W) = 3.504 mm
theta_MS_W1mm = highland(1.0/3.504, p_MeV)

# 2cm Water: X₀(H₂O) = 360.8 mm
theta_MS_H2O = highland(20.0/360.8, p_MeV)

print(f"\nMultiple scattering (Highland formula, p = {p_MeV:.0f} MeV):")
print(f"  MIMOSA planes (6 × 50µm Si): {theta_MS_6mimosa*1e3:.3f} mrad")
print(f"  1mm Tungsten               : {theta_MS_W1mm*1e3:.3f} mrad")
print(f"  2cm Water                  : {theta_MS_H2O*1e3:.3f} mrad")

# ── 5. Total predicted sigma_phi (current definition: MM - upstream) ──────────
# phi = slope_MM - slope_upstream
# Contributions: MM noise + upstream noise + MS in target (scattering contaminates phi)
print(f"\n{'='*55}")
print("Predicted sigma_phi (phi = MM slope - UPSTREAM slope):")
print(f"  [contaminates phi with target scattering angle]")
for label, theta_MS_tgt in [("No target", 0), ("1mm W", theta_MS_W1mm), ("2cm H2O", theta_MS_H2O)]:
    sigma_total = np.sqrt(sigma_slope_MM**2 + sigma_slope_up**2
                          + theta_MS_6mimosa**2 + theta_MS_tgt**2)
    print(f"  {label:<12}: {sigma_total*1e3:.2f} mrad")

# ── 6. Corrected definition: phi = MM slope - DOWNSTREAM slope ────────────────
print(f"\nPredicted sigma_phi (phi = MM slope - DOWNSTREAM slope):")
print(f"  [cleanly isolates magnetic deflection]")
for label, theta_MS_tgt in [("No target", 0), ("1mm W", theta_MS_W1mm), ("2cm H2O", theta_MS_H2O)]:
    sigma_total = np.sqrt(sigma_slope_MM**2 + sigma_slope_dn**2
                          + theta_MS_6mimosa**2)
    # target MS cancels — downstream arm already saw it, MM arm also saw it
    print(f"  {label:<12}: {sigma_total*1e3:.2f} mrad")

# ── 7. Measured phi_std from data ─────────────────────────────────────────────
print(f"\n{'='*55}")
print("Measured phi_std from data:")
align_x, align_y = compute_alignment(NOTARGET_PATH)
for path, label in [
    ("Runs/No Target Runs/1601727557.root", "No target"),
    ("Runs/Target Runs/1601730086.root",    "1mm W"),
    ("Runs/Target Runs/1601747831.root",    "2cm H2O"),
]:
    reco     = open_tree(path)
    up_v     = upstream_valid(reco)
    post_v   = post_BRM_valid(reco, up_v)
    trk_up   = upstream_track(reco, up_v, align_x, align_y)
    phi      = mag_def_angle(reco, trk_up, up_v, post_v)
    print(f"  {label:<12}: median={np.nanmedian(phi)*1e3:.2f} mrad, "
          f"std={np.nanstd(phi)*1e3:.2f} mrad  (N={post_v.sum():,})")
