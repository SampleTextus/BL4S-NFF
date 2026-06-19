#!/usr/bin/env python3
import sys
import numpy as np
import awkward as ak
import uproot
from pathlib import Path
from collections import namedtuple
import matplotlib.pyplot as plt

# Z Positions in cm
zmimosa = np.array([35.0, 38.6, 41.4, 48.5, 51.5, 53.65])
ztarget = 44.95 # estimated as midpoint of MIMOSA positions
zbrmc = 211.5
zbrmin = zbrmc - 75 # since length of BRM is 150 cm
zbrmout = zbrmc + 75
zdwc = 339.75
zmm = np.array([384.5, 515.0])
zbrmlen = 150.0

# Masses in GeV
mproton = 938.272*pow(10,-3)
melectron = 0.511*pow(10,-3)
mpion = 139.570*pow(10,-3)
mdelta = 1232.0*pow(10,-3)

# Useful constants
nrg_beam = 4.6 # GeV
pelectron = np.sqrt(nrg_beam**2 - melectron**2) #GeV/c
GeV_factor = 0.299792 # for dealing with magnetic field in GeV 

# Access data from file reco branch if valid and readable 
def open_tree(path):
    f = uproot.open(path)
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

# Checks if DWC data is present in the reco tree
def wdwc(reco):
    if "NDWC0_xPosition" not in reco.keys():
        return False
    return len(reco["NDWC0_xPosition"].array())>0

# Isolate valid runs. Check if mimosas were hit. Gives an array of booleans for each event
def upstream_valid(reco):
    nx0 = reco["NMimosa0_xPosition"].array(library="np")
    ny0 = reco["NMimosa0_yPosition"].array(library="np")
    nx1 = reco["NMimosa1_xPosition"].array(library="np")
    ny1 = reco["NMimosa1_yPosition"].array(library="np")
    nx2 = reco["NMimosa2_xPosition"].array(library="np")
    ny2 = reco["NMimosa2_yPosition"].array(library="np")
    return (nx0 == 1) & (ny0 ==1) & (nx1 == 1) & (ny1 == 1) & (nx2 == 1) & (ny2 == 1)

def downstream_valid(reco):
    nx3 = reco["NMimosa3_xPosition"].array(library="np")
    ny3 = reco["NMimosa3_yPosition"].array(library="np")
    nx4 = reco["NMimosa4_xPosition"].array(library="np")
    ny4 = reco["NMimosa4_yPosition"].array(library="np")
    nx5 = reco["NMimosa5_xPosition"].array(library="np")
    ny5 = reco["NMimosa5_yPosition"].array(library="np")
    return (nx3 == 1) & (ny3 == 1) & (nx4 == 1) & (ny4 == 1) & (nx5 == 1) & (ny5 == 1)

def pre_BRM_valid(reco): return upstream_valid(reco) & downstream_valid(reco)

#print(f"proportion of events passing upstream cut: {upstream_valid(open_tree(path)).sum()} / {len(upstream_valid(open_tree(path)))}") #% of events passing upstream cut
#print(f"proportion of events with valid downstream: {downstream_valid(open_tree(path)).sum()} / {len(downstream_valid(open_tree(path)))}") #% of events passing downstream cut

# Track construction: run an ols on the upstram detectors
# Since the positions are jagged, flatten first
def upstream_track(reco):
    z_up = zmimosa[:3]
    validated = pre_BRM_valid(reco)
    x_up = np.stack([ak.to_numpy(ak.flatten(reco["Mimosa0_xPosition"].array()[validated])),
                     ak.to_numpy(ak.flatten(reco["Mimosa1_xPosition"].array()[validated])),
                     ak.to_numpy(ak.flatten(reco["Mimosa2_xPosition"].array()[validated]))])
    y_up = np.stack([ak.to_numpy(ak.flatten(reco["Mimosa0_yPosition"].array()[validated])),
                     ak.to_numpy(ak.flatten(reco["Mimosa1_yPosition"].array()[validated])),
                     ak.to_numpy(ak.flatten(reco["Mimosa2_yPosition"].array()[validated]))])
    b1_upx, b0_upx = np.polyfit(z_up, x_up, 1)
    b1_upy, b0_upy = np.polyfit(z_up, y_up, 1)
    track_up = namedtuple("track_up", ["b1_upx", "b0_upx", "b1_upy", "b0_upy"])
    return track_up(b1_upx, b0_upx, b1_upy, b0_upy)

def downstream_track(reco):
    z_dn = zmimosa[3:]
    validated = pre_BRM_valid(reco)
    x_dn = np.stack([ak.to_numpy(ak.flatten(reco["Mimosa3_xPosition"].array()[validated])),
                     ak.to_numpy(ak.flatten(reco["Mimosa4_xPosition"].array()[validated])),
                     ak.to_numpy(ak.flatten(reco["Mimosa5_xPosition"].array()[validated]))])
    y_dn = np.stack([ak.to_numpy(ak.flatten(reco["Mimosa3_yPosition"].array()[validated])),
                     ak.to_numpy(ak.flatten(reco["Mimosa4_yPosition"].array()[validated])),
                     ak.to_numpy(ak.flatten(reco["Mimosa5_yPosition"].array()[validated]))])
    b1_dnx, b0_dnx = np.polyfit(z_dn, x_dn, 1)
    b1_dny, b0_dny = np.polyfit(z_dn, y_dn, 1)
    track_dn = namedtuple("track_dn", ["b1_dnx", "b0_dnx", "b1_dny", "b0_dny"])
    return track_dn(b1_dnx, b0_dnx, b1_dny, b0_dny)


# Extrapolation of tracks to target and BRM
def upstream_extrapolation(track_up):
    x_up = track_up.b1_upx * ztarget + track_up.b0_upx
    y_up = track_up.b1_upy * ztarget + track_up.b0_upy
    return x_up, y_up

def downstream_extrapolation(track_dn):
    x_dn = track_dn.b1_dnx * zbrmin + track_dn.b0_dnx
    y_dn = track_dn.b1_dny * zbrmin + track_dn.b0_dny
    return x_dn, y_dn

# Since we only need the partial derivatives, we use the slopes
def scatter_angle(track_up, track_dn):
    theta_x = np.arctan(track_dn.b1_dnx) - np.arctan(track_up.b1_upx)
    theta_y = np.arctan(track_dn.b1_dny) - np.arctan(track_up.b1_upy)
    return np.sqrt(theta_x**2 + theta_y**2)

def post_BRM_valid(reco):
    validated = pre_BRM_valid(reco)
    nmm1 = reco["NChamber1_position"].array(library="np")[validated]
    nmm2 = reco["NChamber2_position"].array(library="np")[validated]
    return (nmm1 == 1) & (nmm2 == 1)


# Magnetic deflection: Regress the downstream positions and compare to the upstream track.
#NB: We use the upstream since the downstream leads to the detected position on the MM
#NB: Chamber positions are in mm, not cm
#NB: Chamber data is jagged, so flatten first
def mag_def_angle(reco, track_up):
    pre_valid = pre_BRM_valid(reco)
    post_valid = post_BRM_valid(reco)
    pos1 = ak.to_numpy(ak.flatten(reco["Chamber1_position"].array()[pre_valid][post_valid])) * 0.1
    pos2 = ak.to_numpy(ak.flatten(reco["Chamber2_position"].array()[pre_valid][post_valid])) * 0.1
    detected_pos = np.stack([pos1, pos2])
    b1_detected, b0_detected = np.polyfit(zmm, detected_pos, 1)
    return np.arctan(b1_detected) - np.arctan(track_up.b1_upx[post_valid])

# Here we use the median deflection angle to get the estimation of the magnetic field 
def mag_field_calibration(reco, track_up):
    phi = mag_def_angle(reco, track_up)
    return np.sin(np.nanmedian(phi))*pelectron/GeV_factor
    
# Now we use the actual angle of deflection
def momentum_track (reco,track_up):
    phi = mag_def_angle(reco,track_up)
    return GeV_factor*mag_field_calibration(reco,track_up)/np.sin(phi)

# Recovering the mass of proton post collision
# Use Lorentz invariance in terms of masses and energies
# Look at momenta of concerned particles before and after
def inv_mass_rec(reco,track_up,track_dn):
    post_valid = post_BRM_valid(reco)
    pelectron_new = momentum_track(reco,track_up)
    theta = scatter_angle(track_up,track_dn)[post_valid]
    W = np.sqrt((mproton + np.sqrt(melectron**2 + pelectron**2)
                    - np.sqrt(melectron**2 + pelectron_new**2))**2
                    - (pelectron**2 + pelectron_new**2 - 2*pelectron*pelectron_new*np.cos(theta)))
    return W[np.isfinite(W)]

# Plotting histogram with events and recovered masses
def plot_inv_mass(reco,track_up,track_dn):
    W = inv_mass_rec(reco,track_up,track_dn)
    
    plt.figure()
    plt.hist(W, bins=50, range=(0.5, 2))
    plt.axvline(mdelta, color='r', linestyle='--', label=f'$\\Delta^+(1232)$ = {mdelta:.3f} GeV')
    plt.xlabel('W (GeV/c^2)')
    plt.ylabel('Events')
    plt.legend()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python bl4s_nff.py <path_to_root_file>")
        sys.exit(1)
    reco = open_tree(sys.argv[1])
    track_up = upstream_track(reco)
    track_dn = downstream_track(reco)
    plot_inv_mass(reco, track_up, track_dn)

if __name__ == "__main__":
    main()