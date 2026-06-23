#!/usr/bin/env python3
import sys
import numpy as np
import awkward as ak
from bl4s_nff import open_tree
from bl4s_nff import upstream_valid
from bl4s_nff import downstream_valid

def upstream_track(reco, up_valid):
    meanx_up = 0.1 * np.stack([ak.to_numpy(ak.mean(reco["Mimosa0_xPosition"].array()[up_valid], axis= 1)),
                     ak.to_numpy(ak.mean(reco["Mimosa1_xPosition"].array()[up_valid], axis= 1)),
                     ak.to_numpy(ak.mean(reco["Mimosa2_xPosition"].array()[up_valid], axis= 1))])
    meany_up = 0.1 * np.stack([ak.to_numpy(ak.mean(reco["Mimosa0_yPosition"].array()[up_valid], axis= 1)),
                     ak.to_numpy(ak.mean(reco["Mimosa1_yPosition"].array()[up_valid], axis= 1)),
                     ak.to_numpy(ak.mean(reco["Mimosa2_yPosition"].array()[up_valid], axis= 1))])
    return meanx_up, meany_up

def downstream_track(reco, dn_valid):
    meanx_dn = 0.1 * np.stack([ak.to_numpy(ak.mean(reco["Mimosa3_xPosition"].array()[dn_valid], axis= 1)),
                     ak.to_numpy(ak.mean(reco["Mimosa4_xPosition"].array()[dn_valid], axis= 1)),
                     ak.to_numpy(ak.mean(reco["Mimosa5_xPosition"].array()[dn_valid], axis= 1))])
    meany_dn = 0.1 * np.stack([ak.to_numpy(ak.mean(reco["Mimosa3_yPosition"].array()[dn_valid], axis= 1)),
                     ak.to_numpy(ak.mean(reco["Mimosa4_yPosition"].array()[dn_valid], axis= 1)),
                     ak.to_numpy(ak.mean(reco["Mimosa5_yPosition"].array()[dn_valid], axis= 1))])
    return meanx_dn, meany_dn


def compute_alignment(notarget_path):
    run = open_tree(notarget_path)
    meanx_up, meany_up = upstream_track(run, upstream_valid(run))
    meanx_dn, meany_dn = downstream_track(run, downstream_valid(run))
    align_x = np.concatenate([np.mean(meanx_up, axis=1), np.mean(meanx_dn, axis=1)])
    align_y = np.concatenate([np.mean(meany_up, axis=1), np.mean(meany_dn, axis=1)])
    return align_x, align_y

def main():
    align_x, align_y = compute_alignment("Runs/1601727557.root")
    print("Alignment offsets x (cm):", align_x)
    print("Alignment offsets y (cm):", align_y)

if __name__ == "__main__":
    main()