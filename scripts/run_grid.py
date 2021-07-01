import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt
import sys
import pickle
import argparse
import logging
import os

from ginkgo import invMass_ginkgo

parser = argparse.ArgumentParser()
parser.add_argument("job_num", help="The number of the job that's running", type=int)
args = parser.parse_args()

Nsamples = 20000
minLeaves = 1
maxLeaves = 150
maxNTry = 50000

n_cuts = 40
n_lambda = 40

#cut_min = 4
#cut_max = 90
#lambda_min = 1e-1
#lambda_max = 5

#cut_min = 26
#cut_max = 43
cut_min = 33
cut_max = 38
lambda_min = 1.35
lambda_max = 2.4

cut_vals = np.linspace(cut_min, cut_max, n_cuts)
lambda_vals = np.linspace(lambda_min, lambda_max, n_lambda)

grid_cut, grid_lambda = np.meshgrid(cut_vals, lambda_vals)

j, i = divmod(args.job_num, 40)

rate2=torch.tensor(8.)
pt_min = torch.tensor(float(grid_cut[j,i]))

### Physics inspired parameters to get ~ between 20 and 50 constituents
QCD_rate = float(grid_lambda[j,i])

QCD_mass = 30.

rate=torch.tensor([QCD_rate,QCD_rate]) #Entries: [root node, every other node] decaying rates. Choose same values for a QCD jet
M2start = torch.tensor(QCD_mass**2)

jetM = np.sqrt(M2start.numpy())

jetdir = np.array([1,1,1])
jetP = 400.
jetvec = jetP * jetdir / np.linalg.norm(jetdir)

jet4vec = np.concatenate(([np.sqrt(jetP**2 + jetM**2)], jetvec))


simulator = invMass_ginkgo.Simulator(jet_p=jet4vec,
                                     pt_cut=float(pt_min),
                                     Delta_0=M2start,
                                     M_hard=jetM ,
                                     num_samples=Nsamples,
                                     minLeaves =minLeaves,
                                     maxLeaves = maxLeaves,
                                     maxNTry = maxNTry)

jet_list = simulator(rate)

num_leaves = [len(x["leaves"]) for x in jet_list]
leaf_dist = np.histogram(num_leaves, bins=np.arange(1,120), density=True)[0]

hist_savename = "ginkgo_hist_20000_jets_jetp_400_lambda_{:n}_ptcut_{:n}_{}_{}".format(
    int(grid_lambda[j,i]*1000),
    int(grid_cut[j,i]),
    j,
    i)

savename = "ginkgo_20000_jets_jetp_400_lambda_{:n}_ptcut_{:n}_{}_{}".format(
    int(grid_lambda[j,i]*1000),
    int(grid_cut[j,i]),
    j,
    i)

np.save(os.path.join("/scratch/mdd424/data/ginkgo", hist_savename), leaf_dist)

simulator.save(jet_list, "/scratch/mdd424/data/ginkgo", savename)
