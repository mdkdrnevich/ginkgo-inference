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

Nsamples = 8000
minLeaves = 1
maxLeaves = 60
maxNTry = 20000

n_cuts = 40
n_lambda = 40

cut_vals = np.linspace(4, 90, n_cuts)
lambda_vals = np.linspace(1e-1, 5, n_lambda)

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

simulator.save(jet_list, "/scratch/mdd424/data/ginkgo", "ginkgo_8000_jets_jetp_400_lambda_{:n}_ptcut_{:n}_{}_{}".format(
    int(grid_lambda[j,i])*1000,
    int(grid_cut[j,i]),
    j,
    i))
