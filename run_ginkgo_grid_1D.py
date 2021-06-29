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


n_lambda = 150

#lambda_min = 1.6
#lambda_max = 2.75
lambda_min = 1.9
lambda_max = 3.05

pt_cut = 30.0

lambda_vals = np.linspace(lambda_min, lambda_max, n_lambda)

i = args.job_num

rate2=torch.tensor(8.)
pt_min = torch.tensor(float(pt_cut))

### Physics inspired parameters to get ~ between 20 and 50 constituents
QCD_rate = float(lambda_vals[i])

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

hist_savename = "ginkgo_hist_20000_jets_1D_jetp_400_lambda_{:n}_ptcut_{:n}_{}".format(
    int(lambda_vals[i])*1000,
    int(pt_cut),
    i)

savename = "ginkgo_20000_jets_1D_jetp_400_lambda_{:n}_ptcut_{:n}_{}".format(
    int(lambda_vals[i])*1000,
    int(pt_cut),
    i)

np.save(os.path.join("/scratch/mdd424/data/ginkgo", hist_savename), leaf_dist)

simulator.save(jet_list, "/scratch/mdd424/data/ginkgo", savename)
