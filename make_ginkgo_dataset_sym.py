import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt
import sys
import pickle
import argparse
import logging
import os

from ginkgo import invMass_symmetric_ginkgo as  invMass_ginkgo

Nsamples = 20000
minLeaves = 1
maxLeaves = 60
maxNTry = 200000

pt_min = torch.tensor(9)

### Physics inspired parameters to get ~ between 20 and 50 constituents
QCD_rate = 1.5

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

simulator.save(jet_list, "./data", "ginkgo_20000_jets_no_cuts_lambda_15_pt_min_9_jetp_400_with_perm_sym")