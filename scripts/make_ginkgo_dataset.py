import numpy as np
import torch
import argparse
import logging

from ginkgo import invMass_ginkgo


def main():
    parser = argparse.ArgumentParser(description='Generate Ginkgo jet dataset')
    parser.add_argument('--nsamples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--min-leaves', type=int, default=1, help='Minimum number of leaves')
    parser.add_argument('--max-leaves', type=int, default=60, help='Maximum number of leaves')
    parser.add_argument('--max-ntry', type=int, default=200000, help='Maximum number of tries')
    parser.add_argument('--pt-min', type=float, default=30, help='Minimum pT cut')
    parser.add_argument('--qcd-rate', type=float, default=2.4, help='QCD decay rate')
    parser.add_argument('--qcd-mass', type=float, default=30., help='QCD mass')
    parser.add_argument('--jet-p', type=float, default=400., help='Jet momentum')
    parser.add_argument('--output-dir', type=str, default='../data', help='Output directory')
    parser.add_argument('--output-name', type=str, default=None, help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    Nsamples = args.nsamples
    minLeaves = args.min_leaves
    maxLeaves = args.max_leaves
    maxNTry = args.max_ntry
    pt_min = torch.tensor(args.pt_min)
    QCD_rate = args.qcd_rate
    QCD_mass = args.qcd_mass
    jetP = args.jet_p
    
    rate = torch.tensor([QCD_rate, QCD_rate])
    M2start = torch.tensor(QCD_mass**2)
    jetM = np.sqrt(M2start.numpy())
    
    jetdir = np.array([1, 1, 1])
    jetvec = jetP * jetdir / np.linalg.norm(jetdir)
    jet4vec = np.concatenate(([np.sqrt(jetP**2 + jetM**2)], jetvec))
    
    simulator = invMass_ginkgo.Simulator(jet_p=jet4vec,
                                         pt_cut=float(pt_min),
                                         Delta_0=M2start,
                                         M_hard=jetM,
                                         num_samples=Nsamples,
                                         minLeaves=minLeaves,
                                         maxLeaves=maxLeaves,
                                         maxNTry=maxNTry)
    
    jet_list = simulator(rate)
    
    if args.output_name is None:
        output_name = "ginkgo_{}_jets_no_cuts_lambda_{}_pt_min_{}_jetp_{}".format(
            Nsamples, int(QCD_rate * 10), int(pt_min), int(jetP))
    else:
        output_name = args.output_name
    
    simulator.save(jet_list, args.output_dir, output_name)


if __name__ == '__main__':
    main()
