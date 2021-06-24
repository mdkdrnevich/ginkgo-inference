import sys
sys.path.append("./ClusterTrellis/src")

import os
import pickle
import string
import time
import logging
import numpy as np
import importlib
import argparse

from ClusterTrellis import run_experiments as rpe
from ClusterTrellis import HierarchicalTrellis
from ClusterTrellis.trellis_node import TrellisNode

from ClusterTrellis.utils import get_logger
logger = get_logger(level=logging.WARNING)

"""Replace with model auxiliary scripts to calculate the energy function"""
#from ClusterTrellis import Ginkgo_likelihood as likelihood
from ginkgo import likelihood_invM as likelihood

class ModelNode(TrellisNode):
    """Class to define the node pairwise splitting energy and calculate node features for a given model """

    def __init__(self,
                 model_params,
                 elements = None,
                 children = None,
                 map_features = None):
        TrellisNode.__init__(self, model_params, elements, children, map_features)


    def get_energy_of_split(self, a_node, b_node):
        """Model energy function.
        Args: sibling nodes
        returns: model pairwise splitting energy
        """

        logger.debug(f"computing energy of split: {a_node, b_node}")

        split_llh = likelihood.split_logLH_with_stop_nonstop_prob(a_node.map_features,
                                           b_node.map_features,
                                           self.model_params["delta_min"],
                                           self.model_params["lam"])
        logger.debug(f"split_llh = {split_llh}")

        return split_llh


    def compute_map_features(self, a_node, b_node):
        """Auxiliary method to get the parent vertex features. This is model dependent. In Ginkgo these are the momentum and parent         invariant mass.
        Args: sibling nodes
        returns: list where each entry is a parent feature, e.g. [feature1,feature2,...]
        """
        momentum = a_node.map_features + b_node.map_features
        logger.debug(f"computing momentum for {a_node, b_node, momentum}")

        
#         logger.debug(f"computing  parent invariant mass {a_node, a_node.map_features, b_node, b_node.map_features}")
#         pP = a_node.map_features + b_node.map_features

#         """Parent invariant mass squared"""
#         tp1 = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2
#         logger.debug(f"tp =  {tp1}")

        return momentum
    
    
def runTrellisOnly(gt_trees,
                   model_params,
                   NleavesMin =3, 
                   NleavesMax= 4, 
                   MaxNjets = 1):
    
    """Create and fill the trellis for a given set of leaves
    Args: 
    gt_trees : truth level  trees.
    NleavesMin: minimum number of leaves to select  trees
    NleavesMax: maximum number of leaves to select  trees
    MaxNjets: maximum number of trees to run the trellis code.
    
    Returns:
    results: dictionary with the following {key:values}
                Z: partition functions,
                trellis_MLE: MAP hierarchy, 
                RunTime: time for each tree
                totJets": total number of trees
                gt_llh: truth level trees llh
 
    """
    
    """ Keep only trees with leaves between NleavesMin and NleavesMax"""
    smallJetIndex =[i for i,gt_tree in enumerate(gt_trees)  if NleavesMin<=len(gt_tree["leaves"])<NleavesMax]
            
    
    results = {"Z":[], "trellis_MLE":[], "RunTime":[], "Ntrees":[]}
    
    if len(smallJetIndex)>0:
        
        GT_trees = np.asarray(gt_trees)[smallJetIndex]
        
        """ Total number of trees to run the trellis"""
        totTrees = np.min([MaxNjets,len(smallJetIndex)])

        results["totTrees"] = totTrees
        results["Nleaves"] = [len(GT_trees[m]["leaves"]) for m in range(totTrees)]
        results["gt_llh"]= [np.sum(GT_trees[m]["logLH"]) for m in range(totTrees)]

        for m in range(totTrees):

            #if m%50==0:
            #    print("Creating trellis for jet #",m)
                
            startTime = time.time()
            
            data_params = GT_trees[m]
            N=len(data_params['leaves'])
            
            """Replace with current model parameters"""
            leaves_features =[ data_params['leaves'][i] for i in range(N)]
            #model_params ={}
            #model_params["delta_min"] = float(data_params['pt_cut'])
            #model_params["lam"]= float(data_params['Lambda'])

            """ Create and fill the trellis"""
            trellis, Z, map_energy, Ntrees,  totTime = rpe.compare_map_gt_and_bs_trees(GT_trees[m],
                                                                                       ModelNode,
                                                                                       leaves_features,
                                                                                       model_params)

            results["Z"].append(Z)
            results["trellis_MLE"].append(map_energy)
            results["Ntrees"].append(Ntrees)

            endTime = time.time() - startTime
            results["RunTime"].append(endTime)

    else:
        print("There are no jets in the dataset with the required number of leaves ")

    return results, trellis

    
def load_jets(filename):
    """Load truth binary trees"""
    root_dir = "/home/mdd424/research/trellis_inference/data/"
    filename = os.path.join(root_dir, filename)
    with open(filename + ".pkl", "rb") as fd:
        Truth= pickle.load(fd, encoding='latin-1')
    return Truth


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_num", help="The minibatch number which is used as the random seed", type=int)
    parser.add_argument("job_num", help="The number of the job that's running", type=int)
    args = parser.parse_args()
    
    gt_trees = load_jets("ginkgo_50000_jets_no_cuts_lambda_21_pt_min_36_jetp_400")
    
    NleavesMin =1
    NleavesMax=100
    MaxNjets = 1000

    n_cuts = 40
    n_lambda = 40

    #cut_vals = np.linspace(4, 90, n_cuts)
    #lambda_vals = np.linspace(1e-1, 5, n_lambda)
    cut_vals = np.linspace(26, 43, n_cuts)
    lambda_vals = np.linspace(1.35, 2.4, n_lambda)
    grid_cut, grid_lambda = np.meshgrid(cut_vals, lambda_vals)
    
    seed = args.batch_num
    np.random.seed(seed)
    
    j, i = divmod(args.job_num, 40)

    model_params = {"delta_min": grid_cut[j,i], "lam": grid_lambda[j,i]}

    results, _ =  runTrellisOnly(np.random.choice(gt_trees, 1000, replace=False), 
                                 model_params,
                                 NleavesMin =NleavesMin, 
                                 NleavesMax= NleavesMax, 
                                 MaxNjets = MaxNjets)
    results["delta_min"] = grid_cut[j,i]
    results["lam"] = grid_lambda[j,i]
    results["coords"] = (j,i)
    
    outdir = "/scratch/mdd424/data/trellis/batch" + str(seed)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    out_filename = os.path.join(outdir, "trellis_1000_jets_lambda_{:n}_ptcut_{:n}_{}_{}.pkl".format(
        int(grid_lambda[j,i])*1000,
        int(grid_cut[j,i]),
        j,
        i))
    with open(out_filename, "wb") as f:
        pickle.dump(results, f, protocol=2)
        