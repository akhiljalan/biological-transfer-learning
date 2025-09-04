import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import scipy
import os
from itertools import product
from datetime import datetime
import warnings

# Custom imports
from graph_estimation import graphon_families as gf
from graph_estimation import graphon_est_ell2 as ge 
from graph_estimation import utils
from graph_estimation.utils import truncate_sparse_matrix
from graph_estimation.graphon_experiments import get_boolean_sample
import graph_estimation.bionetworks_utils as bu 

def load_metabolic_networks(): 
    '''
    Load metabolic networks from sparse matrix format. 
    Returns a dictionary of dataframes, where each key corresponds 
    to a species. 
    '''
    path_to_saved = 'data/BiGG_data/metabolite_networks_processed/'
    fnames = os.listdir(path_to_saved)

    # Grab list of all unique species names. 
    uq_species = [x.split('_sparse_')[0] for x in fnames if 'mat' in x]
    df_dict = {}
    # Keys of dict are indexed by species names. 
    for k in uq_species:
        df_name = path_to_saved + k + '_sparse_mat.npz'
        txt_name = path_to_saved + k + '_metabolite_names.txt'
        comp_names = path_to_saved + k + '_compartment_names.txt'

        sparse_mat = scipy.sparse.load_npz(df_name)
        metabolite_names = [str(x) for x in np.loadtxt(txt_name, dtype=str)]
        compartment_names = [str(x) for x in np.loadtxt(comp_names, dtype=str)]
        df = bu.create_df(sparse_mat, metabolite_names, compartment_names)
        df_dict[k] = df
    
    # Grab a list of known node names, from a reference species. 
    # Note Recon3D is homo sapiens. 
    shared_nodes_all = set(list(df_dict['Recon3D'].index))
    for k in df_dict.keys(): 
        cur_nodes = set(list(df_dict[k].index))
        # print(len(shared_nodes_all))
        shared_nodes_all = shared_nodes_all.intersection(cur_nodes)
    

    shared_df_dict = {}
    shared_nodes = list(shared_nodes_all)
    for k in df_dict.keys(): 
        df_filtered, _ = bu.filter_df_by_metabolites(df_dict[k], shared_nodes)
        shared_df_dict[k] = df_filtered
        # print(df_filtered.shape)
    return shared_df_dict

def metabolic_network_estimator(source_df, target_df, nq_val): 
    '''
    Creates a submatrix of the target network adjacency matrix, with 
    nq_val nodes shown, and the rest masked. 

    Uses transfer learning to estimate the full target network adjacency matrix. 

    source_df: Dataframe for source network (e.g. species A)
    target_df: Dataframe for target network (e.g. species B)
    nq_val: Number of nodes in the submatrix that are seen. 

    Returns: Adjancency matrix of estimated target network. 
    '''
    np_val = len(source_df)
    nodes_subset = np.random.choice(np_val, size=nq_val, replace=False)


    # Grab the numeric entries of each dataframe, truncate to [0,1], and cast to numpy array. 
    source_mat_full = truncate_sparse_matrix(
        scipy.sparse.csr_matrix(source_df.iloc[:, :-1])).toarray()
    target_mat_full = truncate_sparse_matrix(
        scipy.sparse.csr_matrix(target_df.iloc[:, :-1])).toarray()

    # Generate {0,1}-valued sample of source network (Bernoulli sampling). 
    source_mat_sample = get_boolean_sample(source_mat_full)
    # Get row-wise L2 distances. 
    distance_estimates_source = ge.slice_distances_ell2_est(source_mat_sample)

    # Prepare Boolean {0,1} sample of target network. 
    target_matrix_subset = truncate_sparse_matrix(
        target_mat_full[:, nodes_subset][nodes_subset].tocsr()
    )
    target_matrix_subset_sample = get_boolean_sample(target_matrix_subset)
    distance_estimates_subset = distance_estimates_source[:, nodes_subset].copy()

    # Set mask matrix 
    quantile_cutoff = 50 * np.sqrt(np.log(nq_val) / nq_val)
    # Mask all entries in each row below the quantile cutoff 
    # (i.e. retain only the neighbors that are closest in L2 distance)
    mask = ge.bottom_percentile_mask(distance_estimates_subset, quantile_cutoff).astype(np.float32)

    # Construct (np x nq) scaling matrix 
    row_sums = np.sum(mask, axis=1)
    row_norms = np.divide(1.0, row_sums)
    mask_normalized = np.diag(row_norms) @ mask

    # Estimate target network adjacency matrix. 
    target_matrix_estimate = mask_normalized @ target_matrix_subset_sample @ mask_normalized.T

    return target_matrix_estimate

def main(): 
    metabolic_networks_df_dict = load_metabolic_networks()


if __name__ == "__main__": 
    main() 