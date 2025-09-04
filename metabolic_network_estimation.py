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

def iter_exp_metabolic_all_methods(P_df, Q_df, n_vals, 
                                   p_flip_vals,
                                np_fixed, 
                                num_trials, seed=11):
    np.random.seed(seed)
    df_col_names = ['np', 'nq', 'our_algo', 'spectral_algo',
       'bitflip_algo', 'p_flip']
    results = pd.DataFrame(
        columns = df_col_names
    )
    P_full = scipy.sparse.csr_matrix(P_df.iloc[:, :-1])
    Q_full = scipy.sparse.csr_matrix(Q_df.iloc[:, :-1])
    
    Q_full_np = truncate_sparse_matrix(Q_full.copy()).toarray()
    P_full_np = truncate_sparse_matrix(P_full.copy()).toarray()

    P_full_sample = get_boolean_sample(truncate_sparse_matrix(
        P_full)
    ).toarray()
    d_est_p = ge.slice_distances_ell2_est(P_full_sample)
    
    for nq in n_vals: 
        h_quantile = 50 * np.sqrt(np.log(nq) / nq)
        n_p = len(P_df)
        n_q = nq
        for _ in range(num_trials): 
            subset_context = np.random.choice(n_p, size=n_q, replace=False)
            subset_indices = subset_context
            Q_subset = truncate_sparse_matrix(
                Q_full[:, subset_context][subset_context].tocsr()
            )
            Q_subset_sample = get_boolean_sample(Q_subset).toarray()
            
            d_est_subset = d_est_p[:, subset_indices].copy()
            # print(d_est_subset.shape)
            mask = ge.bottom_percentile_mask(d_est_subset, 
                                            h_quantile).astype(np.float32)
            row_sums = np.sum(mask, axis=1)
            row_norms = np.divide(1.0, row_sums)
            mask_normalized = np.diag(row_norms) @ mask
            # print(mask_normalized.shape)
            # print(Q_subset_sample.shape)
            
            q_hat_ours = mask_normalized @ Q_subset_sample @ mask_normalized.T

            qhat_ours_err = ge.frob_error(q_hat_ours, Q_full_np)

            q_hat_eigvec = ge.eigenvec_feature_transfer_baseline(P_full_sample, Q_subset_sample, 
                                                                 subset_indices)
            qhat_eigvec_err = ge.frob_error(q_hat_eigvec, Q_full_np)
            

            for p_flip in p_flip_vals: 
                qhat_bitflip, _, _, _ = ge.get_usvt_bitflipped_estimator(Q_full_np, 
                                                                        subset_indices, p_flip = p_flip)
                qhat_bitflip_err = ge.frob_error(qhat_bitflip, Q_full_np)
                # df_col_names = ['alpha', 'beta', 'np', 'nq', 'our_algo', 'spectral_algo',
                #     'bitflip_algo', 'p_flip']
                results.loc[len(results)] = [
                    np_fixed, nq, qhat_ours_err, qhat_eigvec_err, 
                    qhat_bitflip_err, p_flip
                ]
        print(f'{nq=}, {p_flip=}')
    return results 

def main(): 
    metabolic_networks_df_dict = load_metabolic_networks()
    

if __name__ == "__main__": 
    main() 