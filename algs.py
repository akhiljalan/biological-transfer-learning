import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy

def assign_labels(sorted_sequence, k):
    # Calculate the breakpoints to divide [0, 1] into k equal intervals
    breakpoints = np.linspace(0, 1, k+1)
    
    # Use digitize to determine which interval each element in sorted_sequence belongs to
    labels = np.digitize(sorted_sequence, breakpoints[1:])  # Exclude the first endpoint
    
    return labels

def block_average_matrix(input_matrix, labels):
    N = input_matrix.shape[0]

    # Determine the number of label groups
    # note that the labels are 0, 1, ..., k-1
    k = np.max(labels) + 1 

    # Initialize the block-averaged matrix of size (k x k)
    block_avg_matrix = np.zeros((k, k))

    # Extend the block-averaged matrix to N x N shape based on labels
    extended_matrix = np.zeros((N, N))
    
    # Iterate over each block (label pair) and compute the average
    for i in range(0, k):  # Labels are 1-based
        for j in range(0, k):
            # Find indices corresponding to rows with label i and columns with label j
            rows_indices = np.where(labels == i)[0]
            cols_indices = np.where(labels == j)[0]
            
            # Extract the submatrix based on label group indices
            submatrix = input_matrix[rows_indices[:, None], cols_indices]
            
            # Compute the average of the submatrix
            if len(rows_indices) > 0 and len(cols_indices) > 0:
                block_avg_matrix[i,j] = np.mean(submatrix)
                extended_matrix[rows_indices[:, None], cols_indices] = block_avg_matrix[i,j]
    
    return extended_matrix, block_avg_matrix

def frob_error(q, q_hat): 
    n_q = q.shape[0]
    q_no_diag = q - np.diag(np.diag(q))
    q_hat_no_diag = q_hat - np.diag(np.diag(q_hat))
    error = np.linalg.norm(q_no_diag - q_hat_no_diag, ord='fro') ** 2
    error_normalized = error / (n_q ** 2)
    return error_normalized

def calculate_mean_stdev(dictionary):
    """
    Calculate the mean and 2 times the standard deviation for each list of floats
    associated with keys in a given dictionary.

    Parameters:
    dictionary (dict): A dictionary where keys are numbers and values are lists of floats.

    Returns:
    tuple: Two lists containing means and 2 times the standard deviations for each list of floats.
    """
    means = []
    stdevs = []
    k_vals_sorted = sorted(dictionary.keys())

    # Iterate through each key-value pair in the dictionary
    for key in k_vals_sorted: 
        # Calculate mean and standard deviation of the list of floats
        value = dictionary[key]
        mean_value = np.mean(value)
        std_value = np.std(value)

        # Append mean and 2 times the standard deviation to the respective lists
        means.append(mean_value)
        stdevs.append(2 * std_value)

    return k_vals_sorted, means, stdevs

def plot_with_errorbars(errors_dict, np_multiple_str, 
                        num_trials,
                        title_descr,
                        savepath=None): 
    keys, means, stdevs = calculate_mean_stdev(errors_dict)
    plt.errorbar(keys, means, yerr=stdevs, fmt='o', capsize=10, markersize=20)
    # plt.xlabel('Keys')
    # plt.ylabel('Means')
    plt.title('Means with Error Bars (2 * Stdev)')

    plt.title(f'Rankings Algorithm, {num_trials} Trials, {title_descr}')

    plt.xticks(keys, keys);
    plt.xlabel(f'n_Q, with n_P = {np_multiple_str}')
    plt.ylabel('Frobenius Error on Full Q')
    plt.tight_layout()
    # plt.title('Error of Rankings Algorithm on SBM')
    # plt.legend()
    if savepath is not None: 
        plt.savefig(savepath, dpi=700.0)

####################
## NBHD SMOOTHING

def slice_distances_ell2_est(adj_mat): 
    adj = adj_mat.astype(np.float32)
    prod = adj.T @ adj
    n = adj.shape[0]
    dist_estimates = np.zeros((n, n))
    for i in range(n): 
        for j in range(i): 
            diff = np.linalg.norm(prod[i, :] - prod[j, :])
            correction = prod[i, i] - prod[j, i] 
            correction2 = prod[j, j] - prod[j, i]
            est = diff - correction - correction2 
            dist_estimates[i, j] = est 
    out = dist_estimates + dist_estimates.T
    return out


def bottom_percentile_mask(matrix, h):
    """
    Compute a binary mask matrix where each element is 1 if it belongs to the lowest h percentiles of its row.

    Args:
    - matrix (numpy array): Input matrix of shape (n, n) containing float values.
    - h (float): Percentage threshold (between 0 and 100) to determine lowest percentiles.

    Returns:
    - mask (numpy array): Binary mask matrix of shape (n, n) where lowest h percentiles are 1 and the rest are 0.
    """
    # Compute the percentile threshold for each row
    percentile_threshold = np.percentile(matrix, h, axis=1, interpolation='midpoint')

    # Create a binary mask based on the percentile threshold
    mask = (matrix <= percentile_threshold[:, np.newaxis]).astype(np.uint8)

    return mask


def get_nbhd_smoothed(adj_matrix, d_est, h_quantile): 
    mask = bottom_percentile_mask(d_est, h_quantile).astype(np.float32)
    row_sums = np.sum(mask, axis=1)
    row_norms = np.divide(1.0, row_sums)
    mask_normalized = np.diag(row_norms) @ mask
    p_hat = mask_normalized @ adj_matrix
    p_upper = np.triu(p_hat, k=1)
    final_est = p_upper + p_upper.T.copy()
    return final_est

def get_nbhd_smooth_estimator_simple(adj_matrix, h_quantile): 
    d_est = slice_distances_ell2_est(adj_matrix)
    return get_nbhd_smoothed(adj_matrix, d_est, h_quantile)

def matrix_completion_nbhd_smoothed(adj_matrix_q, 
                                    d_est_p, 
                                    h_quantile, 
                                    index_subset): 
    # use the D estimate, but select only the cols corresp to n_Q 
    d_est_subset = d_est_p[:, index_subset].copy()
    mask = bottom_percentile_mask(d_est_subset, h_quantile).astype(np.float32)
    row_sums = np.sum(mask, axis=1)
    row_norms = np.divide(1.0, row_sums)
    mask_normalized = np.diag(row_norms) @ mask
    q_hat_completed = mask_normalized @ adj_matrix_q @ mask_normalized.T
    q_upper = np.triu(q_hat_completed, k=1)
    final_est = q_upper + q_upper.T.copy()
    return final_est

def matrix_completion_from_pair(gp):
    # gp is GraphonPair type 
     
    q_sample_mat = gp.Q.g1_sample.toarray()
    q_truth_mat = gp.Q.g1

    d_est_large = slice_distances_ell2_est(
        gp.P.g1_sample.toarray())

    subset_indices = gp.subset_indices
    d_est_subset = d_est_large[:, subset_indices].copy()
    nq_nodes = gp.nq
    h_quantile = 100 * np.sqrt(np.log(nq_nodes) / nq_nodes)


    mask = bottom_percentile_mask(d_est_subset, 
                                          h_quantile).astype(np.float32)
    row_sums = np.sum(mask, axis=1)
    row_norms = np.divide(1.0, row_sums)
    mask_normalized = np.diag(row_norms) @ mask
    q_hat_completed = mask_normalized @ q_sample_mat @ mask_normalized.T
    return q_hat_completed


def matrix_completion_from_matrices(p_sample, q_sample, 
                                    subset_indices, d_est_large=None, 
                                    h_quantile=None): 
    
    if d_est_large is None: 
        d_est_large = slice_distances_ell2_est(
            p_sample)

    # subset_indices = gp.subset_indices
    d_est_subset = d_est_large[:, subset_indices].copy()
    nq_nodes = q_sample.shape[0]
    if h_quantile is None: 
        h_quantile = 100 * np.sqrt(np.log(nq_nodes) / nq_nodes)


    mask = bottom_percentile_mask(d_est_subset, 
                                          h_quantile).astype(np.float32)
    row_sums = np.sum(mask, axis=1)
    row_norms = np.divide(1.0, row_sums)
    mask_normalized = np.diag(row_norms) @ mask
    q_hat_completed = mask_normalized @ q_sample @ mask_normalized.T
    return q_hat_completed

def calculate_mean_stdev(dictionary):
    """
    Calculate the mean and 2 times the standard deviation for each list of floats
    associated with keys in a given dictionary.

    Parameters:
    dictionary (dict): A dictionary where keys are numbers and values are lists of floats.

    Returns:
    tuple: Two lists containing means and 2 times the standard deviations for each list of floats.
    """
    means = []
    stdevs = []
    k_vals_sorted = sorted(dictionary.keys())

    # Iterate through each key-value pair in the dictionary
    for key in k_vals_sorted: 
        # Calculate mean and standard deviation of the list of floats
        value = dictionary[key]
        mean_value = np.mean(value)
        std_value = np.std(value)

        # Append mean and 2 times the standard deviation to the respective lists
        means.append(mean_value)
        stdevs.append(2 * std_value)

    return k_vals_sorted, means, stdevs


#####################################
##### BASELINES 
#####################################

from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def spectral_clustering(adj_matrix, k, seed=42):
    """
    Perform k-cluster spectral clustering on an adjacency matrix and construct the clustering matrix.

    Args:
        adj_matrix (numpy.ndarray): The n x n adjacency matrix of the graph.
        k (int): The number of clusters.

    Returns:
        clustering_matrix (numpy.ndarray): The n x k clustering matrix.
    """
    L = laplacian(adj_matrix.astype(np.float32), normed=True)
    _, eigvecs = eigh(L, subset_by_index=[0, k-1])
    kmeans = KMeans(n_clusters=k, random_state=seed)
    labels = kmeans.fit_predict(eigvecs)
    n = adj_matrix.shape[0]
    clustering_matrix = np.zeros((n, k), dtype=int)
    for i in range(n):
        clustering_matrix[i, labels[i]] = 1
    return clustering_matrix

def community_fraction_matrix(clustering_matrix):
    """
    Constructs a diagonal matrix with the fraction of nodes in each community.

    Args:
        clustering_matrix (numpy.ndarray): The n x k clustering matrix.

    Returns:
        numpy.ndarray: The k x k diagonal matrix with fractions on the diagonals.
    """
    n, k = clustering_matrix.shape
    
    # Calculate the fraction of nodes in each community
    node_counts = np.sum(clustering_matrix, axis=0).astype(np.float64)  # Sum over rows to get the number of nodes in each community
    # fractions = node_counts / n

    node_inv = np.divide(np.ones_like(node_counts), node_counts, 
                         out=np.zeros_like(node_counts), where=node_counts!=0)
    
    # Construct the diagonal matrix
    fraction_matrix = np.diag(node_inv)
    
    return fraction_matrix


def sbm_cluster_transfer(A_P, A_Q, xi_subset, k_P=None, k_Q=None, seed=42):
    """
    Implements the given pseudocode.

    Args:
        A_P (numpy.ndarray): The n_P x n_P adjacency matrix of the graph P.
        A_Q (numpy.ndarray): The n_Q x n_Q adjacency matrix of the graph Q.
        S (list): The subset of [n_P] such that |S| = n_Q.

    Returns:
        numpy.ndarray: The resulting matrix Q.
    """
    n_P = A_P.shape[0]
    n_Q = A_Q.shape[0]
    
    # Step 2: Estimate clusterings
    if k_P is None: 
        k_P = int(np.ceil((np.sqrt(n_P))))  # You might want to choose k_P in a more informed way
    if k_Q is None: 
        k_Q = int(np.ceil((np.sqrt(n_Q))))  # You might want to choose k_Q in a more informed way

    hat_Z_P = spectral_clustering(A_P, k_P, seed=seed)
    hat_Z_Q = spectral_clustering(A_Q, k_Q, seed=seed)

    # Step 3: Initialize hat_Pi
    hat_Pi = np.zeros((k_P, k_Q), dtype=int)

    z_p_subset = hat_Z_P[xi_subset].copy()

    # hat pi is a least squares soln 
    # hat_pi = np.linalg.pinv(z_p_subset.T @ z_p_subset) @ z_p_subset.T @ hat_Z_Q

    hat_pi, residuals, rank, singvals = np.linalg.lstsq(z_p_subset, 
                                                             hat_Z_Q, 
                                                             rcond=None)
    
    # Step 4: Populate hat_Pi
    # for indx, i in enumerate(xi_subset):
    #     j_P = np.argmax(hat_Z_P[i])
    #     j_Q = np.argmax(hat_Z_Q[indx])
    #     hat_Pi[j_P, j_Q] = 1

    # Normalize the rows of hat_Pi to sum to 1
    # row_sums = hat_Pi.sum(axis=1, keepdims=True).squeeze()
    # # row_sums[row_sums == 0] = 1  # Prevent division by zero

    # row_sum_mat = np.diag(np.divide(1.0, row_sums))
    # hat_Pi_norm = row_sum_mat @ hat_Pi

    # Step 5: Compute hat_B_Q
    aq_float = A_Q.astype(np.float32)

    fraction_matrix = community_fraction_matrix(hat_Z_Q)
    hat_B_Q = fraction_matrix @ hat_Z_Q.T @ aq_float @ hat_Z_Q @ fraction_matrix

    # Step 6: Compute hat_Q
    hat_Q = hat_Z_P @ hat_pi @ hat_B_Q @ hat_pi.T @ hat_Z_P.T

    return hat_Q, hat_pi, hat_B_Q, hat_Z_P, hat_Z_Q



def spectral_clustering_q_est(p_sample_mat, q_sample_mat, subset_indices, kq=None): 
    '''
    Computes kq-dim spectral embedding of both matrices
    where kq = dim of embedding. 

    Default is kq = np.sqrt(nq)
    Default is with quantile of h = np.sqrt(log(nq) / nq)
    '''
    lapl_p  = scipy.sparse.csgraph.laplacian(
        scipy.sparse.csr_matrix(p_sample_mat).astype(np.float32),
        normed=True 
    )
    lapl_q = scipy.sparse.csgraph.laplacian(
        scipy.sparse.csr_matrix(q_sample_mat).astype(np.float32),
        normed=True 
    )
    nq = int(q_sample_mat.shape[0])
    h_quantile = 100 * np.sqrt(np.log(nq) / nq)
    if kq is None: 
        kq = int(np.sqrt(nq))
    
    p_eigvals, p_eigvecs = scipy.sparse.linalg.eigsh(lapl_p, k = kq)
    q_eigvals, q_eigvecs = scipy.sparse.linalg.eigsh(lapl_q, k = kq)

    pdist_out = scipy.spatial.distance.pdist(p_eigvecs)
    pdist_mat = scipy.spatial.distance.squareform(pdist_out)

    pdist_col_subset = pdist_mat[:, subset_indices].copy()
    percentiles = bottom_percentile_mask(pdist_col_subset, h_quantile)

    percentiles_normalized = np.diag(np.divide(1.0, np.sum(percentiles, axis=1))) @ percentiles
    q_hat = percentiles_normalized @ q_sample_mat @ percentiles_normalized.T
    q_hat_norm = np.clip(q_hat, 0, 1)
    return q_hat_norm

def eigenvec_feature_transfer_baseline(p_sample, q_sample, xi_subset, kp = None, kq = None): 
    print('doing ase now.')
    # default is kp, kq are sqrt(nP), sqrt(nQ) resp. 
    p_eigvals, p_eigvecs = np.linalg.eigh(p_sample)
    q_eigvals, q_eigvecs = np.linalg.eigh(q_sample)
    if kp is None: 
        kp = int(np.sqrt(p_sample.shape[0]))
    if kq is None: 
        kq = int(np.sqrt(q_sample.shape[0]))
    p_eigvals_pos = np.abs(p_eigvals)
    q_eigvals_pos = np.abs(q_eigvals)

    p_cutoff = np.sort(p_eigvals_pos)[::-1][kp]
    q_cutoff = np.sort(q_eigvals_pos)[::-1][kq]


    # num_p_eigvals_positive = len(np.where(p_eigvals >= 0)[0])
    # num_q_eigvals_positive = len(np.where(q_eigvals >= 0)[0])
    # kp = min(kp, num_p_eigvals_positive)
    # kq = min(kq, num_q_eigvals_positive)
    
    
    p_indices = np.where(p_eigvals_pos > p_cutoff)[0]
    q_indices = np.where(q_eigvals_pos > q_cutoff)[0]
    # p_eigvals_signs = np.sign(p_eigvals)[p_indices]
    # p_correction_signs = np.diag(p_eigvals_signs)

    p_features = p_eigvecs[:, p_indices].copy()
    q_features = q_eigvecs[:, q_indices].copy()
    q_eigval_marix = np.diag(q_eigvals[q_indices])
    
    # p_features = p_eigvecs[:][p_indices].copy()
    # q_features = q_eigvecs[:][q_indices].copy()
    q_eigval_marix = np.diag(q_eigvals[q_indices])
    
    p_features_restricted = p_features[xi_subset].copy()

    proj_matrix, residuals, rank, singvals = np.linalg.lstsq(p_features_restricted, 
                                                             q_features, 
                                                             rcond=None)
    q_hat = p_features @ proj_matrix @ q_eigval_marix @ proj_matrix.T @ p_features.T
    return np.clip(q_hat, 0, 1)

####################
## bit flipping baseline

def gen_bitflip_matrix(Q_full, subset_indices, p_flip = 0.1): 
    # Generate a symmetric "bit flip matrix."
    # Entry (i,j) = 1 iff that entry is to be flipped in the sample. 
    # Q_full at subset_indices is always set to 0 in the bitflip matrix. 
    # Meaning, those bits are never flipped. 
    """Generate a sparse sample boolean matrix from a dense matrix."""
    flip_probs_mat = p_flip * np.ones_like(Q_full)
    for idx1 in subset_indices: 
        for idx2 in subset_indices: 
            flip_probs_mat[idx1, idx2] = 0.0     
    sparse_mat = scipy.sparse.triu(flip_probs_mat, k=1)
    nonzero_float_entries = sparse_mat.data
    sparse_tri = scipy.sparse.csr_matrix(sparse_mat)
    sample_bool = bin_samples_rand2(1, nonzero_float_entries, seed=42).squeeze(-1)
    sparse_tri.data = sample_bool
    symm_sparse = sparse_tri + sparse_tri.T
    return symm_sparse

def gen_sparse_sample_boolean_mat(mat):
    """Generate a sparse sample boolean matrix from a dense matrix."""
    sparse_mat = scipy.sparse.triu(mat, k=1)
    nonzero_float_entries = sparse_mat.data
    sparse_tri = scipy.sparse.csr_matrix(sparse_mat)
    sample_bool = bin_samples_rand2(1, nonzero_float_entries, seed=42).squeeze(-1)
    sparse_tri.data = sample_bool
    symm_sparse = sparse_tri + sparse_tri.T
    return symm_sparse

def get_usvt_bitflipped_estimator(Q_matrix_full, subset_indices, p_flip = 0.1):
    q_sample =  gen_sparse_sample_boolean_mat(Q_matrix_full).toarray()
    bitflip_mat = gen_bitflip_matrix(Q_matrix_full, subset_indices, p_flip=p_flip).toarray()
    # if bitlip mat contains a zero, nothing changes. 1 --> 1, 0 --> 0
    # if it contains a 1, 1 --> 0 and 0 --> -1. Therefore, doing abs value 
    sample_flipped = q_sample ^ bitflip_mat #XOR operation 
    return usvt(sample_flipped), sample_flipped, q_sample, bitflip_mat


def usvt(sample_mat): 
    # note that the proportion of observed values is assumed to be one. 
    # see also eq 4 of Xu 2018 ("Rates of Convergence of Spectral Methods....")
    threshold = 2.02 * np.sqrt(sample_mat.shape[0])
    uvecs, singvals, vh = np.linalg.svd(sample_mat, hermitian=True)
    indices_kept = np.where(singvals > threshold)[0]
    
    u_subset = uvecs[:, indices_kept].copy()
    singvals_subset = singvals[indices_kept]
    v_subset = vh[indices_kept].copy()
    prod = u_subset @ np.diag(singvals_subset) @ v_subset
    out = np.clip(prod, 0, 1)
    return out


def est_q_three_methods(p_sample, q_sample, q_matrix_full_ground_truth, subset_indices, 
        p_flip_list, d_est_large=None,
    kp=None, kq=None, h_quantile=None): 
    q_hat_ours = matrix_completion_from_matrices(p_sample, q_sample, 
                                                subset_indices, 
                                                d_est_large=d_est_large, 
                                                h_quantile=h_quantile)
    q_hat_bitflips_list = []
    for p_flip in p_flip_list:
        qhat_bitflip, _, _, _ = get_usvt_bitflipped_estimator(q_matrix_full_ground_truth, 
            subset_indices, p_flip = p_flip
        )
        q_hat_bitflips_list.append(qhat_bitflip)
    q_est_sbm, hat_pi, hat_b, zphat, zqhat = sbm_cluster_transfer(
        p_sample, q_sample, subset_indices, 
        k_P = kp, 
        k_Q = kq
    )
    return q_hat_ours, q_est_sbm, q_hat_bitflips_list