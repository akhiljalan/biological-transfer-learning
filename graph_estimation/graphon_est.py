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

def simple_slice_distances_estimate(adj_mat): 
    adj = adj_mat.astype(np.float32)
    prod = adj.T @ adj
    n = adj.shape[0]
    dist_estimates = np.zeros((n, n))
    for i in range(n): 
        for j in range(i): 
            diff = np.abs(prod[i, :] - prod[j, :])
            diff[i] = 0.0
            diff[j] = 0.0
            dist_estimates[i, j] = np.max(diff)
    out = dist_estimates + dist_estimates.T
    return out

def top_percentile_mask(matrix, h):
    """
    Compute a binary mask matrix where each element is 1 if it belongs to the top h percentiles of its row.

    Args:
    - matrix (numpy array): Input matrix of shape (n, n) containing float values.
    - h (float): Percentage threshold (between 0 and 100) to determine top percentiles.

    Returns:
    - mask (numpy array): Binary mask matrix of shape (n, n) where top h percentiles are 1 and the rest are 0.
    """
    # Compute the percentile threshold for each row
    percentile_threshold = np.percentile(matrix, 100 - h, axis=1, method='median_unbiased')

    # Create a binary mask based on the percentile threshold
    mask = (matrix >= percentile_threshold[:, np.newaxis]).astype(np.uint8)

    return mask

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
    d_est = simple_slice_distances_estimate(adj_matrix)
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

    d_est_large = simple_slice_distances_estimate(
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
                                    subset_indices): 
    

    d_est_large = simple_slice_distances_estimate(
        p_sample)

    # subset_indices = gp.subset_indices
    d_est_subset = d_est_large[:, subset_indices].copy()
    nq_nodes = q_sample.shape[0]
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