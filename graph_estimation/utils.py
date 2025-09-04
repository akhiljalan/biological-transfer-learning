import numpy as np
import scipy 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

def pickle_dump(obj, file_path):
    """
    Serialize the given object and save it to the specified file using pickle.

    Args:
        obj: The Python object to be serialized.
        file_path (str): The path to the file where the serialized data will be saved.
        
    Returns:
        None
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object serialized and saved to '{file_path}' successfully.")

    except IOError:
        print(f"Error occurred while writing to '{file_path}'.")

def pickle_load(file_path):
    """
    Load a pickled object from the specified file.

    Args:
        file_path (str): The path to the file containing the pickled data.

    Returns:
        obj: The Python object loaded from the file.
    """
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        print(f"Object loaded from '{file_path}' successfully.")
        return obj

    except IOError:
        print(f"Error occurred while reading from '{file_path}'.")
        return None

def bin_samples_rand2(n, mu, seed=42):
    '''
    mu: List of floats in [0,1] describing Bernoulli mean probabilities. 
    n: number of samples per entries of mu. 
    '''
    rng = np.random.default_rng(seed=seed)
    return (rng.random(size=(len(mu), n)) < mu[:, None]).astype(np.uint8)

def gen_sparse_sample_boolean_mat(sparse_mat): 
    nonzero_float_entries = scipy.sparse.triu(sparse_mat, k=1).data
    sparse_tri = scipy.sparse.csr_matrix(scipy.sparse.triu(sparse_mat, k = 1))
    sample_bool = bin_samples_rand2(1, nonzero_float_entries).squeeze(-1)
    sparse_tri.data = sample_bool
    symm_sparse = sparse_tri + sparse_tri.T
    return symm_sparse

def gen_dense_sample_boolean_mat(input_probs_mat): 
    # Extract nonzero float entries from the upper triangular part (excluding diagonal)
    sparse_sample_mat = scipy.sparse.csr_matrix(input_probs_mat)
    symm_sparse = gen_sparse_sample_boolean_mat(sparse_sample_mat)
    symm_dense = symm_sparse.toarray()
    return symm_dense

def vis_heatmap(g, savepath=None): 
    sns.heatmap(g, cmap='viridis', 
                square=True, xticklabels=False, yticklabels=False)
    if savepath:
        plt.savefig(savepath)

def get_indices_in_set(xi_subset, xi_set):
    """
    Returns a numpy array of indices corresponding to elements in `xi_subset` within `xi_set`.

    Args:
    xi_subset (numpy.ndarray): 1D array of elements from which to find indices in `xi_set`.
    xi_set (numpy.ndarray): 1D array representing the reference set of elements.

    Returns:
    numpy.ndarray: 1D array of indices where elements from `xi_subset` are found in `xi_set`.
    """
    # Create a mapping of elements in xi_set to their indices
    index_map = {element: idx for idx, element in enumerate(xi_set)}
    
    # Get indices of elements in xi_subset using the index_map
    indices = np.array([index_map[element] for element in xi_subset if element in index_map])
    
    return indices

def sort_matrix_by_row_sum(matrix, list1, list2):
    """
    Sorts the rows of a scipy sparse matrix based on descending row sums,
    and permutes the corresponding lists accordingly.

    Parameters:
    matrix (scipy.sparse.csr_matrix): Input sparse matrix (N x N).
    list1 (list): First list of length N.
    list2 (list): Second list of length N.

    Returns:
    tuple: A tuple containing:
           - sorted_indices (numpy.ndarray): Indices representing the sorted order of rows.
           - sorted_matrix (scipy.sparse.csr_matrix): Matrix with rows sorted by descending row sums.
           - sorted_list1 (list): First list permuted according to the sorted matrix rows.
           - sorted_list2 (list): Second list permuted according to the sorted matrix rows.
    """
    # Calculate row sums of the matrix
    row_sums = np.asarray(matrix.sum(axis=1)).flatten()

    # Get sorting indices based on descending row sums
    sorted_indices = np.argsort(row_sums)[::-1]

    # Sort the matrix rows and columns based on sorted indices
    sorted_matrix = matrix[sorted_indices, :][:, sorted_indices]

    # Permute the lists based on sorted indices
    sorted_list1 = [list1[i] for i in sorted_indices]
    sorted_list2 = [list2[i] for i in sorted_indices]

    return sorted_indices, sorted_matrix, sorted_list1, sorted_list2

def truncate_sparse_matrix(matrix):
    """
    Truncate the values of a sparse matrix to the range [0, 1].

    Parameters:
    matrix (scipy.sparse.spmatrix): Input sparse matrix.

    Returns:
    scipy.sparse.spmatrix: Sparse matrix with truncated values in [0, 1].
    """
    # Ensure matrix is a scipy.sparse.spmatrix
    if not scipy.sparse.isspmatrix(matrix):
        raise ValueError("Input matrix must be a scipy.sparse matrix.")

    # Create a copy of the input matrix (to avoid modifying the original)
    truncated_matrix = matrix.copy()

    # Truncate values to [0, 1]
    truncated_matrix.data = np.clip(truncated_matrix.data, 0, 1)

    return truncated_matrix