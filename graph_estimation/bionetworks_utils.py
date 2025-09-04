import numpy as np
import scipy 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

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

def create_df(adj_mat, metabolite_names, compartment_names): 
       sorted_indices, sorted_matrix, sorted_list1, sorted_list2 = sort_matrix_by_row_sum(
              adj_mat, metabolite_names, compartment_names)
       df = pd.DataFrame.sparse.from_spmatrix(sorted_matrix, columns=sorted_list1, index=sorted_list1)


       # df['metabolite'] = sorted_list1
       df['compartment'] = sorted_list2
       df['degree'] = np.mean(sorted_matrix, axis=1)
       return df 

def shared_adj_matrices(adj_1, adj_2, nodes_list_1, nodes_list_2):
    """
    Returns a numpy array containing the indices of elements from `subset` present in `strings_list`.

    Args:
    strings_list (list): A list of strings.
    subset (list): A subset of `strings_list`.

    Returns:
    np.ndarray: A 1D numpy array containing the indices of elements from `subset` present in `strings_list`.
    """
    shared_nodes = list(set(nodes_list_1).intersection(set(nodes_list_2)))

    # Create a mapping of string to index for strings_list
    index_map_1 = {string: idx for idx, string in enumerate(nodes_list_1)}
    index_map_2 = {string: idx for idx, string in enumerate(nodes_list_2)}
    
    # Get indices of elements in subset using the index_map
    indices_1 = [index_map_1[string] for string in shared_nodes if string in index_map_1]
    indices_2 = [index_map_2[string] for string in shared_nodes if string in index_map_2]
    
    
    # Convert list of indices to numpy array
    mat_subset_1 = adj_1[:, indices_1][indices_1]
    mat_subset_2 = adj_2[:, indices_2][indices_2]
    # indices_array = np.array(indices)
    
    return mat_subset_1, mat_subset_2, shared_nodes

def shared_nodes_dfs(df_1, df_2, 
                     compartment_type=None,
                     clip = True):
    """
    NOTE: expects the index and col names to be metabolite names 
    """
    if compartment_type is not None: 
        df_1_cur = df_1[df_1['compartment'] == compartment_type].copy()
        df_2_cur = df_2[df_2['compartment'] == compartment_type].copy()
    else: 
        df_1_cur = df_1.copy()
        df_2_cur = df_2.copy()
    
    shared_nodes = sorted(
        list(set(df_1_cur.index).intersection(
            set(df_2_cur.index)
            )
        )
    )

    df_1_cur_subset = df_1_cur.loc[shared_nodes][shared_nodes].copy()
    df_2_cur_subset = df_2_cur.loc[shared_nodes][shared_nodes].copy()

    

    assert df_1_cur_subset.shape[0] == df_2_cur_subset.shape[0]
    assert df_1_cur_subset.shape[1] == df_2_cur_subset.shape[1]

    non_numeric_cols = list(df_1_cur.select_dtypes(include='object').columns)
    for col in non_numeric_cols: 
        df_1_cur_subset[col] = df_1_cur[col].loc[shared_nodes]
        df_2_cur_subset[col] = df_2_cur[col].loc[shared_nodes]

    return df_1_cur_subset, df_2_cur_subset, shared_nodes


def filter_df_by_metabolites(df_1, metabolites_list): 
    """
    NOTE: expects the index and col names to be metabolite names 
    """
    df_1_cur = df_1.copy()
    shared_nodes = sorted(metabolites_list)

    df_1_cur_subset = df_1_cur.loc[shared_nodes][shared_nodes].copy()

    non_numeric_cols = list(df_1_cur.select_dtypes(include='object').columns)
    for col in non_numeric_cols: 
        df_1_cur_subset[col] = df_1_cur[col].loc[shared_nodes]

    return df_1_cur_subset, shared_nodes