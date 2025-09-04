import numpy as np
import scipy 
import scipy.sparse as sp
from . import graphon_families as gf 
from . import graphon_est 
from . import utils


def run_sbm_test(n_vals, num_trials, np_power = 1.5, random_seed=42):
    # np_power: exponent such that n_p = n_q^{np_power}.
    # Can set between [1.0, 2.0]

    frob_errors_transfer = {}
    np.random.seed(random_seed)

    for idx, n in enumerate(n_vals): 
        frob_errors_transfer[n] = []
        for _ in range(num_trials): 
            
            gp = gf.get_sbm_pair(n_Q = n, 
                              n_P = int(np.power(n, np_power)))
            q_hat = graphon_est.matrix_completion_from_pair(gp)
            q_extended_graphon = gp.Q_extended
            err = graphon_est.frob_error(q_extended_graphon.g1, q_hat)
            frob_errors_transfer[n].append(err) 
    return frob_errors_transfer

def get_boolean_sample(sparse_lil_mat): 
    return gen_sparse_sample_boolean_mat(
        utils.truncate_sparse_matrix(sparse_lil_mat.tocsr()))

def gen_sparse_sample_boolean_mat(sparse_input):
    """Generate a sparse sample boolean matrix from a sparse matrix."""
    sparse_mat = scipy.sparse.triu(sparse_input, k=1)
    nonzero_float_entries = sparse_mat.data
    sparse_tri = scipy.sparse.csr_matrix(sparse_mat)
    sample_bool = utils.bin_samples_rand2(1, nonzero_float_entries).squeeze(-1)
    sparse_tri.data = sample_bool
    symm_sparse = sparse_tri + sparse_tri.T
    return symm_sparse