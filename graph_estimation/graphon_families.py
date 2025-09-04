import numpy as np
import scipy 
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from .utils import bin_samples_rand2, get_indices_in_set, gen_dense_sample_boolean_mat
from scipy.stats import dirichlet

def gen_theta_two_param(k, a, b): 
    # gets a k x k connectivities matrix 
    # diagonal is a, off diagonal is b 
    theta = b * np.ones((k,k)) + np.diag((a - b) * np.ones(k))
    return theta 

def gen_noisy_theta_two_param(k, a, b, eps=0.01, seed=42): 
    # gets a k x k connectivities matrix 
    # diagonal is a, off diagonal is b 
    theta = b * np.ones((k,k)) + np.diag((a - b) * np.ones(k))
    np.random.seed(seed)
    noise = np.random.uniform(-1 * eps, eps, size=(k, k))
    noisy_theta = np.clip(theta + noise, 0, 1)
    return noisy_theta 

def graphon_mmsb(x, y, theta): 
    prob = x.T @ theta @ y
    return prob    

def graphon_liza_2(x, y, period=5, bias=0.5, phase_shift = True, invert=False): 
    # sin{5π(u+v−1)+1}/2+0·5
    phase = 1 + period * (x + y - 1) 

    if phase > 0.3 and phase < 0.7 and phase_shift: 
        max_phase = 0.5
        scale = 0.5 
        phase = max_phase + scale * (phase - max_phase) 
    val = (1-bias) + (bias) * np.sin(phase * np.pi)
    if invert:
        return 1.0 - val
    return val

def graphon_liza_rotated(x, y, period=5, bias=0.5, phase_shift = False, invert=False): 
    # sin{5π(u+v−1)+1}/2+0·5
    x_diff = np.abs(x - 0.5)
    if x > 0.5: 
        x_new = 0.5 - x_diff
    else: 
        x_new = 0.5 + x_diff
    return graphon_liza_2(x_new, y, period=period, bias=bias,phase_shift=phase_shift, invert=invert)
    

def graphon_d_dim(x, y, alpha=0.1): 
    diff = 0.5 * np.linalg.norm(x - y) 
    return 0.5 - np.power(diff, alpha)


def graphon_d_dim_exp(x, y, scale=2.5): 
    diff = scale * np.linalg.norm(x - y) 
    out = 1.0 / np.exp(diff)
    return out


def graphon0(x, y, k = 20, p=0.1, q=0.01):
    t1 = np.floor(k * x)
    t2 = np.floor(k * y)
    
    if t1 == t2:
        return p
    else:
        return q

def graphon1(x, y):
    """
    Compute the value of graphon1 for given inputs x and y.

    Parameters:
    - x (float): Input value x.
    - y (float): Input value y.

    Returns:
    - val (float): Computed value of graphon1.
    """
    val = np.sin(5 * np.pi * (x + y - 1) + 1) / 2 + 0.5
    return val

# Define other graphon functions (graphon0, graphon2, graphon3, graphon4, graphon5) similarly.

def graphon2(x, y):
    """
    Compute the value of graphon2 for given inputs x and y.

    Parameters:
    - x (float): Input value x.
    - y (float): Input value y.

    Returns:
    - val (float): Computed value of graphon2.
    """
    val = 1 - 0.5 * max(x, y)
    return val

def graphon3(x, y):
    """
    Compute the value of graphon3 for given inputs x and y.

    Parameters:
    - x (float): Input value x.
    - y (float): Input value y.

    Returns:
    - val (float): Computed value of graphon3.
    """
    val = 1 - (1 + np.exp(-15 * (0.8 * abs(x - y)) ** (4 / 5) - 0.1)) ** (-1)
    return val

def graphon4(x, y):
    """
    Compute the value of graphon4 for given inputs x and y.

    Parameters:
    - x (float): Input value x.
    - y (float): Input value y.

    Returns:
    - val (float): Computed value of graphon4.
    """
    val = ((x**2 + y**2) / 3) * np.cos(1 / (x**2 + y**2)) + 0.15
    return val

def graphon5(x, y):
    """
    Compute the value of graphon5 for given inputs x and y.

    Parameters:
    - x (float): Input value x.
    - y (float): Input value y.

    Returns:
    - val (float): Computed value of graphon5.
    """
    val = 1 / (1 + np.exp(-x - y))
    return val

def graphon_alpha(x, y, smoothness=0.1, bias=1.0, scale=1.0): 
    base = 0.5 * (np.power(x, smoothness) + np.power(y, smoothness))
    out = (bias + base * scale) / (bias + scale)
    # diff = np.abs(x - y)
    # z1 = scale * (1.0 - np.power(diff, alpha))
    # z2 = bias + z1
    # total = (z1 + z2) / (bias + scale)
    return out

def graphon_alpha_d_dim(x, y, 
                        x_center,
                        y_center,
                        dim=1, 
                        smoothness=0.1, bias=0.0, scale=1.0): 
    base = 0.5 * (np.power(np.linalg.norm(x - x_center), smoothness) + 
                  np.power(np.linalg.norm(y - y_center), smoothness))
    # norm_factor = np.sqrt(dim)
    out = (bias + base * scale) / (bias + scale)
    return np.clip(out, 0, 1)



def graphon_alpha_old(x, y, alpha=0.1, bias=1.0, scale=1.0): 
    diff = np.abs(x - y)
    z1 = scale * (1.0 - np.power(diff, alpha))
    z2 = bias + z1
    total = (z1 + z2) / (bias + scale)
    return total

def gen_latent_positions(n):
    """Generate latent positions xi for n nodes."""
    # Generate n random numbers between 0 and 1 using NumPy's random.rand function
    random_numbers = np.random.rand(n)
    
    # Sort the random numbers in ascending order using NumPy's sort function
    sorted_random_numbers = np.sort(random_numbers)
    
    return sorted_random_numbers

class Graphon:
    def __init__(self, xi_set, 
                 graphon_function = graphon1):
        # self.graphon_fn = graphon_function
        self.xi = xi_set
        self.n = len(xi_set)
        self.graphon_fn = graphon_function
        self.g1 = self.generate_graphon_matrix()
        self.g1_sample = self.generate_sparse_sample()
        
    def generate_graphon_matrix(self):
        """Generate the graphon matrix g1."""
        g1 = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                g1[i, j] = self.graphon_fn(self.xi[i], self.xi[j])
        return g1

    def generate_sparse_sample(self):
        """Generate a sparse sample of the graph matrix g1."""
        return scipy.sparse.csr_matrix(self.gen_sparse_sample_boolean_mat())

    def gen_sparse_sample_boolean_mat(self):
        """Generate a sparse sample boolean matrix from a dense matrix."""
        sparse_mat = scipy.sparse.triu(self.g1, k=1)
        nonzero_float_entries = sparse_mat.data
        sparse_tri = scipy.sparse.csr_matrix(sparse_mat)
        sample_bool = bin_samples_rand2(1, nonzero_float_entries).squeeze(-1)
        sparse_tri.data = sample_bool
        symm_sparse = sparse_tri + sparse_tri.T
        return symm_sparse

class GraphonPair:
    def __init__(self, dim=1,
                 p_graphon_fn = graphon1, 
                 q_graphon_fn = graphon1, n_p=500, n_q=100):
        self.np = n_p  # Size of the full graphon matrix P
        self.nq = n_q  # Size of the subsampled square matrix Q

        xi_set, xi_subset, sample_nodes = self.gen_xi_pair()
        self.xi_set = xi_set 
        self.xi_subset = xi_subset
        self.subset_indices = sample_nodes
        # Generate the full graphon matrix P
        self.P = Graphon(xi_set=xi_set, 
                         graphon_function=p_graphon_fn)
                                 
        self.Q = Graphon(xi_set=xi_subset,
                         graphon_function=q_graphon_fn)
        self.Q_extended = Graphon(xi_set = xi_set, 
                                graphon_function = q_graphon_fn)


    def gen_xi_pair(self): 
        random_numbers = np.random.rand(self.np)
    
        # Sort the random numbers in ascending order using NumPy's sort function
        xi_set = np.sort(random_numbers)
        
        sample_nodes = np.sort(np.random.choice(self.np, self.nq, replace=False))
        xi_subset = np.sort(xi_set[sample_nodes])

        return xi_set, xi_subset, sample_nodes


class GraphonHighDim:
    def __init__(self, xi_set, dim=1, 
                 graphon_function = graphon1):
        # self.graphon_fn = graphon_function
        self.xi = xi_set
        self.n = len(xi_set)
        self.graphon_fn = graphon_function
        self.g1 = self.generate_graphon_matrix()
        self.g1_sample = self.generate_sparse_sample()
        
    def generate_graphon_matrix(self):
        """Generate the graphon matrix g1."""
        g1 = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                g1[i, j] = self.graphon_fn(self.xi[i], self.xi[j])
        return g1

    def generate_sparse_sample(self):
        """Generate a sparse sample of the graph matrix g1."""
        return scipy.sparse.csr_matrix(self.gen_sparse_sample_boolean_mat())

    def gen_sparse_sample_boolean_mat(self):
        """Generate a sparse sample boolean matrix from a dense matrix."""
        sparse_mat = scipy.sparse.triu(self.g1, k=1)
        nonzero_float_entries = sparse_mat.data
        sparse_tri = scipy.sparse.csr_matrix(sparse_mat)
        sample_bool = bin_samples_rand2(1, nonzero_float_entries).squeeze(-1)
        sparse_tri.data = sample_bool
        symm_sparse = sparse_tri + sparse_tri.T
        return symm_sparse


class GraphonPairHighDim:
    def __init__(self, dim=1,
                 p_graphon_fn = graphon1, 
                 q_graphon_fn = graphon1, n_p=500, n_q=100):
        self.np = n_p  # Size of the full graphon matrix P
        self.nq = n_q  # Size of the subsampled square matrix Q
        self.dim = dim
        xi_set, xi_subset, sample_nodes = self.gen_xi_pair()
        self.xi_set = xi_set 
        self.xi_subset = xi_subset
        self.subset_indices = sample_nodes
        # Generate the full graphon matrix P
        self.P = GraphonHighDim(xi_set=xi_set, dim=dim,
                         graphon_function=p_graphon_fn)
                                 
        self.Q = GraphonHighDim(xi_set=xi_subset, dim=dim,
                         graphon_function=q_graphon_fn)
        self.Q_extended = GraphonHighDim(xi_set = xi_set, dim=dim,
                                graphon_function = q_graphon_fn)


    def gen_xi_pair(self): 
        # generate random points on unit sphere 
        random_positions = np.random.normal(size=(self.np, self.dim))
        row_norms = np.linalg.norm(random_positions, axis=1)
        norm_matrix = np.diag(np.divide(1.0, row_norms))
        positions_normalized = norm_matrix @ random_positions
        
        # scale by unif(0,1)^{1/d} to get unif on the ball
        unif_scales = np.random.uniform(low=0.0, high=1.0, 
            size=self.np)
        scales = [np.power(x, 1.0 / self.dim) for x in unif_scales]
        scale_mat = np.diag(scales)
        xi_set = scale_mat @ positions_normalized

        
        sample_nodes = np.sort(np.random.choice(self.np, self.nq, replace=False))
        xi_subset = xi_set[sample_nodes]

        return xi_set, xi_subset, sample_nodes

    
def get_sbm_pair(n_Q, n_P, kp_power = 0.5, kq_power = 0.5, 
                 ap = 0.7, bp = 0.3,
                 aq = 0.9, bq=0.1): 
    kq = np.floor(np.power(n_Q, kp_power))
    kp = np.floor(np.power(n_P, kq_power))
    sbm_fn_p = lambda x, y: graphon0(x, y, k = kp, p = ap, q=bp)
    sbm_fn_q = lambda x, y: graphon0(x, y, k = kq, p = aq, q=bq)
    
    gp = GraphonPair(
                p_graphon_fn=sbm_fn_p,
                q_graphon_fn=sbm_fn_q,
                n_p=n_P,
                n_q=n_Q
            )
    return gp 

def get_mmsb_pair_two_param(n_Q, n_P, 
                kp_power = 0.5, kq_power = 0.5, 
                  ap = 0.7, bp = 0.3,
                 aq = 0.9, bq=0.1,
                 noisy=True, 
                 noise_level=0.01, seed=42):
    np.random.seed(seed) 
    kq = int(np.floor(np.power(n_Q, kp_power)))
    kp = int(np.floor(np.power(n_P, kq_power)))
    xi_set = dirichlet(alpha = np.ones(int(kp))).rvs(n_P)
    xi_subset = np.random.choice(n_P, size=n_Q)
    comm_subset = np.random.choice(kp, size=kq)

    xi_set_q = xi_set[:, comm_subset].copy()
    row_sums = np.sum(xi_set_q, axis=1)
    d_mat = np.diag(np.divide(1.0, row_sums))
    xi_set_q_normalized = d_mat @ xi_set_q
    # subset_indices = get_indices_in_set(xi_subset, xi_set)

    if not noisy: 
        theta_P = gen_theta_two_param(kp, ap, bp)
        theta_Q = gen_theta_two_param(kq, aq, bq)
    if noisy: 
        theta_P = gen_noisy_theta_two_param(kp, ap, bp, eps=noise_level)
        theta_Q = gen_noisy_theta_two_param(kq, aq, bq, eps=noise_level)

    p_matrix_full = xi_set @ theta_P @ xi_set.T
    q_matrix_full = xi_set_q_normalized @ theta_Q @ xi_set_q_normalized.T
    q_submatrix = q_matrix_full[:, xi_subset][xi_subset].copy()
    
    p_sample = gen_dense_sample_boolean_mat(p_matrix_full)
    q_sample = gen_dense_sample_boolean_mat(q_submatrix)

    return p_matrix_full, q_matrix_full, p_sample, q_sample, xi_subset