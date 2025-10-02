import numpy as np
import scipy.linalg
from ._covariance import biwhitening_fast


def _get_haar_vectors(k, d):
    if k > d:
        raise ValueError("k cannot be greater than d.")

    rand_mat = np.random.randn(d, k)
    # Using scipy routine, otherwise its not uniformly distributed (see stackoverflow)
    q, r = np.linalg.qr(rand_mat)
    d = r.diagonal(offset=0, axis1=-2, axis2=-1)
    q *= (d/abs(d))[..., np.newaxis, :]
    return q[:, :k]

# Good approximation to sparse haar
def _get_sparse_orthogonal_vectors(k, n, sparsity):
    block_size = max(1, int((1-sparsity)* n))
    while n % block_size != 0:
        block_size += 1
    num_blocks = n // block_size
    Q = np.zeros((n, n))
    for i in range(num_blocks):
        rand_mat = np.random.randn(block_size, block_size)
        q, r = np.linalg.qr(rand_mat)
        d = np.diag(r)
        q *= np.sign(d)
        start_idx = i * block_size
        Q[start_idx:start_idx + block_size, start_idx:start_idx + block_size] = q

    # Permute rows and columns to distribute non-zero entries
    row_perm = np.random.permutation(n)
    col_perm = np.random.permutation(n)
    Q = Q[row_perm, :]
    Q = Q[:, col_perm]

    return Q[:, :k]


def _get_spiked_covariance(n, p, k, s2=1.0, lmin = 1, lmax = 30, vec_sparsity = 0, seed=None):
    if k > min(n, p):
        raise ValueError("k cannot be greater than min(n, p).")

    gamma = p / n
    bbp = s2 * np.sqrt(gamma)
    if seed is not None:
        np.random.seed(seed)
    h_vecs = _get_sparse_orthogonal_vectors(k, p, sparsity=vec_sparsity)
    eigvals = np.linspace(bbp + lmin, bbp + lmax, k)

    sq_mat = h_vecs @ np.diag(eigvals) @ h_vecs.T
    return np.eye(p) * s2 + sq_mat, h_vecs[:,::-1], s2 + eigvals[::-1]

def get_random_spiked_sparse(n, p, k, s2=1.0, seed=None, lmin=0.5, lmax=10, 
                             vec_sparsity=0, mat_sparsity=0.5):

    cov_mat, h_vecs, pop_spikes = _get_spiked_covariance(n, p, k, s2, seed = seed,
                                                         lmin=lmin, lmax=lmax, 
                                                         vec_sparsity=vec_sparsity)
    sqrt_cov = np.real(scipy.linalg.sqrtm(cov_mat))
    np.random.seed()
    rand_mat = np.random.randn(n, p)
    mask = np.random.rand(n, p) < mat_sparsity
    rand_mat[mask] = 0
    D1, D2 = biwhitening_fast(rand_mat)        
    rand_mat = D1@rand_mat@D2
    
    return rand_mat @ sqrt_cov, h_vecs, pop_spikes

def get_random_spiked(n, p, k, s2=1.0, seed=None, lmin=0.5, lmax=10, vec_sparsity=0):

    cov_mat, h_vecs, pop_spikes = _get_spiked_covariance(n, p, k, s2, 
                                                         lmin=lmin, lmax=lmax, seed=seed,
                                                         vec_sparsity=vec_sparsity)
    sqrt_cov = np.real(scipy.linalg.sqrtm(cov_mat))
    np.random.seed()
    rand_mat = np.random.randn(n, p)

    return rand_mat @ sqrt_cov, h_vecs, pop_spikes


def get_overlap(eigenvectors, initial_vectors, sum=False):
    overlaps = (eigenvectors.T @ initial_vectors)**2
    if not sum:
        return overlaps.diagonal()

    return overlaps.sum(axis = 1)

def predicted_overlap(pop_spike, gamma, s2=1):
    _pop_spike = pop_spike/s2
    return ((_pop_spike - 1)**2 - gamma) / ((_pop_spike - 1) * (_pop_spike - 1 + gamma))

def predicted_eigval(pop_spike, gamma, s2=1):
    _pop_spike = pop_spike/s2
    return s2*(_pop_spike + gamma*_pop_spike/(_pop_spike-1))

def invert_predicted_eigval(sample_spike, gamma, s2=1):
    _sample_spike = sample_spike/s2
    mbar = ((gamma - 1 - _sample_spike) + np.sqrt((_sample_spike - 1 - gamma)**2 - 4*gamma))/2/_sample_spike
    return (-1/mbar)*s2