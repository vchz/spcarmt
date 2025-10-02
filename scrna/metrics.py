import numpy as np
import pandas as pd
from collections import defaultdict
import scipy

def pairwise_modularity(connectivity_matrix, clusters, get_weights=False, as_ratio=False):

    clusters = pd.Categorical(clusters)
    cluster_indices = defaultdict(list)
    for idx, label in enumerate(clusters):
        cluster_indices[label].append(idx)

    labels = list(cluster_indices.keys())
    n = len(labels)
    observed = np.zeros((n, n))
    cluster_totals = np.zeros(n)
    matrix = np.array(connectivity_matrix)

    for i, a in enumerate(labels):
        idx_a = cluster_indices[a]
        for j, b in enumerate(labels):
            if j < i:
                continue
            idx_b = cluster_indices[b]
            block = matrix[np.ix_(idx_a, idx_b)]

            weight = block.sum()

            observed[i, j] = weight
            observed[j, i] = weight
            cluster_totals[i] += weight
            if i != j:
                cluster_totals[j] += weight

    total_weight = matrix.sum()
    proportions = cluster_totals / total_weight
    expected = np.outer(proportions, proportions) * total_weight

    if get_weights:
        return {
            "observed": pd.DataFrame(observed, index=labels, columns=labels),
            "expected": pd.DataFrame(expected, index=labels, columns=labels)
        }
    elif as_ratio:
        return pd.DataFrame(observed / expected, index=labels, columns=labels)
    else:
        return pd.DataFrame((observed - expected) / total_weight, index=labels, columns=labels)


def sin2_subspaces(data, ydata, key_x ='sPCs', key_y = 'sPCs'):

    x = data
    y = ydata

    Wx = x.varm[key_x]
    Wy = y.varm[key_y]
    min_dim = min(Wx.shape[1], Wy.shape[1])

    V = Wx.T@Wy
    U, S, VT = np.linalg.svd(V)
    cos_thetas = S[:min_dim]
    return 1 - cos_thetas**2


def umap_weights(distances):
        
    # Because when bootstrapping, many points can have exactly the same position
    # As a result the rootfind might fail
    # We could equivalently change k for the 'true' number of neighbors
    _distances = distances+0.01*np.random.uniform(0,1, size=distances.shape[1]) 
    k = _distances.shape[1]
    rhos = np.array([dist[dist > 0].min() for dist in _distances])
    
    def func(sigma, distance, rho):

        return np.exp(-np.maximum(distance - rho, 0)/sigma).sum() - np.log2(k)
    
    sigmas = np.ones_like(rhos)
    for i in range(_distances.shape[0]):
        try:
            sigmas[i] = scipy.optimize.root_scalar(func, x0=1, bracket=[0.00001, 100*np.max(_distances[i,:])], args=(_distances[i,:], rhos[i]), xtol=1e-2).root
        except:
            print('erreur in umap_weights', _distances[i,:], rhos[i])
            raise
            
    weights = np.exp(-np.maximum(_distances - rhos[:,None], 0)/sigmas[:,None])

    return weights
