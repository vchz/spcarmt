import numpy as np
from ..rmt._covariance import BiwhitenedCovarianceEstimator
from ..base import FeaturesTransformer, SamplesTransformer
import scipy.optimize
from sklearn.decomposition import SparsePCA
from scipy.sparse import csr_matrix
from scipy.linalg import sqrtm
import rpy2.robjects as robjects
import scanpy as sc

def _lowdin_orthogonalize(W):

    S = W.T @ W
    S_inv_sqrt = np.linalg.inv(sqrtm(S))
    Q = W @ S_inv_sqrt

    return Q

def _gs_decorrelation(w, W):

    w -= np.linalg.multi_dot([w, W.T, W])
    return w
    
def _gs_orthogonalize(W):

    for k in range(W.shape[1]):
        w = W[:,k]
        w = _gs_decorrelation(w, W[:,:k].T)
        w /= np.sqrt((w**2).sum())
        W[:,k] = w

    return W

def _fista_spca(X, n_comps, max_iter=10000, penalty=0.01, 
               tol=1e-3, verbose=False, ortho='gs'):

    
        _ortho_funcs = {
            'gs': _gs_orthogonalize,
            'lowdin': _lowdin_orthogonalize
        }
        _ortho_func = _ortho_funcs[ortho]
        
        tk = 1
        lims = []
        p = 1/20
        q = 1
        r = 4

        S = X.T@X/(X.shape[0]-1)
        s, V = np.linalg.eigh(S)
        lmax = np.max(s)
        W = V[:,-n_comps:][:,::-1]
        W_temp = W.copy()

        def _prox(S, W, t):

            W_ = W + 2*t*S@W
            W_ = np.maximum((np.abs(W_) - penalty*t), 0)*np.sign(W_)

            return W_
        
        t = 0.5/((2*lmax))
        for i in range(max_iter):
            W_old = np.copy(W)

            W = _prox(S, W_temp, t)
            W = _ortho_func(W)

            lim = np.sqrt(np.trace((W - W_old)@(W - W_old).T))
            lims.append(lim)

            if verbose:
                if (i % verbose == 0 and i > 0):
                    print('iter', i, 'lim', lim, 
                          'penalty', penalty)

            tk_temp = (p + np.sqrt(q + r*tk**2))/2
            W_temp = W + (tk - 1)*(W - W_old)/(tk_temp)
            tk = tk_temp

            if lim < tol:
                print('FISTA stopping criterion reached:', 
                      'iter', i, 'lim', lim, 
                      'penalty', penalty)
                break

            
        else:
            print('FISTA reached maximum number of iterations:',
                  'lim', lim, 
                  'penalty', penalty)

        return W, lims
            
def _adjust_penalty(X, target, spca=None, verbose=0):

    if spca is None:
        spca = sklearnSPCA()

    _, V = np.linalg.eigh(X.T@X/(X.shape[0]-1))
    n_comps = spca.n_comps
    V = V[:,-n_comps:][:,::-1]
    
    count = 0
    
    def _overlap_with_data(penalty):

        nonlocal count
        spca.penalty = penalty
        W_acc = spca._get_W(X)
        Z = W_acc.T@V
        _, sv, _ = np.linalg.svd(Z)

        count += 1

        if verbose:
            if count % verbose == 0:
                print('root finding iteration', count,'distance to root', (1 - sv**2).sum() - target, 'penalty', penalty)
       

        return ((1 - sv**2).sum() - target)
    
    bracket = (0.0001, 4)
    if spca.__class__.__name__ == 'AManpgSPCA':
        bracket = (0.0001, 800)
    if spca.__class__.__name__ == 'GpowerSPCA':
        bracket = (0.0001, 0.35)
    if spca.__class__.__name__ == 'FantopeSPCA':
        print('runing fantope, bracket', (0.0001, 0.04))
        bracket = (0.0001, 0.04)

    print('optimizing')
 
    res = scipy.optimize.root_scalar(_overlap_with_data,
            bracket=bracket,
            xtol=5e-3)
    print('optimized, penalty', res.root)
    penalty = res.root

    return penalty
    
class PCA(FeaturesTransformer):

    def __init__(self, n_comps='auto', auto_plot=False):

        self.n_comps = n_comps
        self.auto_plot = auto_plot
        
    def _fit(self, data):
        
        if self.n_comps == 'auto':
            cov_est = BiwhitenedCovarianceEstimator().fit(data)
            score = cov_est.score(data)
            
            print('score', score[0], 'pvalue', score[1])
            print('n_comps', cov_est.n_comps_)

            self.n_comps = cov_est.n_comps_
            
            if self.auto_plot:
                cov_est.plot_eigenspectrum(data)
        
        return self
    
    def _transform(self, data):
                
        sc.pp.pca(data.adata,
                  n_comps=self.n_comps,
                  zero_center=True)
        
        return data
    
    def transform(self, data, y=None):
        
        data = super().transform(data, y)
        self._fit(data)
        data = self._transform(data)

        return data


class AdaptiveSPCA(PCA):

    def __init__(self, scale = 0.6, spca=None, verbose=0):

        if spca is None:
            spca = sklearnSPCA()

        self.spca = spca
        self.scale = scale
        self.verbose = verbose

    def _fit(self, data):

        cov_est = BiwhitenedCovarianceEstimator().fit(data)
        score = cov_est.score(data)
        self.n_comps_ = cov_est.n_comps_
        self.distance_ = cov_est.distance_
        print('score', score[0], 'pvalue', score[1])
        print('n_comps', cov_est.n_comps_)
        
        return self
    
    def _transform(self, data, y = None):
            
        self.spca.n_comps = self.n_comps_
    
        X = data.adata.X
        if isinstance(X, csr_matrix):
            X = X.toarray()

        penalty = _adjust_penalty(X.astype(np.float64), 
                                self.distance_, 
                                spca=self.spca,
                                verbose=self.verbose)

        self.spca.penalty = self.scale*penalty
        W = self.spca._get_W(X)

        data.adata.varm['sPCs'] = W
        data.adata.obsm['X_spca'] = X@W
        data.adata.uns['spca'] = dict(
            n_comps=self.n_comps_,
            opt_penalty=penalty,
            method=self.spca.__class__.__name__,
            penalty=self.scale*penalty
        )

        return data
    
class _SPCA(PCA):

    def __init__(self, n_comps='auto', penalty=0.1, tol=1e-4, verbose=0):

        self.n_comps = n_comps
        self.penalty = penalty
        self.tol = tol
        self.verbose = verbose
        
    def _transform(self, data, y=None):

        X = data.adata.X
        if isinstance(X, csr_matrix):
            X = X.toarray()

        W = self._get_W(X)
        data.adata.varm['sPCs'] = W
        data.adata.obsm['X_spca'] = X@W
        data.adata.uns['spca'] = dict(
            n_comps=self.n_comps,
            method=self.__class__.__name__,
            penalty=self.penalty
        )

        return data
    
class AManpgSPCA(_SPCA):

    def __init__(self, n_comps='auto', penalty=0.1, tol=1e-4, verbose=0, ridge=1, ortho='gs'):

        super().__init__(n_comps = n_comps,
                         penalty = penalty,
                         tol = tol,
                         verbose = verbose)
        
        self.ridge = ridge
        self.ortho = ortho

    def _get_W(self, X):

        try:
            import sparsepca

        except ModuleNotFoundError:
            raise NameError(
                'Module sparsepca is not installed: it is necessary to use AManPG()'
            )
        

        if isinstance(X, csr_matrix):
            X = X.toarray()

        method = {
            'gs': _gs_orthogonalize,
            'lowdin': _lowdin_orthogonalize
        }
            
        print('running amanpg with L1 penalty', self.penalty, 'and L2 penalty', self.ridge)
        ret = sparsepca.spca(X, self.penalty*np.ones((self.n_comps, 1)), self.ridge, 
                                x0=None, 
                                y0=None, 
                                k=self.n_comps, 
                                gamma=0.5, 
                                type=0, 
                                maxiter=2e4, 
                                tol=self.tol,
                                normalize=False, 
                                verbose=self.verbose)
        W = method[self.ortho](ret['loadings'])

        return W
    

class FistaSPCA(_SPCA):

    def __init__(self, n_comps='auto', penalty=0.1, tol=1e-4, verbose=0, ortho='gs'):

        super().__init__(n_comps = n_comps,
                         penalty = penalty,
                         tol = tol,
                         verbose = verbose)
        
        self.ortho = ortho
        
    def _get_W(self, X):

        if isinstance(X, csr_matrix):
            X = X.toarray()

        W, error = _fista_spca(X, self.n_comps,
                               penalty=self.penalty,
                               verbose=self.verbose,
                               tol=self.tol,
                               ortho=self.ortho)

        return W
    

# By default SparsePCA centers the data
class sklearnSPCA(_SPCA):

    def __init__(self, n_comps='auto', penalty=0.1, tol=1e-4, verbose=0, ortho='gs'):

        super().__init__(n_comps = n_comps,
                         penalty = penalty,
                         tol = tol,
                         verbose = verbose)
        
        self.ortho = ortho

    def _get_W(self, X):

        method = {
            'gs': _gs_orthogonalize,
            'lowdin': _lowdin_orthogonalize
        }

        if isinstance(X, csr_matrix):
            X = X.toarray()

        spca = SparsePCA(n_components = self.n_comps, 
                         alpha = self.penalty, 
                         tol=self.tol, 
                         verbose=self.verbose).fit(X)
        W = method[self.ortho](spca.components_.T)

        return W

# This method can produce completely empty vectors
class GpowerSPCA(_SPCA):

    def __init__(self, n_comps='auto', penalty=0.1, tol=1e-4, verbose=0, ortho='gs', block=1):

        super().__init__(n_comps = n_comps,
                         penalty = penalty,
                         tol = tol,
                         verbose = verbose)
        
        self.ortho = ortho
        self.block = block

    def _get_W(self, X):

        method = {
            'gs': _gs_orthogonalize,
            'lowdin': _lowdin_orthogonalize
        }

        robjects.r('''
                library(sparsePCA)
                gpower_spca <- function(data_mat, n_comps, penalty, block) {
                    spca = sparsePCA(data_mat, n_comps, penalty, block=block, scale=FALSE)
                    return(spca)
                }
            ''')
        
        if isinstance(X, csr_matrix):
            X = X.toarray()

        gpower = robjects.globalenv['gpower_spca']
        
        gpower_spca = gpower(X, 
                             self.n_comps, 
                             self.penalty*np.ones(self.n_comps),
                             int(self.block))
        
        W = method[self.ortho](gpower_spca[0])
 
        return W
    
class FantopeSPCA(_SPCA):

    def _get_W(self, X):

        method = {
            'gs': _gs_orthogonalize,
            'lowdin': _lowdin_orthogonalize
        }

        robjects.r('''
                library(gradfps)
                fps_spca <- function(cov_mat, n_comps, penalty) {
                    spca = gradfps_prox(cov_mat, n_comps, lambda=penalty, control=list(verbose=1, fan_maxiter=20))
                    return(spca)
                }
            ''')
        
        if isinstance(X, csr_matrix):
            X = X.toarray()

        fps = robjects.globalenv['fps_spca']
        
        fps_spca = fps(X.T@X/(X.shape[0]-1), 
                       self.n_comps, 
                       self.penalty)
        
        U,_,VT = np.linalg.svd(fps_spca[0])
        W = U[:,:self.n_comps]
        
        return W