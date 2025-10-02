import scipy.optimize
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ..base import _BaseTransformer
from scipy.sparse import csr_matrix
from scipy.stats import kstest
from tqdm import tqdm
import scipy.stats
import numpy as np
import scipy.linalg
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _get_scaling_marchenko_pastur(X):

    n, p = X.shape
    if isinstance(X, csr_matrix):
        X = X.toarray()

    Y = X/np.sqrt(n)
    if p > n:
        Y = X.T/np.sqrt(p)
    eigs, _ = np.linalg.eigh((Y.T@Y))
    lmed = np.median(eigs)
    alpha = lmed/marchenko_pastur_median(p/n)
    
    return alpha


def marchenko_pastur_density(x, q):

    lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2

    return ((1/(2*np.pi*x*q)) if (x > 0) else 0)*(0 if (x > lambda_max or x < lambda_min) else np.sqrt((lambda_max-x)*(x-lambda_min)))

def marchenko_pastur_cumulative(x, q):

    lambda_min = (1 - np.sqrt(q))**2

    F = 0
    if (q > 1):
        F = 1 - 1/q

    if (x >= lambda_min):
        F += scipy.integrate.quad(marchenko_pastur_density,
                                  lambda_min,
                                  x,
                                  args=(q))[0]

    return F

def marchenko_pastur_median(q):

    if q > 1:
        q = 1/q
        
    lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2

    def _func(x):
        _int = scipy.integrate.quad(
            marchenko_pastur_density, lambda_min,  x, args=(q))[0]
        return _int - 0.5

    return scipy.optimize.brentq(_func,
                                 lambda_min,
                                 lambda_max,
                                 xtol=0.003)


def biwhitening(X, max_iter=1000, vperiod=5, tol=1e-5):

    if isinstance(X, csr_matrix):
        X = X.toarray()
    P = X**2
    _P = np.copy(P)
    p = P.shape[1]
    n = P.shape[0]

    r = p*np.ones((n, 1))
    c = n*np.ones((p, 1))
    _D1 = np.diag(np.squeeze(r))
    _D2 = np.diag(np.squeeze(c))

    F1 = 0
    F2 = 0

    it = 0
    for i in range(max_iter):

        Fold1 = F1
        Fold2 = F2

        _X = np.sqrt(_D1).dot(X).dot(np.sqrt(_D2))
        target_col = 1 + _X.mean(axis=0)[:, None]**2
        c = n*target_col / P.T.dot(r)
        _D2 = np.diag(np.squeeze(c))

        _X = np.sqrt(_D1).dot(X).dot(np.sqrt(_D2))
        target_row = 1 + _X.mean(axis=1)[:, None]**2
        r = p*target_row / P.dot(c)
        _D1 = np.diag(np.squeeze(r))

        _X = np.sqrt(_D1).dot(X).dot(np.sqrt(_D2))
        F1 = np.abs(_X.var(axis=0) - 1)
        F2 = np.abs(_X.var(axis=1) - 1)
        lim1 = np.max(np.abs((F1 - Fold1)))
        lim2 = np.max(np.abs((F2 - Fold2)))
        

        if it % vperiod == 0:
            print('iter', it, 'lim1', lim1, 'lim2', lim2)
        it += 1

        if max(lim1, lim2) < tol:
            print('stopping criterion reached', 'iter', it, 'lim1', lim1, 'lim2', lim2)
            break

    D1 = np.diag(np.sqrt(np.squeeze(r)))
    D2 = np.diag(np.sqrt(np.squeeze(c)))

    alpha = _get_scaling_marchenko_pastur(_X)

    return D1/np.sqrt(alpha), D2


def biwhitening_fast(X, max_iter=200, vperiod=5, tol=1e-5):
    
    if not isinstance(X,csr_matrix):
        X = csr_matrix(X)
        
    P = X.power(2)
    _P = P.copy()
    p = P.shape[1]
    n = P.shape[0]

    r = p*np.ones((n, 1))
    c = n*np.ones((p, 1))
    _D1 = scipy.sparse.diags(np.squeeze(r))
    _D2 = scipy.sparse.diags(np.squeeze(c))

    F1 = 0
    F2 = 0
    it = 0
    for i in range(max_iter):

        Fold1 = F1
        Fold2 = F2

        _X = np.sqrt(_D1)@X@np.sqrt(_D2)
        target_col = 1 + np.asarray(_X.mean(axis=0)).T**2
        
        c = n*target_col / np.asarray(P.T@r)
        _D2 = scipy.sparse.diags(np.squeeze(c))

        _X = np.sqrt(_D1)@X@np.sqrt(_D2)
        target_row = 1 + np.asarray(_X.mean(axis=1))**2
        r = p*target_row / np.asarray(P@c)
        _D1 = scipy.sparse.diags(np.squeeze(r))

        _X = np.sqrt(_D1)@X@np.sqrt(_D2)
        F1 = np.abs(_X.power(2).mean(axis = 0) - np.square(_X.mean(axis=0)))
        F2 = np.abs(_X.power(2).mean(axis = 1) - np.square(_X.mean(axis=1)))
        lim1 = np.max(np.abs((F1 - Fold1)))
        lim2 = np.max(np.abs((F2 - Fold2)))
        

        if it % vperiod == 0:
            print('iter', it, 'lim1', lim1, 'lim2', lim2)
        it += 1

        if max(lim1, lim2) < tol:
            print('stopping criterion reached', 'iter', it, 'lim1', lim1, 'lim2', lim2)
            break

    D1 = scipy.sparse.diags(np.sqrt(np.squeeze(r)))
    D2 = scipy.sparse.diags(np.sqrt(np.squeeze(c)))
    
    alpha = _get_scaling_marchenko_pastur(_X)
    
    return D1/np.sqrt(alpha), D2


def _compute_esd(tA,
                tB,
                wA=None,
                wB=None,
                q=None,
                epsilon=1e-2,
                grid=None,
                maxIter=None):

    if wA is None:
        wA = np.ones(len(tA)) / len(tA)

    if wB is None:
        wB = np.ones(len(tB)) / len(tB)

    if maxIter is None:
        maxIter = int(int(1e3) / epsilon)

    tol = epsilon

    if grid is None:
        grid = np.linspace(0.1, 10, num=int(1e3))

    grid_imag = grid + 1j * epsilon**2
    L = len(grid)

    g1 = np.zeros(L, dtype=complex)
    g2 = np.zeros(L, dtype=complex)
    m = np.zeros(L, dtype=complex)

    for i in tqdm(range(L)):
        z = grid_imag[i]

        def fun_g1(x):
            return q * np.sum(wA * tA / (-z*(1 + tA * x)), axis=0)

        def fun_g2(x):
            return np.sum(wB * tB / (-z*(1 + tB * x)), axis=0)

        v1 = [0, fun_g1(-1 / z)]
        v2 = [-1 / z, fun_g2(0)]

        if i > 0:
            v1 = [g1[i - 1], fun_g1(g2[i - 1])]
            v2 = [g2[i - 1], fun_g2(g1[i - 1])]

        j = 1
        while max(abs((v1[j] - v1[j - 1])/v1[j]),
                  abs((v2[j] - v2[j - 1])/v2[j])) > tol:
            
            v1.append(fun_g1(v2[j]))
            v2.append(fun_g2(v1[j]))
            j += 1

        g1[i] = v1[j]
        g2[i] = v2[j]
        m[i] = np.sum(wA / (-z*(1 + tA * g2[i])), axis=0)

    density = np.imag(m)/np.pi
    mbar = q*m - (1-q)/grid_imag

    return density, m, mbar, g2


# RMT covariance estimator. Needs to have fitted 
class _RMTCovarianceEstimator(_BaseTransformer):


    def _get_cdf(self):

        def _cumu(x):
            vals = (self.grid_ <= x)
            tot = (self.density_[:-1]*np.diff(self.grid_)).sum()
            return ((self.density_[vals][:-1]*np.diff(self.grid_[vals])).sum())/tot

        cdf = np.vectorize(lambda x: _cumu(x))

        return cdf
    

    def score(self, data):

        X = data.adata.X
        if isinstance(X, csr_matrix):
            X = X.toarray()

        n, p = X.shape
        S = (X.T@X)/(n-1)
        eigs, V = np.linalg.eigh(S)

        cdf = self._get_cdf()
        
        return kstest(eigs[(eigs >= 1e-4)], cdf)


class BiwhitenedCovarianceEstimator(_RMTCovarianceEstimator):


    def fit(self, data ,y = None):

        super().fit(data, y)
        
        n, p = data.adata.X.shape
        X = data.adata.X
        if isinstance(X, csr_matrix):
            X = X.toarray()

        q = p/n
        x_min = .0001 if (1 - np.sqrt(q))**2 < .0001 else (1 - np.sqrt(q))**2
        x_max = (1 + np.sqrt(q))**2
        x = np.linspace(x_min, 1.3*x_max, 5000)
        _density = np.vectorize(lambda x, q: marchenko_pastur_density(x, q))
        y = _density(x, q)
    
        S = (X.T@X)/(n-1)
        eigs, _ = np.linalg.eigh(S)
        eigs = eigs[::-1]
        # We take slightly above the edge
        idx_above = (eigs > x_max + 0.01)
        
        self.density_ = y
        self.grid_ = x
        self.q_ = p/n
        self.lplus_ = x_max
        self.lminus_ = x_min
        self.obs_eigs_ = eigs[idx_above]
        self.true_eigs_ = -1/self._mbar(self.obs_eigs_)
        self.n_comps_ = len(self.obs_eigs_)
        self.proj_ = self._proj(self.true_eigs_)
        self.distance_ = (1 - self.proj_).sum()

        return self

    def plot_eigenspectrum(self, data):


        n, p = data.adata.X.shape
        X = data.adata.X
        if isinstance(X, csr_matrix):
            X = X.toarray()

        S = (X.T@X)/(n - 1)
        eigs, V = np.linalg.eigh(S)
        xval = self.lplus_

        spikes_edge = xval + 0.01
        plot_edge = np.floor(1.2*xval)
        bins_left = np.linspace(
            eigs.min(), spikes_edge, p//14, endpoint=True)
        
        a = np.min(np.diff(eigs[eigs > spikes_edge]))
        bins_right = np.sort(np.concatenate((np.maximum(eigs[eigs > spikes_edge] - a, spikes_edge),
                                     eigs[eigs > spikes_edge] + a)))
        
    
        bins = np.concatenate((bins_left, bins_right))

        density = [self.grid_, self.density_]
        vals, bins = np.histogram(eigs,
                                density=True,
                                bins=bins)
        
        eigbins = np.real(np.array([0.5 * (bins[i] + bins[i+1])
                                    for i in range(len(bins)-1)]))

        fig, ax = plt.subplots(nrows=1, ncols=1)
        width = np.diff(bins)

        ax.bar(eigbins, vals, width=width, color='C0', label='data')
        ax.plot(density[0], density[1], 'C3', label='theo.', linestyle='--')
        
        axins = ax.inset_axes([0.46, 0.58, 0.52, 0.40],
                            transform=ax.transAxes)
        
        xm = xval - 1

        axins.bar(eigbins[eigbins > xm], vals[eigbins > xm], width=width[eigbins > xm], color='C0', label='data')
        axins.plot(density[0][density[0] > xm], density[1][density[0] > xm], 'C3', label='theo.', linestyle='--')

        axins.set_yscale('log')
        axins.set_xscale('linear')
        axins.set_ylim(ymin = 1.0e-2)
        axins.spines['right'].set_visible(False)
        axins.set_xlim(xmin = xm, xmax = plot_edge)  

        x0, x1 = axins.get_xlim()
        visible_ticks = [t for t in axins.get_xticks() if t>=x0 and t<=x1] + [xval]
        visible_labels = [l for t, l in zip(axins.get_xticks(), axins.get_xticklabels()) if t>=x0 and t<=x1] + [r'$\lambda_{+}$']

        axins.set_xticks(visible_ticks)
        axins.set_xticklabels(visible_labels)
        
        divider = make_axes_locatable(axins)
        axlog = divider.append_axes("right", size=0.6, pad=-0.026, sharey=axins)
        width = (10**(0.02) - 1)*eigbins[eigbins > xm]
                
        score = self.score(data)
        axlog.annotate(r'KS=%.3f' % score[0], (0.52, 0.85), xycoords = 'axes fraction', fontsize=5)
        axlog.annotate(r'pval=%.3f' % score[1], (0.52, 0.65), xycoords = 'axes fraction', fontsize=5)

        axlog.bar(eigbins[eigbins > xm], vals[eigbins > xm], width=width, color='C0', label='data')
        axlog.set_xscale('log')
        axlog.set_xlim(xmin = plot_edge)
        axlog.spines['left'].set_visible(False)
        axlog.yaxis.set_ticks_position('right')
        axlog.yaxis.set_visible(False)
        
        ax.legend(frameon = False, loc='upper left')
        ax.set_xlabel('eigenvalues')
        ax.set_ylabel('density')
        ax.set_xlim(xmin = -0.05, xmax = plot_edge)
        ax.set_ylim(ymax = 1.1*np.max(density[1]))

        x0, x1 = ax.get_xlim()
        visible_ticks = [t for t in ax.get_xticks() if t>=x0 and t<=x1] + [xm] + [xval]
        visible_labels = [l for t, l in zip(ax.get_xticks(), ax.get_xticklabels()) if t>=x0 and t<=x1] + ['%.2f' % xm] + [r'$\lambda_{+}$']

        ax.set_xticks(visible_ticks)
        ax.set_xticklabels(visible_labels)

        return fig, ax
    
    def _mbar(self, z):
        
        return ((self.q_ - 1 - z) + np.sqrt((z - 1 - self.q_ )**2 - 4*self.q_))/2/z

    def _m(self, z):

        return (self.q_**(-1) - 1)/z + self.q_**(-1)*self._mbar(z)

    def _proj(self, x):

        return ((x - 1)**2 - self.q_)/((x - 1)*(x - 1 + self.q_))




class SeparableCovarianceEstimator(_RMTCovarianceEstimator):
    

    def __init__(self, epsilon=1e-4):
        
        self.epsilon = epsilon

    def fit(self, data, y = None):

        if ~np.isin('Dx', data.adata.obs.keys()) or  ~np.isin('Dy', data.adata.var.keys()):
            Dx, Dy = biwhitening(data.adata.X)
                
            data.adata.obs['Dx'] = np.diag(Dx)
            data.adata.var['Dy'] = np.diag(Dy)

        n, p = data.adata.X.shape
        fac = max((data.adata.obs['Dx']**2).median(), 
                  (data.adata.var['Dy']**2).median()**(-1))
        mat = np.array((1/data.adata.obs['Dx']**2))[:,None]@np.array((1/data.adata.var['Dy']**2))[None,:]
        
        grid = np.logspace(np.log10(0.00001), 2*np.log10(np.max(mat)), 1000)

        density, m, mbar, g2 = _compute_esd(fac/data.adata.var['Dy']**2, 
                                            1/(data.adata.obs['Dx']**2*fac), 
                                            q = p/n, 
                                            epsilon = self.epsilon, 
                                            grid = grid, 
                                            wA = None, 
                                            wB = None)
        
        self.density_ = density
        self.grid_ = grid
        self.m_ = m
        self.mbar_ = mbar
        self.q_ = p/n
        self.g2_ = g2

        return self
    
    def plot_eigenspectrum(self, data, ymin=1e-1):

        n, p = data.adata.X.shape
        X = data.adata.X
        if isinstance(X, csr_matrix):
            X = X.toarray()

        S = (X.T@X)/(n - 1)
        eigs, V = np.linalg.eigh(S)

        density = [self.grid_, self.density_]
        
        ind = np.max(np.where(density[1] > ymin))
        xval = density[0][ind]
        plot_edge = 15*xval

        bins = np.logspace(np.log10(min(np.min(density[0]), 
                                        np.min(eigs))),
                           np.log10(np.max(eigs)), 
                           p//6)
        
        vals, bins = np.histogram(eigs,
                                density=True,
                                bins=bins)
        eigbins = bins[:-1]

        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        width = np.diff(bins)

        ax.bar(eigbins, vals, width=width, color='C0', label='data')
        ax.plot(density[0], density[1], 'C3', label='theo.', linestyle='--')
        
        ax.set_xlabel('eigenvalues')
        ax.set_ylabel('density')

        axins = ax.inset_axes([0.58, 0.58, 0.40, 0.40],
                            transform=ax.transAxes)

        score = self.score(data)

        axins.annotate(r'KS=%.3f' % score[0], (0.57, 0.85), xycoords = 'axes fraction', fontsize=5)
        axins.annotate(r'pval=%.3f' % score[1], (0.57, 0.65), xycoords = 'axes fraction', fontsize=5)

        axins.bar(eigbins[eigbins > xval], vals[eigbins > xval], width=width[eigbins > xval], color='C0', label='data')
        axins.plot(density[0][density[0] > xval], density[1][density[0] > xval], 'C3', label='theo.', linestyle='--')
        axins.set_xscale('log')
        axins.set_yscale('log')
        axins.set_ylim(ymin = 0.005*ymin, ymax = 1.4*density[1][density[0] > xval].max())
        axins.set_xlim(xmax = plot_edge, xmin = xval)

        ax.legend(frameon = False, loc='upper left')
        ax.set_xscale('log')
        ax.set_ylim([ymin, 3.3*np.max(vals)])
        ax.set_yscale('log')
        ax.set_xlim(xmin = 0.7*np.min(eigs), xmax = plot_edge)

        x0, x1 = ax.get_xlim()
        visible_ticks = [t for t in ax.get_xticks() if t>=x0 and t<=x1] + [xval]
        visible_labels = [l for t, l in zip(ax.get_xticks(), ax.get_xticklabels()) if t>=x0 and t<=x1] + ['%.2f' % xval]

        ax.set_xticks(visible_ticks)
        ax.set_xticklabels(visible_labels)
        return fig, ax