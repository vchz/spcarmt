
import numpy as np
import scanpy as sc
import rpy2.robjects as robjects
from ..base import SamplesTransformer
from ..rmt._covariance import biwhitening, biwhitening_fast, _get_scaling_marchenko_pastur
from sklearn.preprocessing import scale, Normalizer
from scipy.sparse import csr_matrix

#TODO we should have a Seurat/R version implemented for each transformer
#TODO Maybe not - SamplesTransformer should have a flag use_stored which will not retransform the data but only use the transformation stored in the anndata
# Basically samples transformers  fit in the anndata
# features transformers fit in the transformer object
#TODO SamplesTransformer are simply stateless transformers (see sklearn), with a tag requires_fit=False
class BiwhiteningScaler(SamplesTransformer):

    def __init__(self, max_iter=2000, tol=1e-6, with_mean=True):

        self.max_iter = max_iter
        self.tol = tol
        self.with_mean = with_mean
        
    def transform(self, data, y = None):

        self.fit(data, y)
        data = super().transform(data, y)
    
        D1, D2 = biwhitening_fast(
            data.adata.X, max_iter=self.max_iter, tol=self.tol)

       
        data.adata.X = D1@data.adata.X@D2

        if self.with_mean:
            data.adata.X -= data.adata.X.mean(axis=0)
            if isinstance(data.adata.X, np.matrix):
                data.adata.X = np.asarray(data.adata.X)

        data.adata.obs['D1'] = D1.diagonal()
        data.adata.var['D2'] = D2.diagonal()
            

        return data

class BiPCAScaler(SamplesTransformer):

    def __init__(self, with_mean=True):

        self.with_mean = with_mean


    def transform(self, data, y=None):
    
        data = super().transform(data, y)
    
        try:
            import bipca
        except ModuleNotFoundError:
            raise NameError(
                'Module bipca is not installed: it is necessary to use BiPCAScaler'
            )

        op = bipca.BiPCA(n_components=-1,seed=42) # get the BiPCA operator, here n_components=-1 is to do full SVD
        op.fit(data.adata.X.toarray())
        op.get_plotting_spectrum()
        op.write_to_adata(data.adata)

        D1 = np.diag(op.left_biwhite)
        D2 = np.diag(op.right_biwhite)
        data.adata.obs['D1'] = op.left_biwhite
        data.adata.var['D2'] = op.right_biwhite

        data.adata.X = D1@data.adata.X@D2
        
        return data

class ComputeSizeFactors(SamplesTransformer):
    
    def __init__(self, obs='size_factor', method='total', tot_counts=1e4):

        self. method = 'total'
        if method in ['total', 'scran']:
            self.method = method
        
        self.tot_counts = tot_counts
        self.obs = obs
        
    def _scran_transform(self, data):

        robjects.r('''
            scranNorm <- function(data_mat, input_groups, size_factors) {
                size_factors = calculateSumFactors(data_mat, clusters=input_groups, min.mean=0.1)
                return(size_factors)
            }
        ''')
        adata_pp = data.adata.copy()
        sc.pp.normalize_total(adata_pp,
                          target_sum=self.tot_counts)

        sc.pp.log1p(adata_pp)
        sc.pp.pca(adata_pp, n_comps=15)
        sc.pp.neighbors(adata_pp)
        sc.tl.louvain(adata_pp, key_added='groups', resolution=0.5)

        input_groups = adata_pp.obs['groups']
        data_mat = data.adata.X.T

        del adata_pp

        scranNorm = robjects.globalenv['scranNorm']
        size_factors = scranNorm(data_mat, input_groups)

        data.adata.obs[self.obs] = size_factors.values

        return data

    def _total_transform(self, data):

        dat = sc.pp.normalize_total(data.adata,
                              target_sum=self.tot_counts,
                              inplace=False)
            
        # Careful in older version of scanpy 'norm_factor' is not divided by the target_sum
        if (dat['norm_factor'] == np.sum(data.adata.X, axis = 1).squeeze()).all():
            print('scanpy norm_factor are count per cells')
            dat['norm_factor'] /= self.tot_counts
        
        data.adata.obs[self.obs] = dat['norm_factor']
        return data
    
    
    def transform(self, data, y=None):
        
        data = super().transform(data, y)
        
        transform_function = {
            'total': self._total_transform,
            'scran': self._scran_transform,
        }[self.method]

        data = transform_function(data)
        return data
    
    
class SizeNormalizer(SamplesTransformer):

    def __init__(self, use_obs='size_factor'):

        self.use_obs = use_obs

    def transform(self, data, y=None):

        data = super().transform(data, y)
        if self.use_obs not in data.adata.obs.columns:
             data = ComputeSizeFactors().fit_transform(data)
        
        data.adata.X /= np.array(data.adata.obs[self.use_obs])[:,None]
        return data



class LogNormalizer(SamplesTransformer):

    def transform(self, data, y=None):

        data = super().transform(data, y)
        sc.pp.log1p(data.adata)
        return data



# TODO: is it a samples or features transformer ?
# In practice it should be a features transformer if whitening on the right
# In this case this is the classical scaler StandardScaler
# Maybe we should be able to choose whether or not to fit the SamplesTransformer
class WhiteningScaler(SamplesTransformer):

    def __init__(self, with_mean=True, side='right'):
        self.with_mean = with_mean
        self.side = side

    def transform(self, data, y=None):

        data = super().transform(data, y)
        axis = {
            'left': 1,
            'right': 0,
        }[self.side]

        data.adata.X = scale(data.adata.to_df(),
                             axis=axis,
                             with_mean=self.with_mean,
                             copy=False)
        alpha = _get_scaling_marchenko_pastur(data.adata.X)
        data.adata.X /= np.sqrt(alpha)
        return data

class RightLeftWhiteningScaler(SamplesTransformer):

    def __init__(self, with_mean=True):
        self.with_mean = with_mean
        self.left = WhiteningScaler(with_mean=with_mean, side='left')
        self.right = WhiteningScaler(with_mean=with_mean, side='right')

    def transform(self, data, y=None):


        data = super().transform(data, y)
        data = self.right.fit_transform(data)
        data = self.left.fit_transform(data)
        alpha = _get_scaling_marchenko_pastur(data.adata.X)
        data.adata.X /= np.sqrt(alpha)

        return data


class RightWhiteningLeftNormalizerScaler(SamplesTransformer):

    def __init__(self, with_mean=True, norm = 'l2'):
        self.with_mean = with_mean
        self.norm = norm
        self.right = WhiteningScaler(with_mean=with_mean, side='right')
        self.left = Normalizer(norm=norm)

    def transform(self, data, y=None):

        data = super().transform(data, y)
        data = self.right.fit_transform(data)
        _X = self.left.fit_transform(data.adata.X)
        alpha = _get_scaling_marchenko_pastur(_X)
        data.adata.X = _X/np.sqrt(alpha)

        return data
    