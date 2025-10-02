from ..data import Data
from ..base import SamplesTransformer, FeaturesTransformer
from ..utils import get_cell_cycle_genes
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix

#        _, counts_per_cell, _ = _normalize_total_helper(
 #           self.adata.X,
 #           exclude_highly_expressed=False,
 #           max_fraction=None,
 #           target_sum=1.0,
 #       )
        
 #       self.adata.obs['n_counts'] = counts_per_cells
 #       if not np.all(counts_per_cell > 0):
 #           print('some cells have zero counts')

class DatasetAligner(SamplesTransformer):

    def __init__(self, 
                 covariates=None, 
                 method='combat_seq',
                 batch_key = None):
        self.covariates = covariates
        self.method = method
        self.batch_key = batch_key

    def _scanpy_combat_transform(self, data):

        sc.pp.combat(data.adata,
                     key=data.batch_key,
                     covariates=self.covariates)
        return data
    
    def _combat_seq_transform(self, data):
        
        from inmoose.pycombat import pycombat_seq

        X = data.adata.X
        was_csr = False
        if isinstance(X, csr_matrix):
            was_csr = True
            X = X.toarray()
        X = X.astype(float)

        batches = data.adata.obs[data.batch_key].values.tolist()
        X = pycombat_seq(X.T, batches)
        if was_csr == True:
            X = csr_matrix(X)

        data.adata.X = X.T
        return data
    
    def transform(self, data, y=None):

        data = super().transform(data, y)
        if (self.batch_key is None):
            self.batch_key = data.batch_key
        
        if (self.batch_key not in data.adata.obs_keys()):
            raise ValueError(
                    'The batch key for integration does not exist in the anndata object'
                )

        transform_function = {
            'scanpy_combat': self._scanpy_combat_transform,
            'combat_seq': self._combat_seq_transform,
        }[self.method]

        return transform_function(data)

class Subsampler(SamplesTransformer):

    def __init__(self,
                 fraction=None,
                 num_cells=None,
                 replace=False,
                 rng=None):
        
        
        self.replace = replace
        self.fraction = fraction
        self.num_cells = num_cells
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

    def transform(self, data, y=None):

        data = super().transform(data, y)
        old_n_obs = data.adata.n_obs

        if self.fraction is not None:
            if self.fraction == 1:
                print('passed resampling, size', old_n_obs)
                return data
            
            new_n_obs = int(self.fraction*old_n_obs)
        elif self.num_cells is not None:
            new_n_obs = min(self.num_cells,old_n_obs)
        else:
            new_n_obs = old_n_obs
        
        obs_indices = self.rng.choice(old_n_obs,
                                      size=new_n_obs,
                                      replace=self.replace)
        data.adata = data.adata[obs_indices,:]
        print('passed resampling, size', data.adata.shape[0])
        return data


class RegressTransformer(FeaturesTransformer):

    def __init__(self, variables=None):

        if variables == 'cell_cycle':
            self.variables = variables
            return
        
        if isinstance(variables, str):
            self.variables = [variables]
        else:
            raise TypeError(
                    'Variable for RegressTransformer should be a string'
                )

    def transform(self, data, y=None):

        data = super().transform(data, y)
        variables = self.variables
        if self.variables == 'cell_cycle':
            s_genes, g2m_genes = get_cell_cycle_genes('./annotations/macosko_cell_cycle_human.txt')
            data = sc.tl.score_genes_cell_cycle(data.adata,
                                 s_genes = s_genes,
                                 g2m_genes = g2m_genes)
            variables = ['S_score', 'G2M_score']

        for var in variables:
            if var not in data.adata.obs_keys():
                raise ValueError(
                    'Variable given absent from anndat object, can\'t regress out'
                )

        sc.pp.regress_out(data.adata,
                          variables)
        return data


class DataToLayer(SamplesTransformer):

    def __init__(self, layer):

        self.layer = layer

    def fit(self, data, y = None):     
        
        super().fit(data, y)
        return self
    
    def transform(self, data, y = None):

        data = super().transform(data, y)
        data.adata.layers[self.layer] = data.adata.X.copy()
        return data
    
class LayerToData(SamplesTransformer):

    def __init__(self, layer):

        self.layer = layer

    def fit(self, data, y = None):     
        
        super().fit(data, y)
        return self
    
    def transform(self, data, y = None):

        data = super().transform(data, y)

        Y = data.adata.layers[self.layer].copy()
        # Sometimes there are issues with old numpy versions
        if (isinstance(data.adata.X, np.ndarray) 
            and isinstance(Y, csr_matrix)):
            data.adata.X = Y.toarray()
        else:
            data.adata.X = Y
            
        return data

class PeaksToGenes(FeaturesTransformer):

    def __init__(self, upstream=5000, 
                 gtf_annot='/mnt/home/vchardes/ceph/annotations/Homo_sapiens.gencode.v32.primary_assembly.annotation.2019.gtf'):

        self.upstream = upstream
        self.gtf_annot = gtf_annot

    def transform(self, data, y = None):
        import episcanpy as esc
        # Peaks not starting with chr are upper cased in the gtf file
        values = data.adata.var_names.values
        for i, name in enumerate(values):
            if not name.split(':')[0].startswith('chr'):
                _name = name.split(':', 1)
                if len(_name) > 1:
                    values[i] = _name[0].upper()+':'+_name[1]
        data.adata.var_names = values

        _data = esc.tl.geneactivity(data.adata, 
                                    self.gtf_annot,
                                    upstream=self.upstream,
                                    annotation=None)
        
        return Data(_data)


