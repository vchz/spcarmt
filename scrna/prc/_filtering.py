import numpy as np
import numpy as np
import scanpy as sc
import seaborn as sns
from scanpy.pp._qc import describe_var, describe_obs
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from ..base import SamplesTransformer, FeaturesTransformer

class CellsQualityFilter(SamplesTransformer):

    def __init__(self,
                 min_genes=20,
                 max_genes=None,
                 max_pct_mt=None,
                 min_tot_counts=None,
                 max_tot_counts=None,
                 plot=False):

        self.min_genes = min_genes
        self.max_genes = max_genes
        self.max_pct_mt = max_pct_mt
        self.min_tot_counts = min_tot_counts
        self.max_tot_counts = max_tot_counts
        self.plot = plot

    def _fit(self, data, y=None):

        qc_vars = []
        if self.max_pct_mt is not None:
            qc_vars = ['mt']
        self.metrics_ = describe_obs(data.adata,
                                      qc_vars=qc_vars,
                                      percent_top=None,
                                      log1p=False,
                                      inplace=False)

        return self

    def transform(self, data, y= None):
        data = super().transform(data, y)
        self._fit(data)

        if (self.plot):
            if self.max_pct_mt is not None:
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
                ax1.set_title('before cell filtering')
                sns.scatterplot(self.metrics_, x='total_counts', y='pct_counts_mt', ax=ax1)
            
                ax1.axhline(y = self.max_pct_mt, color = 'r')
                if self.max_tot_counts is not None:
                    ax1.axvline(x = self.max_tot_counts, color = 'r')   
            else:
                fig, ax2 = plt.subplots(nrows=1, ncols=1)
                
            ax2.set_title('before cell filtering')
            sns.scatterplot(self.metrics_, x='total_counts', y='n_genes_by_counts', ax=ax2)
            
            fig.show()
            if self.min_tot_counts is not None:
                ax2.axvline(x = self.min_tot_counts, color = 'r')
            if self.max_tot_counts is not None:
                ax2.axvline(x = self.max_tot_counts, color = 'r')   
            if self.min_genes is not None:
                ax2.axhline(y = self.min_genes, color = 'k')   
            if self.max_genes is not None:
                ax2.axhline(y = self.max_genes, color = 'k')  

        qc_passed_cells = np.ones(data.adata.X.shape[0], dtype=bool)

        if (self.min_genes is not None):
            qc_passed_cells &= (
                self.metrics_.n_genes_by_counts > self.min_genes)

        if (self.max_genes is not None):
            qc_passed_cells &= (
                self.metrics_.n_genes_by_counts < self.max_genes)

        if (self.max_pct_mt is not None):
            qc_passed_cells &= (
                self.metrics_.pct_counts_mt < self.max_pct_mt)

        if (self.min_tot_counts is not None):
            qc_passed_cells &= (
                self.metrics_.total_counts > self.min_tot_counts)

        if (self.max_tot_counts is not None):
            qc_passed_cells &= (
                self.metrics_.total_counts < self.max_tot_counts)

        qc_passed_cells_ = qc_passed_cells[np.where(qc_passed_cells)[0]].index
        
        before = data.adata.shape[0]
        data.subset_cells(qc_passed_cells_)
        
        print('passed cell quality filter,', before, 'before,', data.adata.shape[0], 'remaining')

        return data

# TODO build a filter to simultaneously ensure there are non zero cells and genes
class GenesQualityFilter(FeaturesTransformer):

    def __init__(self,
                 min_pct_cells=1,
                 max_pct_cells=None,
                 min_gene_tot_counts=0,
                 max_gene_tot_counts=None,
                 keep=None,
                 plot=False):

        self.min_pct_cells = min_pct_cells
        self.max_pct_cells = max_pct_cells
        self.min_gene_tot_counts = min_gene_tot_counts
        self.max_gene_tot_counts = max_gene_tot_counts
        self.keep = keep
        self.plot = plot

    def _fit(self, data, y=None):

        self.metrics_ = describe_var(data.adata,
                                     log1p=False,
                                     inplace=False)

        return self

    def fit(self, data, y = None):

        super().fit(data, y)
        self._fit(data)

        qc_passed_genes = np.ones(data.adata.X.shape[1], dtype=bool)
        self.max_cells_ = None
        self.min_cells_ = None
        
        if self.max_pct_cells is not None:
            self.max_cells_ = (self.max_pct_cells/100)*data.adata.X.shape[0]
            print('maximum number of cells', self.max_cells_)

        if self.min_pct_cells is not None:
            self.min_cells_ = (self.min_pct_cells/100)*data.adata.X.shape[0]
            print('minimum number of cells', self.min_cells_)
        if (self.plot):
            fig, ax1 = plt.subplots(nrows=1, ncols=1)
            ax1.set_title('before gene filtering')
            sns.scatterplot(self.metrics_, x='total_counts', y='n_cells_by_counts', ax=ax1)
            ax1.set_yscale('log')
            ax1.set_xscale('log')
             
            if self.min_gene_tot_counts is not None:
                ax1.axvline(x = self.min_gene_tot_counts, color = 'r')
            if self.max_gene_tot_counts is not None:
                ax1.axvline(x = self.max_gene_tot_counts, color = 'r')   
            if self.min_cells_ is not None:
                ax1.axhline(y = self.min_cells_, color = 'k')   
            if self.max_cells_ is not None:
                ax1.axhline(y = self.max_cells_, color = 'k')  

            fig.show()
            
        if (self.min_pct_cells is not None):
            qc_passed_genes &= (
                self.metrics_.n_cells_by_counts > self.min_cells_)

        if (self.max_pct_cells is not None):
            qc_passed_genes &= (
                self.metrics_.n_cells_by_counts < self.max_cells_)

        if (self.min_gene_tot_counts is not None):
            qc_passed_genes &= (
                self.metrics_.total_counts > self.min_gene_tot_counts)

        if (self.max_gene_tot_counts is not None):
            qc_passed_genes &= (
                self.metrics_.total_counts < self.max_gene_tot_counts)

        self.qc_passed_genes_ = qc_passed_genes[np.where(qc_passed_genes)[0]].index
        
        return self

    def transform(self, data, y = None):

        data = super().transform(data, y)
        sub = self.qc_passed_genes_
        if self.keep is not None:
            sub = np.concatenate((self.qc_passed_genes_,self.keep))
            
        before = data.adata.shape[1]
        data = data.subset_genes(sub)
        print('passed gene quality filter,', before, 'before,', data.adata.shape[1], 'remaining')

        return data

class BackgroundRemover(SamplesTransformer):

    def __init__(self):

        robjects.r('''
            install.packages('SoupX')
            library(SoupX)
            
            soupX <- function(data_tab, data_tod, genes, cells, soupx_groups) {
                rownames(data_tab) = genes
                colnames(data_tab) = cells

                data_tab <- as(data_tab, "sparseMatrix")
                data_tod <- as(data_tod, "sparseMatrix")

                sc = SoupChannel(data_tod, data_tab, calcSoupProfile = FALSE)

                soupProf = data.frame(row.names = rownames(data_tab), est = rowSums(data_tab)/sum(data_tab), counts = rowSums(data_tab))
                sc = setSoupProfile(sc, soupProf)
                sc = setClusters(sc, soupx_groups)

                sc  = autoEstCont(sc, doPlot=FALSE)
                out = adjustCounts(sc, roundToInt = TRUE)

                return(out)
            }
        ''')


    def transform(self, data, y = None):

        data = super().transform(data, y)
        adata_pp = data.adata.copy()
        if (data.raw is None):
            data.raw = data.adata.copy()

        sc.pp.normalize_per_cell(adata_pp)
        sc.pp.log1p(adata_pp)
        sc.pp.pca(adata_pp)
        sc.pp.neighbors(adata_pp)
        sc.tl.leiden(adata_pp, key_added="soupx_groups")

        soupx_groups = adata_pp.obs["soupx_groups"]
        cells = data.adata.obs_names
        genes = data.adata.var_names
        data_tab = data.adata.X.T
        data_tod = data.raw.X.T

        del adata_pp

        soupX = robjects.globalenv['soupX']

        out = soupX(data_tab, data_tod, genes, cells, soupx_groups)

        data.adata.X = out.todense().T
        return data

class DoubletFilter(SamplesTransformer):

    def __init__(self, method='scrublet'):
        self.method = method

    def _scrublet_transform(self, data):

        import scrublet as scrub

        X = data.adata.X
        #if isinstance(data.adata.X, csr_matrix):
            #X = X.toarray()
        
        scr = scrub.Scrublet(X, expected_doublet_rate=0.1)
        doublet_scores, predicted_doublets = scr.scrub_doublets(min_counts=2, 
                                                            min_cells=3, 
                                                            min_gene_variability_pctl=85, 
                                                            n_prin_comps=30)

        if predicted_doublets is None:
            threshold_ = 1.1*max(np.max(scr.doublet_scores_obs_),
                                    np.max(scr.doublet_scores_sim_))
            predicted_doublets = scr.call_doublets(threshold = threshold_)

        data.adata.obs['doublet_scores'] = doublet_scores.copy()
        data.adata = data.adata[predicted_doublets != True,:]
        return data

    def _dblfinder_transform(self, data):
        robjects.r('''
            dblFinder <- function(data_mat, droplet_class) {
                set.seed(123)
                sce = scDblFinder(
                    SingleCellExperiment(
                        list(counts=data_mat),
                    ) 
                )
                droplet_class = sce$scDblFinder.class
                return(droplet_class)
            }
        ''')
        dblFinder = robjects.globalenv['dblFinder']
        droplet_class = dblFinder(data.adata.X.T)

        data.adata.obs["scDblFinder_class"] = droplet_class
        data.adata.obs.scDblFinder_class.value_counts()

        return data

    def transform(self, data, y=None):

        data = super().transform(data, y)
        transform_function = {
            'scrublet': self._scrublet_transform,
            'dblfinder': self._dblfinder_transform,
        }[self.method]

        data = transform_function(data)
        return data

class GeneThresholdFilter(SamplesTransformer):

    def __init__(self,
                 variables=None,
                 threshold=3.5):

        if isinstance(variables, str):
            variables = [variables]
        self.variables = variables
        self.threshold = threshold

    def transform(self, data, y = None):

        data = super().transform(data, y)
        genes = np.where((data.adata.var_names.to_numpy()[
                         :, np.newaxis] == self.variables).any(axis=1))[0]
        cells = (data.adata[:, genes].X > self.threshold).any(axis=1)
        data.subset_cells(~cells)

        return data

class GeneListFilter(FeaturesTransformer):
    
    def __init__(self,
                 list_genes = None):
        
        
        self.list_genes = list_genes

        
    def transform(self, data, y = None):
        
        
        if self.list_genes is None:
            raise ValueError(
                'Need a list of genes to use GeneListFilter().transform()'
            )
            
        data = super().transform(data, y)
        data.subset_genes(self.list_genes)
        
        return data
        
class HighlyVariableFilter(FeaturesTransformer):

    def __init__(self,
                 num=2000,
                 flavor='seurat',
                 layer=None,
                 keep=None,
                 batch_key=None,
                 plot=False):

        self.num = num
        self.layer = layer
        self.flavor = flavor
        self.keep = keep
        self.plot = plot
        self.batch_key = batch_key

    def fit(self, data, y=None):

        super().fit(data, y)

        data_pp = data.adata.copy()
        if isinstance(data_pp.X, csr_matrix):
            data_pp.X = data_pp.X.toarray()
            
        hv = sc.pp.highly_variable_genes(data_pp,
                                        batch_key=self.batch_key,
                                        layer=self.layer,
                                        n_top_genes=self.num,
                                        flavor=self.flavor,
                                        inplace=False)
        


        self.highly_variable = data.adata.var_names[hv.highly_variable]

        return self

    def transform(self, data, y=None):

        data = super().transform(data, y)
        sub = self.highly_variable
        if self.keep is not None:
            sub = np.concatenate((self.highly_variable,self.keep))
        data.subset_genes(sub)
        return data
