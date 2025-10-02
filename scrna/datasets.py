from .prc import *
from .data import load
from sklearn.pipeline import Pipeline
from functools import partial
from sklearn.base import clone

ROOT = '/mnt/home/vchardes/ceph/datasets/'
METHODS = ['spca', 'bipca', 'sclens', 'pca', 'scvi', 'magic', 'dca']
DATASETS = {
    '10xchromiumv1': {'file': '10XChromiumV1_Zheng17_68k_PBMC',
                'filters': [CellsQualityFilter(max_pct_mt=6, 
                                               min_genes=200, 
                                               max_genes=2000, 
                                               max_tot_counts=6500),
                            GenesQualityFilter(min_pct_cells=0.7, 
                                               max_gene_tot_counts=1e5)],
                'celltype_key': 'cell_types',
                'name': 'Zheng2017'},
    'multiome': {'file': 'Multiome_GSE194122_luecken_neurips_2021',
                    'filters': [CellsQualityFilter(max_pct_mt=10, 
                                                min_genes=300,
                                                max_genes=4500, 
                                                max_tot_counts=12000),
                             GenesQualityFilter(min_pct_cells=0.7, 
                                                max_gene_tot_counts=1e5)],
                    'celltype_key': 'cell_type',
                    'name': 'Luecken2021'},

    'indrops':  {'file': 'inDrops_GSE102827_10.1038_s41593-018-0112-6',
                     'filters': [CellsQualityFilter(max_pct_mt=6, 
                                                    min_genes=500, 
                                                    max_genes=4500, 
                                                    max_tot_counts=12000),
                                 GenesQualityFilter(min_pct_cells=0.7, 
                                                    max_gene_tot_counts=1e5)],
                     'celltype_key': None,#'celltype',
                     'name': 'Hrvatin2016'},

    'smart-seq3xpress': {'file': 'Smart-Seq3xpress_E-MTAB-11452_10.1038_s41587-022-01311-4',
                           'filters': [CellsQualityFilter(max_pct_mt=10, 
                                                          min_genes=200, 
                                                          max_genes=6000, 
                                                          max_tot_counts=30000),
                                       GenesQualityFilter(min_pct_cells=0.7, 
                                                          max_gene_tot_counts=1e5)],
                           'celltype_key': None,
                           'name': 'Jensen2022'},

    'drop-seq': {'file': 'Drop-Seq_GSE63472_10.1016_j.cell.2015.05.002',
                    'filters': [CellsQualityFilter(max_pct_mt=6, 
                                                   min_genes=200, 
                                                   max_genes=5000, 
                                                   max_tot_counts=12000),
                                GenesQualityFilter(min_pct_cells=0.7, 
                                                   max_gene_tot_counts=1e5)],
                    'celltype_key': None,
                    'name': 'Macosko2015'},

    'cite-seq': {'file': 'CITE-Seq_GSE128639_10.1016_j.cell.2019.05.031',
                   'filters': [CellsQualityFilter(max_pct_mt=10, 
                                                  min_genes=300,
                                                  max_genes=5000, 
                                                  max_tot_counts=30000),
                                GenesQualityFilter(min_pct_cells=0.7, 
                                                   max_gene_tot_counts=1e5)],
                   'celltype_key': 'celltype.l2',
                   'name': 'Stuart2019'},
    
    '10xchromiumv3': {'file': '10XChromiumV3_GSE111976_10.1038_s41591-020-1040-z',
                 'filters': [CellsQualityFilter(max_pct_mt=30, 
                                                min_genes=500, 
                                                max_genes=7500, 
                                                max_tot_counts=80000),
                             GenesQualityFilter(min_pct_cells=0.7, 
                                                max_gene_tot_counts=2e5)],
                 'celltype_key': None,
                 'name': 'Wang2020'}
    }


def get_subsampled_datasets(dataset, n_large = 40000, n_small = 3000, rng = None):

    if not dataset in DATASETS.keys():
        raise ValueError(
            'dataset %s not in the list of default datasets' % dataset
        )
    
    metadata = DATASETS[dataset]
    file = ROOT+metadata['file']
    filters = metadata['filters']
    key = metadata['celltype_key']
    
    data = load(file, dense=False)
    if key is not None:
        data = data.subset_cells(data.adata.obs_names[~data.adata.obs[key].isna()])

    large_pipe = Pipeline(steps=[('qc_cells', filters[0]),
                                 ('resampling', Subsampler(num_cells=n_large, 
                                                           rng = rng)),
                                 ('qc_genes', filters[1])], verbose = True)
    
    
    large = large_pipe.fit_transform(data)
    subsampler = Subsampler(num_cells=n_small, 
                          rng = rng)
    
    small = subsampler.fit_transform(large.copy())
    small_pipe = clone(large_pipe)
    small_pipe.steps.append(('resampling_2', subsampler))
    
    return large, small, large_pipe, small_pipe
    
def get_preprocessing_pipeline(flavor='seurat', ngenes=2000):

    pipeline = Pipeline(steps=[('to_layer', DataToLayer(layer='counts')),
                               ('qc_genes_1', GenesQualityFilter(min_pct_cells=0.7)),
                               ('qc_cells_1', CellsQualityFilter(min_genes=20)),
                               ('size_factors', ComputeSizeFactors()),
                               ('pr_norm', SizeNormalizer()),
                               ('pr_log', LogNormalizer()),
                               ('highly_variable', HighlyVariableFilter(flavor='seurat', 
                                                                        num=ngenes)),
                               ('to_data', LayerToData(layer='counts')),
                               ('qc_cells_2', CellsQualityFilter(min_genes=5))], verbose=True)


    if flavor == 'seurat_v3':
        pipeline.set_params(highly_variable__flavor='seurat_v3',
                            pr_norm=None,
                            pr_log=None)
        
    return pipeline

def get_pipeline(method, **params):

    if not method in METHODS:
        raise ValueError(
            'method %s not in the list of default methods' % method
        )

    
    funcs = {
        'spca': _get_spca_pipeline,
        'bipca': _get_bipca_pipeline,
        'sclens': _get_sclens_pipeline,
        'pca': _get_pca_pipeline,
        'scvi': _get_scvi_pipeline,
        'dca': _get_dca_pipeline,
        'magic': _get_magic_pipeline
    }

    return funcs[method](**params)

def _get_spca_pipeline(spca=None, scaling=None, log=True, scale=0.6):

    if spca is None:
        spca = sklearnSPCA(tol=1e-4, verbose=1)
        
    if scaling is None:
        scaling = BiwhiteningScaler(with_mean=True)
        
    pipeline =  Pipeline(steps=[('norm', SizeNormalizer(use_obs='size_factor')),
                                ('log', LogNormalizer()),
                                ('scaling', scaling),
                                ('red_dim', AdaptiveSPCA(spca=spca, scale=scale))], verbose=True)
    
    if log == False:
        pipeline.set_params(log = None)
        pipeline.set_params(norm = None)
        
    return pipeline
                                

def _get_bipca_pipeline(counts=False):
    
    # BiPCA takes care of normalization (biwhitening of counts, then shrinking, then library size)
    pipeline =  Pipeline(steps=[('bipca', BiPCA(counts=counts))], verbose=True)
    
    return pipeline

def _get_sclens_pipeline():
    
    # scLENS takes care of normalization (l1, z-score, l2, log)
    pipeline =  Pipeline(steps=[('sclens', scLENS())], verbose=True)
    
    return pipeline
    
def _get_pca_pipeline(n_comps='auto', scaling=None):

    if scaling is None:
        scaling = WhiteningScaler(with_mean=True)
        
    pipeline = Pipeline(steps=[('norm_', SizeNormalizer(use_obs='size_factor')),
                               ('log', LogNormalizer()),
                               ('scaling', scaling),
                               ('red_dim', PCA(n_comps=n_comps))], verbose=True)
    return pipeline


def _get_scvi_pipeline(n_comps='auto', nhyper = 0, seed=0):

    pipeline = Pipeline(steps=[('scvi', scVI(nhyper=nhyper, 
                                             seed=seed)),
                               ('log', LogNormalizer()),
                               ('scaling', WhiteningScaler(with_mean=True)),
                               ('red_dim', PCA(n_comps=n_comps))], verbose=True)
    return pipeline

def _get_magic_pipeline(n_comps='auto', seed=0):

    pipeline = Pipeline(steps=[('norm', SizeNormalizer(use_obs='size_factor')),
                               ('log', LogNormalizer()),
                               ('magic', MAGIC(seed=seed)),
                               ('scaling', WhiteningScaler(with_mean=True)),
                               ('red_dim', PCA(n_comps=n_comps))], verbose=True)
    
    return pipeline

def _get_dca_pipeline(n_comps='auto', nhyper=0):

    pipeline = Pipeline(steps=[('dca', DCA(nhyper=nhyper)),
                               ('norm', SizeNormalizer(use_obs='size_factor')),
                               ('log', LogNormalizer()),
                               ('scaling', WhiteningScaler(with_mean=True)),
                               ('red_dim', PCA(n_comps=n_comps))], verbose=True)
    return pipeline
