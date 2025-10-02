from scrna.datasets import *
from scrna.prc import PCA, AManpgSPCA
from scrna.rmt import BiwhitenedCovarianceEstimator
from scrna.metrics import umap_weights
from scrna.utils import experiment_repr, concat_pipelines
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import scipy
from scrna.utils import NumpyEncoder
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--flavor',
                    required=False,
                    type=str,
                    default='seurat', 
                    dest="flavor")

parser.add_argument('-n', '--ngenes',
                    required=False,
                    type=int,
                    default=2000, 
                    dest="ngenes")

parser.add_argument('-t', '--ntrials',
                    required=False,
                    type=int,
                    default=10, 
                    dest="ntrials")

parser.add_argument('-m', '--methods',
                    required=False,
                    choices=['all', 'scvi', 'dca', 'test', 'scvi_default', 'dca_default'],
                    type=str,
                    default="all", 
                    dest="methods")

args = parser.parse_args()
    
print('Running celltype.py with params', args)

        
knn = {'knn (n=%d)' % n_neighbors: BaggingClassifier(KNeighborsClassifier(n_neighbors=n_neighbors, 
                                                                                    weights='uniform'), 
                                                     n_estimators=30, 
                                                     oob_score=True, 
                                                     random_state=0) 
       
                                      for n_neighbors in np.arange(10, 50, 15, dtype=int)
}

knn_umap = {'knn (umap) (n=%d)' % n_neighbors: BaggingClassifier(KNeighborsClassifier(n_neighbors=n_neighbors, 
                                                                                    weights=umap_weights), 
                                                     n_estimators=30, 
                                                     oob_score=True, 
                                                     random_state=0) 
            
                                      for n_neighbors in np.arange(10, 50, 15, dtype=int)
}

knn_dist = {'knn (distance) (n=%d)' % n_neighbors: BaggingClassifier(KNeighborsClassifier(n_neighbors=n_neighbors, 
                                                                                    weights='distance'), 
                                                     n_estimators=30, 
                                                     oob_score=True, 
                                                     random_state=0) 
            
                                      for n_neighbors in np.arange(10, 50, 15, dtype=int)
}

classifiers = {}
for d in (knn, knn_umap, knn_dist): 
    classifiers.update(d)

errors = {}

rng = np.random.default_rng(12)
trials = args.ntrials

for dataset, info in DATASETS.items():
    
    celltype_key = DATASETS[dataset]['celltype_key']
    if celltype_key is None:
        continue
        
    error = []

    for trial in range(trials):

        print('processing dataset', dataset, 'trial', trial)

        large, small, large_pipe, small_pipe = get_subsampled_datasets(dataset, rng=rng)
        
            
        pr_pipeline = get_preprocessing_pipeline(flavor=args.flavor, 
                                                 ngenes=args.ngenes)
            
        small = pr_pipeline.fit_transform(small)
        large = pr_pipeline.transform(large)
            
        # We need to first run one of the spca to first get the number of components
        pipeline = get_pipeline('spca').set_params(red_dim=None)
        _small = pipeline.fit_transform(small.copy())
        cov_est = BiwhitenedCovarianceEstimator().fit(_small)
        n_comps = cov_est.n_comps_
        score = cov_est.score(_small)
        print('score', score[0], 'pvalue', score[1])
        print('n_comps', cov_est.n_comps_)

        
        # Placeholders for DCA since it only runs in an outdated kernel
        if args.methods == 'all':
            methods = {
                'pca': {'pipeline': 'pca',
                        'keys': ['X_pca'],
                        'data': small.copy(),
                        'params': {'n_comps': n_comps}},
                'sclens': {'pipeline': 'sclens',
                           'keys': ['X_sclens'],
                          'data': small.copy(),
                          'params': {}},
                'pca_biwhitened': {'pipeline': 'pca',
                                   'keys': ['X_pca'],
                        'data': small.copy(),
                        'params': {'n_comps': 'auto',
                                   'scaling': BiwhiteningScaler(with_mean=True)}},
                'bipca_denoised': {'pipeline': 'bipca',
                                   'keys': ['X_bipca'],
                          'data': small.copy(),
                          'params': {'counts': True}},
                'bipca_shrinked': {'pipeline': 'bipca',
                                   'keys': ['X_bipca'],
                          'data': small.copy(),
                          'params': {'counts': False}},
                'fista_gs': {'pipeline': 'spca',
                            'keys': ['X_spca'],
                            'data': small.copy(),
                            'params': {'spca': FistaSPCA(verbose=0, tol=1e-4, ortho='gs')}},
                'fista_lowdin': {'pipeline': 'spca',
                            'keys': ['X_spca'],
                            'data': small.copy(),
                            'params': {'spca': FistaSPCA(verbose=0, tol=1e-4, ortho='lowdin')}},
                'sklearn': {'pipeline': 'spca',
                            'keys': ['X_spca'],
                            'data': small.copy(),
                            'params': {'spca': sklearnSPCA(verbose=1, tol=1e-4)}},
                'sklearn_over': {'pipeline': 'spca',
                            'keys': ['X_spca'],
                            'data': small.copy(),
                            'params': {'spca': sklearnSPCA(verbose=1, tol=1e-4),
                                       'scale': 5}},
                'sklearn_counts': {'pipeline': 'spca',
                                   'keys': ['X_spca'],
                                   'data': small.copy(),
                                   'params': {'log': False}},
                'amanpg':  {'pipeline': 'spca',
                            'keys': ['X_spca'],
                            'data': small.copy(),
                            'params': {'spca': AManpgSPCA(tol=1e-4, verbose=0, ridge=10000)}},

                'magic': {'pipeline': 'magic',
                          'keys': ['X_pca'],
                          'data': small.copy(),
                          'params': {'seed': 0, 
                                     'n_comps': n_comps}},
                'large': {'pipeline': 'pca',
                          'keys': ['X_pca'],
                         'data': large.copy(),
                         'params': {'n_comps': n_comps}},
            }
            
            if args.ngenes > 3000:
                methods['amanpg'] = None
                methods['fista_gs'] = None
                methods['fista_lowdin'] = None
                methods['sklearn_counts'] = None
        

        if args.methods == 'test':
            
            methods = {

                'pca': {'pipeline': 'pca',
                        'keys': ['X_pca'],
                        'data': small.copy(),
                        'params': {'n_comps': n_comps}},
             
                'large': {'pipeline': 'pca',
                          'keys': ['X_pca'],
                         'data': large.copy(),
                         'params': {'n_comps': n_comps}}
            }
            
            
            
        if args.methods == 'scvi':
            
            methods = {
                'pca': {'pipeline': 'pca',
                    'keys': ['X_pca'],
                    'data': small.copy(),
                    'params': {'n_comps': n_comps}},
                'scvi': {'pipeline': 'scvi',
                     'keys': ['X_pca', 'X_scVI'],
                     'data': small.copy(),
                     'params': {'nhyper': 50, 
                                'seed': 0,
                                'n_comps': n_comps}}
            }
            
        if args.methods == 'dca':
            
            methods = {
                'pca': {'pipeline': 'pca',
                        'keys': ['X_pca'],
                        'data': small.copy(),
                        'params': {'n_comps': n_comps}},
                'dca': {'pipeline': 'dca',
                        'keys': ['X_pca', 'X_dca'],
                        'data': small.copy(),
                        'params': {'nhyper': 50, 
                               'n_comps': n_comps}}
            }
            
        if args.methods == 'dca_default':
            
            methods = {
                'pca': {'pipeline': 'pca',
                        'keys': ['X_pca'],
                        'data': small.copy(),
                        'params': {'n_comps': n_comps}},
                'dca': {'pipeline': 'dca',
                        'keys': ['X_pca', 'X_dca'],
                        'data': small.copy(),
                        'params': {'nhyper': 0, 
                                   'n_comps': n_comps}}
            }
            
        if args.methods == 'scvi_default':
            
            methods = {
                'pca': {'pipeline': 'pca',
                    'keys': ['X_pca'],
                    'data': small.copy(),
                    'params': {'n_comps': n_comps}},
                'scvi': {'pipeline': 'scvi',
                     'keys': ['X_pca', 'X_scVI'],
                     'data': small.copy(),
                     'params': {'nhyper': 0, 
                                'seed': 0,
                                'n_comps': n_comps}}
            }

        num = np.array([(small.adata.obs[celltype_key] == ct).sum() for ct in small.adata.obs[celltype_key].unique()])
        nums = np.sort(num) -1
        nums = nums[nums > 30]

        _errors = {}
        spec = {}
        
        
        for i, (name, _method) in enumerate(methods.items()):

            if _method is None:
                continue

            print('method', name)

            pipeline = get_pipeline(_method['pipeline'], **_method['params'])
            _data = pipeline.fit_transform(_method['data'])
       
            if name == 'large':
                _data = _data.subset_cells(small.adata.obs_names)
                
                spec['qc_genes:%d'%i] = {
                    'trunk': concat_pipelines([pr_pipeline,
                                               pipeline]),
                    'name_trunk': name,
                    'align': 'pca:to_layer',
                    'links': ['pca:highly_variable->highly_variable',
                              'pca:qc_genes_1->qc_genes_1']
                }
            elif name == 'pca':
                spec['trunk'] = concat_pipelines([small_pipe, 
                                                  pr_pipeline,
                                                  pipeline])
                spec['name_trunk'] = name
            else:
                spec['qc_cells_2:%d'%i] = {
                    'trunk': pipeline,
                    'name_trunk': name
                }
            
            for key in _method['keys']:
                print('computing observables for key', key, 'in method', name)
                _error = {_name: [] for _name in list(classifiers.keys()) + ['silhouette']}

                for n in tqdm(nums[:2]):

                    celltypes = small.adata.obs[celltype_key].unique()[num > n]
                    train_data = _data.get_cells(celltypes, entry=celltype_key)

                    for _name, _class in classifiers.items():

                        _class.fit(train_data.obsm[key], train_data.obs[celltype_key])
                        _error[_name].append(1 - _class.oob_score_)


                _error['silhouette'].append(silhouette_score(_data.adata.obsm[key], 
                                                             _data.adata.obs[celltype_key]))

                _errors[name+'_'+key] = _error
              
        if args.methods == 'all':
            # placeholder for scvi and dca to build the diagram afterwards
            spec['qc_cells_2:%d'%(i+1)] = {
                    'trunk': get_pipeline('scvi'),
                    'name_trunk': 'scvi'
            }
            
            spec['qc_cells_2:%d'%(i+2)] = {
                    'trunk': get_pipeline('dca'),
                    'name_trunk': 'dca'
            }
            
        if trial == 0:  
            experiment_repr(spec, svg_path = 'figures/%s/%s_%s_exp_%s_%d' % (info['name'].lower(), args.methods, dataset, args.flavor, args.ngenes), align_last=False)
            
        error.append(_errors)
        errors[info['name']+'\n('+dataset+')'] = error
        json.dump(errors, open('figures/celltype_%s_%s_%d_%d.json' % (args.methods, args.flavor, args.ngenes, args.ntrials), 'w'), cls=NumpyEncoder)
