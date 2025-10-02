from scrna.datasets import *
from scrna.prc import PCA, FistaSPCA, sklearnSPCA, AManpgSPCA, GpowerSPCA
from scrna.metrics import sin2_subspaces
from scrna.rmt import BiwhitenedCovarianceEstimator
import numpy as np
from tqdm import tqdm
from scrna.utils import NumpyEncoder
import json
import argparse


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

args = parser.parse_args()
print('Running scan.py with params', args)

pipelines = {
    'gpower': {'pipeline':'spca',
               'params':{'spca': GpowerSPCA(tol=1e-4, verbose=1),
                         'scaling': BiwhiteningScaler(with_mean=True)}},
    'sklearn': {'pipeline':'spca',
                'params':{'spca': sklearnSPCA(tol=1e-4, verbose=1),
                         'scaling': BiwhiteningScaler(with_mean=True)}},
    'amanpg (0)': {'pipeline':'spca',
                   'params':{'spca': AManpgSPCA(tol=1e-4, verbose=0, ridge=0),
                             'scaling': BiwhiteningScaler(with_mean=True)}},
    'amanpg (10000)': {'pipeline':'spca',
                        'params':{'spca': AManpgSPCA(tol=1e-4, verbose=0, ridge=10000),
                                  'scaling': BiwhiteningScaler(with_mean=True)}},
    'fista (gram-schmidt)': {'pipeline':'spca',
              'params':{'spca': FistaSPCA(tol=1e-4, verbose=0),
                        'scaling': BiwhiteningScaler(with_mean=True)}},
    'fista (lowdin)': {'pipeline':'spca',
              'params':{'spca': FistaSPCA(tol=1e-4, verbose=0, ortho='lowdin'),
                        'scaling': BiwhiteningScaler(with_mean=True)}},
    'sklearn (whitening)': {'pipeline':'spca',
                            'params':{'spca': sklearnSPCA(tol=1e-4, verbose=1),
                                      'scaling': WhiteningScaler(with_mean=True)}}
}

dists = {}
facs = np.logspace(np.log10(0.01),np.log10(1.8), 20)

rng = np.random.default_rng(12)

for dataset, info in DATASETS.items():
        
    print('processing dataset', dataset)

    _dists = {} 

    _large, _small, _, _ = get_subsampled_datasets(dataset, rng=rng)
    pr_pipeline = get_preprocessing_pipeline(flavor=args.flavor, 
                                             ngenes=args.ngenes)

    _small = pr_pipeline.fit_transform(_small)
    _large = pr_pipeline.transform(_large)
    
    for (name, pipe) in pipelines.items():
        print('pipeline', name)

        small = get_pipeline(pipe['pipeline'], **pipe['params']).fit_transform(_small.copy())
        n_comps = small.adata.uns['spca']['n_comps']
        opt_penalty = small.adata.uns['spca']['opt_penalty']
        
        small = PCA(n_comps=n_comps).fit_transform(small)

        large_all = get_pipeline('pca', scaling = BiwhiteningScaler(with_mean=True)).fit_transform(_large.copy())
        large_all = large_all.subset_cells(small.adata.obs_names)


        dists_spca = []
        dists_pca = []
        for fac in tqdm(facs):

            spca = pipe['params']['spca']
            spca.penalty = fac*opt_penalty
            spca.n_comps = n_comps
            spca.fit_transform(small)
            
            proj = sin2_subspaces(small.adata, large_all.adata, key_x='sPCs', key_y='PCs')
            dists_spca.append(proj.sum()/proj.shape[0])
            print('dim proj spca', proj.shape[0])
            
            proj_pca = sin2_subspaces(small.adata, large_all.adata, key_x='PCs', key_y='PCs')
            dists_pca.append(proj_pca.sum()/proj_pca.shape[0])
            print('dim proj pca', proj_pca.shape[0])
        
        gain = (np.array(dists_spca) - np.array(dists_pca))/np.array(dists_pca)
        print(gain)
        
        _dists[name] = {'data': gain,
                        'penalty': opt_penalty}

        dists[info['name']+'\n('+dataset+')'] = _dists
        #json.dump(dists, open('figures/scan_%s_%d_b.json' % (args.flavor, args.ngenes), 'w'), cls=NumpyEncoder)