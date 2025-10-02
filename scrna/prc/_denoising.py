import numpy as np
from ..base import SamplesTransformer
import argparse
import json
import os
from hyperopt import hp, space_eval
from torch import device as torch_device
from torch.cuda import is_available as cuda_available
from scipy.sparse import csr_matrix

class MAGIC(SamplesTransformer):

    def __init__(self, genes='all_genes', seed = None):

        self.genes = genes
        self.seed = seed

    def transform(self, data, y=None):
        data = super().transform(data, y)

        try:    
            from magic import MAGIC
        except ModuleNotFoundError:
            raise NameError(
                'Module magic is not installed: it is necessary to use MAGIC()'
            )

        _magic = MAGIC(random_state = self.seed)
        _magic.fit(data.adata, y)
        magic_denoised = _magic.transform(data.adata.X,
                                          genes=self.genes)
        data.adata = magic_denoised.copy()

        return data

class BiPCA(SamplesTransformer):
    
    def __init__(self, counts=True):
        
        self.counts = counts
        
    def transform(self, data, y=None):
        
        data = super().transform(data, y)
        
        try:
            import bipca
            from bipca.experiments import new_svd
            from bipca.utils import make_scipy
        except ModuleNotFoundError:
            raise NameError(
                'error when loading the module bipca'
            )
            
        op = bipca.BiPCA(n_components=-1,seed=42)
        counts = False
        Z = op.fit_transform(data.adata.X.toarray(), counts=self.counts)
        op.write_to_adata(data.adata)
        
        if self.counts == False:
            data.adata.varm['biPCs'] = op.V_Y[:,:op.mp_rank]
            data.adata.obsm['X_bipca'] = bipca.utils.make_scipy(Z)
        else:
            data.adata.X = Z
            U,S,VT = bipca.experiments.new_svd(Z,data.adata.uns["bipca"]['rank'], which=None)
            data.adata.obsm['X_bipca'] = bipca.utils.make_scipy(U*S)
            data.adata.varm['biPCs'] = bipca.utils.make_scipy(VT.T)
        
        return data
    
    
class scLENS(SamplesTransformer):
    
    def transform(self, data, y=None):
        
        data = super().transform(data, y)

        try:
            from scLENS import scLENS
        except ModuleNotFoundError:
            raise NameError(
                'error when loading the module scLENS'
            )
            
        device = torch_device('cuda:0' if cuda_available() else 'cpu')
        sclens = scLENS(device=device)
        X = data.adata.X
        if isinstance(X, csr_matrix):
            X = X.toarray()
            
        sclens.preprocess(X, 
                          min_genes_per_cell=0, 
                          min_cells_per_gene=0)
        data.adata.obsm['X_sclens'] = sclens.fit_transform()
        print('finished sclens, size of dimensionally reduced data', data.adata.obsm['X_sclens'].shape)
        return data
    
class scVI(SamplesTransformer):

    def __init__(self, batch_key=None, nhyper = 0, seed = None, ngpu='auto', tuned_params={}):

        self.batch_key = batch_key
        self.seed = seed
        self.nhyper = nhyper
        self.ngpu = ngpu
        
        if (self.nhyper == 0 
            and tuned_params == {}):
            
            tuned_params = {
                'model': {
                    "n_hidden": 128,
                    "n_latent": 10,
                    "n_layers": 1,
                    "dropout_rate": 0.1,
                    "gene_likelihood": 'zinb',
                },
                'train': {
                    'plan_kwargs': { "lr": 0.001 },
                }
            }
        
        self.tuned_params = tuned_params
        
        
        if (ngpu == 'auto' 
            and 'SLURM_GPUS_ON_NODE' in os.environ):
            self.ngpu = int(os.environ['SLURM_GPUS_ON_NODE'])
            print('scvi: one gpu on node detected, setting ngpu to', self.ngpu)
            
    def transform(self, data, y=None):

        data = super().transform(data, y)
        try:
            import scvi
            from scvi import autotune
            from ray import tune
            
        except ModuleNotFoundError:
            raise NameError(
                'Module scvi is not installed: it is necessary to use scVI()'
            )
        
        model = scvi.model.SCVI
        model.setup_anndata(data.adata, batch_key=self.batch_key)
        
        if self.nhyper > 0:
            scvi_tuner = autotune.ModelTuner(model)

            search_space = {
                "n_hidden": tune.choice([64, 128, 256]),
                "n_latent": tune.choice([5, 10, 15]),
                "n_layers": tune.choice([1, 2, 3]),
                "lr": tune.loguniform(1e-4, 1e-2),
                "dropout_rate": tune.choice([0.1, 0.2]),
                "gene_likelihood": tune.choice(['nb', 'zinb']),
            }

            results = scvi_tuner.fit(
                data.adata,
                seed=self.seed,
                metric="validation_loss",
                num_samples=self.nhyper,
                search_space=search_space,
                max_epochs=800,
                resources={"gpu":self.ngpu},
            )
            print('scvi hyperparameter searched completed,', results.model_kwargs, results.train_kwargs)
            self.tuned_params = {'model': results.model_kwargs, 'train': results.train_kwargs}
        
        vae = scvi.model.SCVI(data.adata, **self.tuned_params['model'])
        vae.train(**self.tuned_params['train'])
        data.adata.obsm["X_scVI"] = vae.get_latent_representation()
        data.adata.X = vae.get_normalized_expression(library_size=1e4)

        return data
    

class DCA(SamplesTransformer):

    def __init__(self, threads=16, nhyper=0, tuned_params={}):

        self.threads = threads
        self.nhyper = nhyper
        if (self.nhyper == 0
            and tuned_params == {}):
            tuned_params = {
                "data": {
                    "norm_input_log": True,
                    "norm_input_zeromean": True,
                    "norm_input_sf": True,
                },
                "model": {
                    "hidden_size": (64, 32, 64),
                    "activation": "relu",
                    "batchnorm": True,
                    "dropout": 0,
                    "aetype": "nb-conddisp",
                    "lr": None,
                    "l1_enc_coef": 0.0,
                    "ridge": 0.0,
                    "input_dropout": 0.0,
                },
                "fit": {
                    "epochs": 300,
                },
            }

            
        self.tuned_params = tuned_params

    
    def transform(self, data, y=None):

        data = super().transform(data, y)
        try:
            from dca.api import dca
            from dca.hyper import hyper
            import tensorflow as tf
            from keras import backend as K

            tf.compat.v1.disable_eager_execution()

        except ModuleNotFoundError:
            raise NameError(
                'Module DCA is not installed: it is necessary to use DCA()'
            )
        
        _args = {'input': data.adata.copy(), 
                 'transpose': False, 
                 'hyperepoch': 800,
                 'debug': False,
                 'outputdir': './tmp',
                 'hypern': self.nhyper}

        args = argparse.Namespace()
        for key, val in _args.items():
            args.__setattr__(key, val)

        # Dictionary as defined in the hyper function of DCA
        if self.nhyper > 0:
            hyper_params_dca = {
                "data": {
                    "norm_input_log": hp.choice('d_norm_log', (True, False)),
                    "norm_input_zeromean": hp.choice('d_norm_zeromean', (True, False)),
                    "norm_input_sf": hp.choice('d_norm_sf', (True, False)),
                    },
                "model": {
                    "lr": hp.loguniform("m_lr", np.log(1e-3), np.log(1e-2)),
                    "ridge": hp.loguniform("m_ridge", np.log(1e-7), np.log(1e-1)),
                    "l1_enc_coef": hp.loguniform("m_l1_enc_coef", np.log(1e-7), np.log(1e-1)),
                    "hidden_size": hp.choice("m_hiddensize", ((64,32,64), (32,16,32),
                                                                (64,64), (32,32), (16,16),
                                                                (16,), (32,), (64,), (128,))),
                    "activation": hp.choice("m_activation", ('relu', 'selu', 'elu',
                                                                'PReLU', 'linear', 'LeakyReLU')),
                    "aetype": hp.choice("m_aetype", ('zinb', 'zinb-conddisp')),
                    "batchnorm": hp.choice("m_batchnorm", (True, False)),
                    "dropout": hp.uniform("m_do", 0, 0.7),
                    "input_dropout": hp.uniform("m_input_do", 0, 0.8),
                    },
                "fit": {
                    "epochs": args.hyperepoch
                    }
            }


            hyper(args)
            with open("./tmp/hyperopt_results/best.json") as json_file:
                best = json.load(json_file)
            dict_params = space_eval(hyper_params_dca, best)
            print('params optimized', dict_params)
            self.tuned_params = dict_params
        else:
            dict_params = self.tuned_params
            
        # First latent mode will write lower dimensional representation
        # Second it will overwrite adata.X with denoised value (no library size correction)
        for mode in ['latent', 'denoise']:
            print('Running DCA in mode:', mode)
            dca_denoised = dca(data.adata.copy(),
                                mode = mode,
                                threads=self.threads,
                                copy=True,
                                log1p=dict_params['data']['norm_input_log'],
                                scale=dict_params['data']['norm_input_zeromean'],
                                normalize_per_cell=dict_params['data']['norm_input_sf'],
                                hidden_size=dict_params['model']['hidden_size'],
                                activation=dict_params['model']['activation'],
                                batchnorm=dict_params['model']['batchnorm'],
                                hidden_dropout=dict_params['model']['dropout'],
                                ae_type=dict_params['model']['aetype'],
                                learning_rate=dict_params['model']['lr'],
                                epochs=dict_params['fit']['epochs'],
                                network_kwds={'l1_enc_coef': dict_params['model']['l1_enc_coef'],
                                              'ridge': dict_params['model']['ridge'],
                                              'input_dropout': dict_params['model']['input_dropout']},
                                verbose=True)
            if mode == 'latent':
                data.adata.obsm['X_dca'] = dca_denoised.obsm['X_dca']

        data.adata.X = dca_denoised.X
        K.clear_session()
        return data