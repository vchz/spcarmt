import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import os
import rpy2.robjects as robjects
from scipy.sparse import csr_matrix
import re

SUFFIXES = ['.txt', '.csv', '.tsv', '.mtx', '.rds', '.h5', '.h5ad', '.h5ad', '.h5seurat']


# TODO: should inherit from AnnData to make everything simpler (and get rid of call self.adata.X)
# Will need to overload copy in particular, and init, by calling the super version
# TODO: rename the show_batch by something like show_by_key
# Overall, keep this class minimal. Any additional variable (metadata etc) should in fact be stored in self.uns
# Except for the visualization methods, no other methods. The metrics should be kept as metrics functions (create a metrics.py file)
class Data:

    def __init__(self, adata,
                 metadata={},
                 dense=False,
                 batch_key=None):

        if isinstance(adata, np.ndarray):
            adata = ad.AnnData(adata)
            
        self.adata = adata
        if(dense and isinstance(adata.X, csr_matrix)):
            self.adata.X = self.adata.X.toarray()
        if (not dense and not isinstance(adata.X, csr_matrix)):
            self.adata.X = csr_matrix(adata.X)

        self.adata.var_names = [name.lower() for name in adata.var_names]
        self.adata.var_names_make_unique()
        self.adata.obs_names_make_unique()
        
        self.adata.var['mt'] = self.adata.var_names.str.startswith('mt-')
        
        self.batch_key = batch_key
        self.metadata = metadata

        if ((batch_key is not None) 
            and (batch_key not in self.metadata.keys())):
            self.metadata[batch_key] = None

    def copy(self):

        new = Data(self.adata.copy(),
                   metadata=self.metadata,
                   batch_key=self.batch_key)
        return new

    def subset_genes(self, subset, entry = None):

        self.adata = self.get_genes(subset, entry)
        return self

    def get_genes(self, subset, entry = None):
            
        if entry is None:
            check = self.adata.var_names
        else:
            check = self.adata.var[entry]

        array = np.isin(check, subset)
        return self.adata[:, array]

    def subset_genes_by(self, obs, value):

        comp = self.adata.var[obs]
        sub = np.isin(comp, value)
        names = self.adata.var_names[sub]
        self.subset_genes(names)
        return self

    def subset_cells(self, subset, entry = None):

        self.adata = self.get_cells(subset, entry)
        return self

    def get_cells(self, subset, entry = None):

        if entry is None:
            check = self.adata.obs_names
        else:
            check = self.adata.obs[entry]

        array = np.isin(check, subset)
        return self.adata[array, :]
 
    def subset_cells_by(self, obs, value):

        comp = self.adata.obs[obs]
        sub = np.isin(comp, value)
        names = self.adata.obs_names[sub]
        self.subset_cells(names)
        return self

    def _show_batch(self, func='pca', key='_batch', color = 'r', toplot = None, ncols=2, equal=True,
            **params):

        dataset = self.split(key)
        l = len(dataset)
        if toplot is not None:
            l = len(toplot)
        nrows = -(-l//ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_size_inches(15, nrows * 5)

        axes = axes.ravel()
        j = 0 
        for i, _data in enumerate(dataset):
            
            if ((toplot is None) or (np.unique(_data.adata.obs[key]) in toplot)):
                self._show(func=func,
                        show=False, 
                        ax=axes[j], **params)
                param = 'na_color'
                if func == 'scatter':
                    param = 'color'
                _params = params.copy()
                _params.update({param: color})
                _data._show(func=func,
                    show=False, 
                    ax=axes[j], 
                    **_params)
                    
                axes[j].set_title(np.unique(_data.adata.obs[key]))
                if equal:
                    axes[j].set_aspect('equal')
                    xlim = axes[j].get_xlim()
                    ylim = axes[j].get_ylim()
                    lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
                    axes[j].set_xlim(lims)
                    axes[j].set_ylim(lims)
                
                axes[j].margins(0.05)
                
                j+=1

        fig.show()
        return fig, axes

    def _show(self, func='umap', **params):

        show_function = {
            'umap': sc.pl.umap,
            'pca': sc.pl.pca,
            'tsne': sc.pl.tsne,
            'scatter': sc.pl.scatter,
        }[func]
        if 'size' not in params.keys():
            params['size'] = 7


        return show_function(self.adata, **params) 

    def split(self, attr):
    
        if self.batch_key is None:
            self.batch_key = '_batch'
            self.metadata[self.batch_key] = None

        samples = []
        for val in np.unique(self.adata.obs[attr].to_numpy()):
            sample = self.adata[self.adata.obs[attr] == val].copy()
            _metadata = self.metadata.copy()
            _metadata[self.batch_key] = val
            _data = Data(sample,
                         metadata=_metadata,
                         batch_key=self.batch_key)
            samples.append(_data)

        return Dataset(samples)

    def __getattr__(self, attr):
            
        show_functions = ['umap', 'tsne', 'pca', 'scatter']
        if attr in show_functions:
            def wrapper(*args, **kwargs):
                return self._show(func=attr, **kwargs)
            return wrapper
        
        if attr in [x+'_batch' for x in show_functions]:
            func = attr.rsplit('_',1)[0]
            def wrapper(*args, **kwargs):
                return self._show_batch(func=func, **kwargs)
            return wrapper
        
        return super().__getattribute__(attr)

    def __add__(self, x):

        if (isinstance(x, Data)):
            samples = [self, x]
            return Dataset(samples)

        else:
            return x.__add__(self)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

# This class is not exactly necessary, it reduces to the concat and groupby functions
# TODO: maybe get rid of it ??duplicated
# Maybe concat should be a standalone function
class Dataset(np.ndarray):

    # Concatenates along the key batch_key the data
    def concat(self, batch_key='_batch'):

        metadata = {}
        adatas = {}
        
        for i, sample in enumerate(self):
            metadata.update({i: sample.metadata})
            key = None
            if batch_key in sample.metadata.keys():
                key = sample.metadata[batch_key]
            if key is None:
                key = i

            adatas.update(
                {key: sample.adata})

        adata = ad.concat(adatas,
                          label=batch_key,
                          index_unique="_",
                          join='outer',
                          fill_value=0)

        all_var = [x.var for x in adatas.values()]
        all_var = pd.concat(all_var, join="outer")
        all_var = all_var[~all_var.duplicated()]
        all_var = all_var[~all_var.index.duplicated()]
        adata.var = all_var.loc[adata.var_names]

        return Data(adata,
                    batch_key=batch_key,
                    metadata=metadata)

    def __new__(cls, input_array, info=None):

        obj = np.asarray(input_array).view(cls)
        return obj

    def groupby(self, attr):

        lst = []
        values = np.unique(self.get(attr))

        for val in values:

            lst.append(self[self.get(attr) == val])

        return np.array(lst)

    def get(self, attr):

        value = []
        for sample in self:
            if (attr in sample.metadata):
                value.append(sample.metadata[attr])
            else:
                value.append(None)
        return np.array(value)

    def __add__(self, x):

        if (isinstance(x, Dataset)):
            return np.append(self, x)
        else:
            return super().__add__(self, x)

def _load_file(f,
              sep=None, 
              dense=False):

    robjects.r('''
        read_rds <- function(file) {
            out = readRDS(file)
            newList <- list("data" = out, "genes" = rownames(out), "cells" = colnames(out))
            return(newList)
        }
    ''')

    def _load_rds(file):

        readRDS = robjects.globalenv['read_rds']
        data = readRDS(file)
                
        df = pd.DataFrame.sparse.from_spmatrix(data[0], index=data[1], columns=data[2])
        _data = ad.AnnData(df)
        return _data
    
    def _read(file):

        if file.split('.')[-1] == 'rds':
   
            try:
                _data = _load_rds(file)
            
            except Exception as e:
                print('couldn\'t read rds file %s with message %s' % (file, repr(e)))
                print('trying again, sometimes it works')
                try:
                    _data = _load_rds(file)
                except Exception as e:
                    print('again couldn\'t read rds file %s with message %s' % (file, repr(e)))
                    raise FileNotFoundError(
                        'Either there is a problem with the RDS file, or this is a problem of IPython. Try rerunning the cell'
                    )
            

        else:
            print('reading file', file)
            if file.endswith('.matrix.h5'):
                print('h5 file, loading with read_10x_h5')
                try:
                    _data = sc.read_10x_h5(file, gex_only = False).T
                except:
                    print('failed, file was not formatted by 10x, fallback on pandas method')
                    _data = ad.AnnData(pd.read_hdf(file))
            else:
                _data = sc.read(file, delimiter=sep, first_column_names=True, cache=True).T

        templates = ['aaacc', 'tttgt']
        if (np.array([bool(re.search('.(%s).' % barcode, x.lower())) for barcode in templates for x in _data.var.index]).any()
            or (np.array([bool(re.search('(ens)[a-zA-Z]+[0-9]+', x.lower())) for x in _data.obs.index]).any())):
            print('transposing the data, identified cell barcodes or ensembl gene ids in the wrong axis')
            _data = _data.T

        return _data

    
    data = Data(_read(f),
                dense=dense)

    return data

def load_file(f,
              sep=None, 
              dense=False):
    
    if os.path.isfile(f): 
        which = [f.endswith(s) for s in SUFFIXES]
        suf = [i for i, elem in enumerate(which) if elem]
    
        if not any(which):
            return None
        
        data = _load_file(f, sep=sep, dense=dense)
        data.metadata = {'file': f.rsplit('/', 1)[1],
                         'folder': f.rsplit('/', 1)[0]+'/',
                         'suffix': SUFFIXES[suf[0]]}
        
        return _load_annotations(_load_names(data))

    else:
        raise FileNotFoundError(
            'File doesn\'t exist'
        )
    
def load(f, sep = None, dense = False):
    
    _suffixes = ['.dge' + suff for suff in SUFFIXES] + ['.matrix' + suff for suff in SUFFIXES]

    
    if os.path.isfile(f): 
        which = [f.endswith(s) for s in _suffixes]
        suf = [i for i, elem in enumerate(which) if elem]
    
        if not any(which):
            return None
        
        data = _load_file(f, sep=sep, dense=dense)
        data.metadata = {'file': f.rsplit('/', 1)[1],
                         'folder': f.rsplit('/', 1)[0]+'/',
                         'suffix': _suffixes[suf[0]]}
        
        return _load_annotations(_load_names(data))
    
    elif os.path.isdir(f):
        dataset = []
        for file in os.listdir(f):
            print('checking', file)
        
            if file.endswith(tuple(_suffixes)):
                dataset.append(load(os.path.join(f, file), dense=dense))
            
        if len(dataset) == 0:
            raise FileNotFoundError(
                'No data found in the folder'

            )
        if len(dataset) == 1:
            return dataset[0]
        print('careful, returning a collection of data object')
        return Dataset(dataset)

    else:
        raise FileNotFoundError(
            'File or folder doesn\'t exist'
        )
    
def _detect_delimiter(filename, n=6):

    def _head(filename, n):
        try:
            with open(filename) as f:
                head_lines = [next(f).rstrip() for x in range(n)]
        except StopIteration:
            with open(filename) as f:
                head_lines = f.read().splitlines()
        return head_lines

    sample_lines = _head(filename, n+1)
    common_delimiters= [',',';','\t',' ','|',':']
    for d in common_delimiters:
        ref = sample_lines[1].count(d)
        if ref > 0:
            if all([ ref == sample_lines[i].count(d) for i in range(2,n+1)]):
                print('separator identified in .txt file')
                return d
    print('no separator identified in .txt file, falling back on default separator')
    return '\t'

def _load_annotations(data):
    
    
    if (isinstance(data, Data)):

        filename = data.metadata['file']
        folder = data.metadata['folder']
        suff = data.metadata['suffix']
        filename = filename.rsplit(suff)[0]
        pattern = r'^(%s)([A-Za-z0-9\-\_\.]+)' % filename
        sep = {'csv':',', 'tsv':'\t', 'txt':None}


        for file in os.listdir(folder):
            if (re.search(pattern, file) and any([file.endswith(annot+'.'+s) for annot in ['.obs', '.var'] for s in sep.keys()])):
                
                print('loading annotations file %s' %file)
                _annot = file.rsplit('.', 2)[1]
                _sep = sep[file.rsplit('.', 1)[1]]

                if file.endswith('.txt'):
                    _sep = _detect_delimiter(os.path.join(folder, file))

                annots = pd.read_csv(os.path.join(folder, file), sep=_sep, header='infer').reset_index()
                cols = list(annots)
                
                if _annot == 'obs':
                    for col in cols:
                        isin = data.adata.obs_names.isin(annots[col])
                        if (isin.any() and 
                            annots[col].unique().shape[0] == annots[col].shape[0]):
                            annots = annots.set_index(col)
                            print('obs labels found in annotation file')
                            break
                    else:
                        print('no obs labels found in annotation file')
                        if annots.shape[0] == data.adata.shape[0]:
                            print('obs is same shape as data, matching index to index')
                            for col in cols: 
                                data.adata.obs[col] = annots[col].to_numpy()
                        else:
                            print('obs is not the same shape as data, skipping it')

                        continue
                        
                    for col in list(annots):
                        data.adata.obs[col] = np.nan
                        annots = annots.loc[data.adata.obs_names[isin]]
                        _new = annots.loc[data.adata.obs_names[isin]][col].to_numpy()
                        data.adata.obs.loc[isin, col] = _new

                elif _annot == 'var':
                    for col in cols:
                        data.adata.var[col] = annots[col].to_numpy()

    elif (isinstance(data, Dataset)):
        dataset = []
        for _data in data:
            dataset.append(_load_annotations(_data))
        data = dataset

    else:
        raise TypeError(
            '_load_annotations function can only accept Data and Dataset '
            'instances'
        )

    return data

# getting names from ensembl annotations if needed
def _get_names_from_ensembl(data):

    if (isinstance(data, Data)):

        files = {
            'ensmug': 'mus_musculus_113.csv',
            'ensg': 'Homo_sapiens.gencode.v32.primary_assembly.annotation.2019.csv'
        }

        for key, val in files.items():
            if data.adata.var_names[0].startswith(key):
                filename = val
                break
        else:
            return data
        
        annots = pd.read_csv('./annotations/'+filename).set_index('gene_id')
        isin = np.isin(data.adata.var_names,annots.index)
        data.adata.var['gene_name'] = data.adata.var_names.values
        data.adata.var['gene_name'][isin] = annots.loc[data.adata.var_names[isin]]['gene_name']
        nans = data.adata.var['gene_name'].isna()
        data.adata.var['gene_name'][nans] = data.adata.var_names[nans]
        data.adata.var['gene_symbol'] = data.adata.var_names
        data.adata.var_names = data.adata.var['gene_name']
        
        return data
    elif (isinstance(data, Dataset)):
        dataset = []
        for _data in data:
            dataset.append(_load_names(_data))
        data = dataset
    else:
        raise TypeError(
            '_get_names_from_ensembl function can only accept Data and Dataset '
            'instances'
        )

    return data

def _load_names(data):

    if (isinstance(data, Data)):
        
        filename = data.metadata['file']
        folder = data.metadata['folder']
        suff = data.metadata['suffix']
        filename = filename.rsplit(suff)[0]
        pattern = r'^(%s)([A-Za-z0-9\-\_\.]+)' % filename
        sep = {'csv':',', 'tsv':'\t', 'txt':None}

        genes = False
        barcodes = False
        for file in os.listdir(folder):
            if (re.search(pattern, file) and any([file.endswith(annot+'.'+s) for annot in ['.features', '.genes', '.barcodes'] for s in sep.keys()])):
                
                _annot = file.rsplit('.', 2)[1]
                _sep = sep[file.rsplit('.', 1)[1]]
                if file.endswith('.txt'):
                    _sep = _detect_delimiter(os.path.join(folder, file))
                _data = pd.read_csv(os.path.join(folder, file), sep=_sep, header=None)

                if _annot == 'barcodes':
                    if (data.adata.var_names.shape[0] == _data.shape[0]):
                        print('matrix is transposed, transposing back to cells x genes')
                        data.adata = data.adata.copy().transpose()
                    
                    if (data.adata.obs_names.shape[0] != _data.shape[0]):
                        print('number of barcodes does not match data')
                        continue

                    data.adata.obs_names = _data[0]
                    barcodes = True
                    print('found matching barcodes')

                elif (_annot == 'genes' or _annot == 'features'):
                    
                    # the gene names are always in the second column
                    if _data.shape[1] > 1:
                        names = [x.lower() for x in _data[1]]
                    else:
                        names = [x.lower() for x in _data[0]]

                    if len(names) != data.adata.var_names.shape[0]:
                        print('number of gene names does not match data')
                        continue
                    data.adata.var_names = names
                    genes = True
                    print('found maching gene/features names')


        if not genes:       
            print('no gene/features names file found, looking for ensembl annotations')
            data = _get_names_from_ensembl(data)

        if not barcodes:
            print('no cell barcodes file found')

        data.adata.var_names_make_unique()
        data.adata.obs_names_make_unique()

    elif (isinstance(data, Dataset)):
        dataset = []
        for _data in data:
            dataset.append(_load_names(_data))
        data = dataset

    else:
        raise TypeError(
            '_load_names function can only accept Data and Dataset '
            'instances'
        )

    return data
