
import scanpy as sc
from ..base import SamplesTransformer, FeaturesTransformer
from sklearn.cluster import KMeans
    
class NNGraph(SamplesTransformer):

    def __init__(self,
                 n_neighbors=20,
                 metric='euclidean',
                 use_rep='X_pca'):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.use_rep = use_rep

    def transform(self, data, y=None):

        data = super().transform(data, y)
        sc.pp.neighbors(data.adata,
                        use_rep=self.use_rep,
                        metric=self.metric,
                        knn=True,
                        n_neighbors=self.n_neighbors)
        return data


class UMAP(FeaturesTransformer):

    def transform(self, data, y=None):

        data = super().transform(data, y)
        sc.tl.umap(data.adata)
        return data


class KMeansClustering(SamplesTransformer):

    def __init__(self, key='kmeans', use_rep='X_pca', n_clusters = 1):

        self.n_clusters = n_clusters
        self.key = key
        self.use_rep = use_rep
        
    def transform(self, data, y=None):

        data = super().transform(data, y)
        kmeans = KMeans(n_clusters=self.n_clusters).fit(data.adata.obsm[self.use_rep])
        data.adata.obs[self.key] = kmeans.labels_.astype(str)
        
        return data
    

class CommunityClustering(SamplesTransformer):

    def __init__(self, resolution = 0.4, method='leiden', key=None):

        self.resolution = resolution
        self.method = method
        self.key = key
        if key is None:
            self.key = method
        

    def transform(self, data, y=None):

        data = super().transform(data, y)
        transform_function = {
            'leiden': sc.tl.leiden,
            'louvain': sc.tl.louvain
        }[self.method]
        
        
        transform_function(data.adata, 
                           resolution=self.resolution, 
                           key_added=self.key)

        return data
    
class FindAllMarkers(FeaturesTransformer):

    def __init__(self, layer, group, num_genes=15):

        self.layer = layer
        self.group = group
        self.num_genes = num_genes

    def fit(self, data, y = None):     
        
        super().fit(data, y)
        return self
    
    def transform(self, data, y = None):

        data = super().transform(data, y)
        sc.tl.rank_genes_groups(
            data.adata, groupby=self.group, method="wilcoxon", 
            key_added=self.layer + '_' + self.group, n_genes=self.num_genes
        )

        return data