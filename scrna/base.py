
from sklearn.base import BaseEstimator, TransformerMixin
from .data import Data

class _BaseTransformer(TransformerMixin, BaseEstimator):

    def _validate_data_compatible(self, X, y='no_validation', **params):
        
        try:
            from sklearn.utils.validation import validate_data
            return validate_data(self, X, y, **params)
        
        except:
            if hasattr(self, '_validate_data'):
                return self._validate_data(X, y, **params)
            raise ModuleNotFoundError(
                'Problem with sklearn version, there is no validate_data function, nor is there a _validate_data \
                 private method'
            )

            
    def fit(self, data, y=None):

        if not isinstance(data, Data):
            _ = self._validate_data_compatible(data, y=y, accept_sparse=True)
            return self
        _ = self._validate_data_compatible(data.adata.X, y=y, accept_sparse=True)
        return self

    def transform(self, data, y=None):

        if not isinstance(data, Data):
            X = self._validate_data_compatible(data, y=y, reset=False, accept_sparse=True)
            return Data(X)
        
        X = self._validate_data_compatible(data.adata.X, y=y, reset=False, accept_sparse=True)
        
        return data

# TODO: add some characteristics ensuring that samples are not changed
class FeaturesTransformer(_BaseTransformer):
    pass

# TODO: add some characteristics ensuring that features are not changed
class SamplesTransformer(_BaseTransformer):
    pass