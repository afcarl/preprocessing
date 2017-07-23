import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class YearExtractor(BaseEstimator, TransformerMixin):
    """Transform datetime column by extracting the year from it

    This estimator extracts the year from the datetime column and scales it if necessary. An alternative
    is to dummify the years, which can be useful if you expect non-linear relationships between the year and
    the target or non-linear interactions between the year and other features, however this fails when you
    get years in your test set or actual predictions that are not in your training set. At maximum only one of
    scale and dummify can be set to True. Scale uses scikit-learns StandardScaler

    Parameters
    ----------
    scale : boolean, optional, default True

    dummify : boolean, optional, default False"""

    def __init__(self, scale=True, dummify=False):
        self.scale = scale
        self.dummify = dummify
        self.fitted = False

        if scale and dummify:
            raise ValueError("Scale and dummify cannot both be True")

        self.scaler = StandardScaler()
        self.seen_years = set()

    def _reset(self):
        self.scaler = StandardScaler()
        self.seen_years = set()
        self.fitted = False

    def fit(self, X, y=None):
        if X.shape[1] > 1:
            raise ValueError("YearExtractor expects exactly one column")
        self._reset()
        self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        self.fitted = True
        if self.dummify:
            years = X.dt.year
            self.seen_years += years
        else:
            if self.scale:
                self.scaler.partial_fit(X.dt.year, y)

    def transform(self, X, y='deprecated', copy=None):
        if not self.fitted:
            raise ValueError("YearExtractor has not been fitted")
        if not self.scale:
            if not self.dummify:
                return X.dt.year
            else:
                years = sorted(list(self.seen_years))
                years_cat = pd.Categorical(X, categories=years, ordered=True)
                return pd.get_dummies(years_cat)
        return self.scaler.transform(X, y, copy)

#test_df = pd.DataFrame({'a': [pd.datetime.strptime("01-05-1987"), pd.datetime.strptime("02-05-1987"),
#                              pd.datetime.strptime("01-05-1988"), pd.datetime.strptime("01-05-1997")]})
#print(test_df)