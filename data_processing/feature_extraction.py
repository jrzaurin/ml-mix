from typing import Any, Dict

import scipy.sparse as sp
from umap import UMAP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class ConvertToLilSparse(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if sp.issparse(X):
            return X.tolil()
        else:
            return X


class FeatureExtractor:
    def __init__(
        self,
        vectorizer_name: str = "count",
        max_vocab_size: int = 5000,
        reduce_dim: bool = False,
        umap_params: Dict[str, Any] = {},
    ):
        self.max_vocab_size = max_vocab_size
        self.vectorizer_name = vectorizer_name
        self.reduce_dim = reduce_dim
        self.umap_params = umap_params

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        if self.vectorizer_name == "count":
            vectorizer = CountVectorizer(
                min_df=10,
                max_features=self.max_vocab_size,
                preprocessor=lambda x: x,
                tokenizer=lambda x: x,
            )
        elif self.vectorizer_name == "tfidf":
            vectorizer = TfidfVectorizer(
                min_df=10,
                max_features=self.max_vocab_size,
                preprocessor=lambda x: x,
                tokenizer=lambda x: x,
            )
        else:
            raise ValueError("Invalid vectorizer type. Choose 'count' or 'tfidf'.")

        if self.reduce_dim:
            umap = UMAP(**self.umap_params)

            pipeline = Pipeline(
                [
                    ("vectorizer", vectorizer),
                    ("tolil", ConvertToLilSparse()),
                    ("umap", umap),
                ]
            )
        else:
            pipeline = Pipeline([("vectorizer", vectorizer)])

        return pipeline

    def fit(self, X) -> "FeatureExtractor":
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)
