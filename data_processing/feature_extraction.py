from typing import Any, Dict

from umap import UMAP
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextVectorizer:
    def __init__(
        self,
        vectorizer: str = "count",
        max_vocab_size: int = 5000,
        reduce_dim: bool = False,
        umap_params: Dict[str, Any] = {},
    ):
        self.max_vocab_size = max_vocab_size
        self.vectorizer = vectorizer
        self.reduce_dim = reduce_dim
        self.umap_params = umap_params

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        if self.vectorizer == "count":
            vectorizer = CountVectorizer(
                max_features=self.max_vocab_size,
                preprocessor=lambda x: x,
                tokenizer=lambda x: x,
            )
        elif self.vectorizer == "tfidf":
            vectorizer = TfidfVectorizer(
                max_features=self.max_vocab_size,
                preprocessor=lambda x: x,
                tokenizer=lambda x: x,
            )
        else:
            raise ValueError("Invalid vectorizer type. Choose 'count' or 'tfidf'.")

        if self.reduce_dim:
            umap = UMAP(**self.umap_params)
            pipeline = Pipeline([("vectorizer", vectorizer), ("umap", umap)])
        else:
            pipeline = Pipeline([("vectorizer", vectorizer)])

        return pipeline

    def fit(self, X):
        self.pipeline.fit(X)

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)
