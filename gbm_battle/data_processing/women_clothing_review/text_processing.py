from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from data_processing.utils import load_col_types
from data_processing.feature_extraction import FeatureExtractor


class WomenClothingReviewTextProcessor:
    def __init__(
        self,
        dataset_name: str = "women_clothing_review",
        multiple_feat_extractors: bool = False,
        feat_extractor_params: Dict[str, Any] | Dict[str, Dict[str, Any]] = {},
    ):
        self.dataset_name = dataset_name
        self.col_types = load_col_types(dataset_name)
        self.multiple_feat_extractors = multiple_feat_extractors
        self.feat_extractor_params = feat_extractor_params

        self.feat_extractor = self._set_vectorizer()

    def _set_vectorizer(self) -> Dict[str, FeatureExtractor]:
        feat_extractor: Dict[str, FeatureExtractor] = {}
        for text_col in self.col_types["text"]:
            if self.multiple_feat_extractors:
                feat_extractor[text_col] = FeatureExtractor(
                    **self.feat_extractor_params[text_col]
                )
            else:
                feat_extractor[text_col] = FeatureExtractor(
                    **self.feat_extractor_params
                )

        return feat_extractor

    def fit(self, df: pd.DataFrame) -> "WomenClothingReviewTextProcessor":
        for col in self.col_types["text"]:
            self.feat_extractor[col].fit(df[col + "_tokenized"])
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, csr_matrix | np.ndarray]:
        features: Dict[str, Any] = {}
        for col in self.col_types["text"]:
            features[col] = self.feat_extractor[col].transform(df[col + "_tokenized"])
        return features

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, csr_matrix | np.ndarray]:
        return self.fit(df).transform(df)


if __name__ == "__main__":
    # from data_processing.women_clothing_review.tab_preparation import (
    #     WomenClothingReviewTabPreparer,
    # )
    # from data_processing.women_clothing_review.text_preparation import (
    #     WomenClothingReviewTextPreparer,
    # )

    # df = pd.read_parquet("raw_data/datasets/women_clothing_review/train.pq")
    # tab_preparer = WomenClothingReviewTabPreparer()
    # df_prepared = tab_preparer.fit_transform(df)

    # text_preparer = WomenClothingReviewTextPreparer(
    #     tokenizer_name="spacy", with_bigrams=True
    # )
    # df_text_prepared = text_preparer.fit_transform(df_prepared)
    # df_text_prepared.to_parquet("df_text_prepared.pq")

    df_text_prepared = pd.read_parquet("df_text_prepared.pq")
    df_text_prepared = df_text_prepared.sample(1000).reset_index(drop=True)

    feature_extractor_params = {
        "vectorizer_name": "tfidf",
        "max_vocab_size": 5000,
        "reduce_dim": True,
        "umap_params": {"metric": "hellinger"},
    }

    text_processor = WomenClothingReviewTextProcessor(
        feat_extractor_params=feature_extractor_params
    )
    features = text_processor.fit_transform(df_text_prepared)
