from typing import Literal

import pandas as pd
from pytorch_widedeep.utils import Tokenizer as FastaiTokenizer

from data_processing.utils import load_col_types
from data_processing.tokenizers import (
    BigramTokenizer,
    NLTKLemmaTokenizer,
    SpacyLemmaTokenizer,
)


class WomenClothingReviewTextPreparer:
    def __init__(
        self,
        dataset_name: str = "women_clothing_review",
        tokenizer_name: Literal["fastai", "spacy", "nltk"] = "fastai",
        remove_stopwords: bool = False,
        with_bigrams: bool = False,
    ):
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.remove_stopwords = remove_stopwords
        self.with_bigrams = with_bigrams

        self.col_types = load_col_types(dataset_name)

    def fit(self, df: pd.DataFrame) -> "WomenClothingReviewTextPreparer":
        if self.tokenizer_name == "fastai":
            self.tokenizer: FastaiTokenizer | SpacyLemmaTokenizer | NLTKLemmaTokenizer = (
                FastaiTokenizer()
            )
        elif self.tokenizer_name == "spacy":
            self.tokenizer = SpacyLemmaTokenizer(self.remove_stopwords)
        elif self.tokenizer_name == "nltk":
            self.tokenizer = NLTKLemmaTokenizer(self.remove_stopwords)
        else:
            raise ValueError("tokenizer must be one of 'fastai', 'spacy', or 'nltk'")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.col_types["text"]:
            tokenized_text = self.tokenizer.process_all(df[col].tolist())
            if self.with_bigrams:
                tokenized_text = BigramTokenizer().process_all(tokenized_text)
            df[col + "_tokenized"] = tokenized_text
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


if __name__ == "__main__":
    from data_processing.women_clothing_review.tab_preparation import (
        WomenClothingReviewTabPreparer,
    )

    df = pd.read_parquet("raw_data/datasets/women_clothing_review/train.pq")
    preparer = WomenClothingReviewTabPreparer()
    df_prepared = preparer.fit_transform(df)
    text_preparer = WomenClothingReviewTextPreparer(
        tokenizer_name="nltk", with_bigrams=True
    )
    df_text_prepared = text_preparer.fit_transform(df_prepared)
