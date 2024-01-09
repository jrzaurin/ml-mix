import pandas as pd

from data_processing.utils import load_col_types


class WomenClothingReviewTextProcessor:
    def __init__(
        self,
        dataset_name: str = "women_clothing_review",
    ):
        self.dataset_name = dataset_name
        self.col_types = load_col_types(dataset_name)

    def fit(self, df: pd.DataFrame) -> "WomenClothingReviewTextProcessor":
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
