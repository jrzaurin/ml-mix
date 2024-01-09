from typing import Any, List, Union

import pandas as pd
from pytorch_widedeep.utils import LabelEncoder

from data_processing.utils import load_col_types


class WomenClothingReviewTabProcessor:
    def __init__(
        self,
        dataset_name: str = "women_clothing_review",
        cols_to_scale: Union[List[str], str] = [],
        scaler: Any | None = None,
    ):
        self.dataset_name = dataset_name
        self.cols_to_scale = cols_to_scale
        self.scaler = scaler

        if cols_to_scale and scaler:
            raise ValueError("If cols_to_scale is not None, a scaler must be provided")

        if scaler:
            assert hasattr(scaler, "fit") and hasattr(scaler, "transform"), (
                "scaler must have fit and transform methods. "
                "For example, from sklearn.preprocessing import StandardScaler"
            )

        self.col_types = load_col_types(dataset_name)

    def fit(self, df: pd.DataFrame) -> "WomenClothingReviewTabProcessor":
        self.cat_cols = [
            c
            for c in df.columns
            if c in self.col_types["cat"] and c != self.col_types["target"][0]
        ]
        if self.cat_cols:
            self.label_encoder = LabelEncoder(columns_to_encode=self.cat_cols)
            self.label_encoder.fit(df)

        self.cont_cols = [
            c
            for c in df.columns
            if c in self.col_types["num"] and c != self.col_types["target"][0]
        ]

        if self.cont_cols and self.cols_to_scale:
            self.scaler.fit(df[self.cols_to_scale].values)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_res = self.label_encoder.transform(df.copy())

        df_res = df_res.rename(columns={self.col_types["target"][0]: "target"})

        if self.scaler:
            df_res[self.cols_to_scale] = self.scaler.transform(
                df_res[self.cols_to_scale].values
            )

        return df_res

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


if __name__ == "__main__":
    from data_processing.women_clothing_review.tab_preparation import (
        WomenClothingReviewTabPreparer,
    )

    df = pd.read_parquet("raw_data/datasets/women_clothing_review/train.pq")
    preparer = WomenClothingReviewTabPreparer()
    df_prepared = preparer.fit_transform(df)
    processor = WomenClothingReviewTabProcessor()
    df_processed = processor.fit_transform(df_prepared)
