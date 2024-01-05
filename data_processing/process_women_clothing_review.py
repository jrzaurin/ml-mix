from typing import Any, List, Union

import pandas as pd
from pytorch_widedeep.utils import LabelEncoder

from data_processing.utils import (
    load_col_types,
    drop_high_nan_cols,
    drop_unnamed_columns,
    nan_with_number_imputer,
    drop_quasi_constant_cols,
    nan_with_unknown_imputer,
    join_colnames_to_lowercase,
    cols_to_lowercase_and_strip,
)


class WomenClothingReviewTabPreparer:
    def __init__(
        self,
        dataset_name: str = "women_clothing_review",
        replace_cat_nans: bool = True,
        replace_num_nans: bool = True,
    ):
        self.col_types = load_col_types(dataset_name)
        self.replace_cat_nans = replace_cat_nans
        self.replace_num_nans = replace_num_nans

    def fit(self, df: pd.DataFrame) -> "WomenClothingReviewTabPreparer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc = drop_unnamed_columns(join_colnames_to_lowercase(df.copy()))

        dfc = drop_high_nan_cols(dfc)

        dfc = drop_quasi_constant_cols(dfc)

        cat_cols = [
            c
            for c in df.columns
            if c in self.col_types["cat"] and c != self.col_types["target"][0]
        ]

        if cat_cols:
            dfc = cols_to_lowercase_and_strip(dfc, cat_cols)

            if self.replace_cat_nans:
                dfc = nan_with_unknown_imputer(dfc, cat_cols)

        cont_cols = [
            c
            for c in df.columns
            if c in self.col_types["num"] and c != self.col_types["target"][0]
        ]

        if cont_cols:
            if self.replace_num_nans:
                dfc = nan_with_number_imputer(dfc, cont_cols)

        return dfc

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class WomenClothingReviewTabProcessor:
    def __init__(
        self,
        dataset_name: str = "women_clothing_review",
        ignore_columns: List[str] = ["clothing_id"],
        cols_to_scale: Union[List[str], str] = [],
        scaler: Any | None = None,
    ):
        self.dataset_name = dataset_name
        self.ignore_columns = ignore_columns
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
        cat_cols = [
            c
            for c in df.columns
            if c in self.col_types["cat"]
            and c not in self.ignore_columns + self.col_types["target"]
        ]
        if cat_cols:
            self.label_encoder = LabelEncoder(columns_to_encode=cat_cols)
            self.label_encoder.fit(df)

        cont_cols = [
            c
            for c in df.columns
            if c in self.col_types["num"]
            and c not in self.ignore_columns + self.col_types["target"]
        ]

        if cont_cols and self.cols_to_scale:
            self.scaler.fit(df[self.cols_to_scale].values)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_res = df.copy().drop(self.ignore_columns, axis=1)

        df_res = self.label_encoder.transform(df_res)

        df_res = df_res.rename(columns={self.col_types["target"][0]: "target"})

        if self.scaler:
            df_res[self.cols_to_scale] = self.scaler.transform(
                df_res[self.cols_to_scale].values
            )

        return df_res

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


if __name__ == "__main__":
    df = pd.read_parquet("raw_data/datasets/women_clothing_review/train.pq")
    processor = WomenClothingReviewTabProcessor()
    df_processsed = processor.fit_transform(df)
