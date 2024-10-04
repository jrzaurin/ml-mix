import pandas as pd

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
        cols_to_drop: list[str] = ["clothing_id"],
        replace_cat_nans: bool = False,
        replace_num_nans: bool = False,
    ):
        self.dataset_name = dataset_name
        self.cols_to_drop = cols_to_drop
        self.replace_cat_nans = replace_cat_nans
        self.replace_num_nans = replace_num_nans

        self.col_types = load_col_types(dataset_name)

    def fit(self, df: pd.DataFrame) -> "WomenClothingReviewTabPreparer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc = drop_unnamed_columns(join_colnames_to_lowercase(df.copy()))
        dfc = dfc.drop(columns=self.cols_to_drop, axis=1)

        dfc = drop_high_nan_cols(dfc)

        dfc = drop_quasi_constant_cols(dfc)

        self.cat_cols = [
            c
            for c in dfc.columns
            if c in self.col_types["cat"] and c != self.col_types["target"][0]
        ]

        if self.cat_cols:
            dfc = cols_to_lowercase_and_strip(dfc, self.cat_cols)
            dfc[self.cat_cols] = dfc[self.cat_cols].astype("category")

            if self.replace_cat_nans:
                dfc = nan_with_unknown_imputer(dfc, self.cat_cols)

        self.cont_cols = [
            c
            for c in dfc.columns
            if c in self.col_types["num"] and c != self.col_types["target"][0]
        ]

        if self.cont_cols:
            if self.replace_num_nans:
                dfc = nan_with_number_imputer(dfc, self.cont_cols)

        return dfc

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


if __name__ == "__main__":
    df = pd.read_parquet("raw_data/datasets/women_clothing_review/train.pq")
    preparer = WomenClothingReviewTabPreparer()
    df_prepared = preparer.fit_transform(df)
