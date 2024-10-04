from typing import Any, Dict, List

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

DROP_COLS = [
    "listing_url",
    "scrape_id",
    "last_scraped",
    "picture_url",
    "host_id",
    "host_url",
    "host_thumbnail_url",
    "host_picture_url",
    "weekly_price",
    "monthly_price",
    "maximum_nights",
    "calendar_updated",
    "has_availability",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "calendar_last_scraped",
    "number_of_reviews",
    "first_review",
    "last_review",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "requires_license",
    "license",
    "require_guest_profile_picture",
    "require_guest_phone_verification",
    "host_verifications_jumio",
    "host_verifications_government_id",
    "host_verifications_kba",
    "host_verifications_zhima_selfie",
    "host_verifications_facebook",
    "host_verifications_work_email",
    "host_verifications_google",
    "host_verifications_sesame",
    "host_verifications_manual_online",
    "host_verifications_manual_offline",
    "host_verifications_offline_government_id",
    "host_verifications_selfie",
    "host_verifications_reviews",
    "host_verifications_identity_manual",
    "host_verifications_sesame_offline",
    "host_verifications_weibo",
    "host_verifications_email",
    "host_verifications_sent_id",
    "host_verifications_phone",
]


def try_join_str(x: str | Any) -> str | None:
    try:
        return " ".join(eval(x))
    except Exception:
        return None


def prepare_amenities_and_host_verifications(
    df: pd.DataFrame, col_types: Dict[str, List[str]]
) -> pd.DataFrame:
    df["amenities"] = df.amenities.apply(
        lambda x: " ".join(
            [
                s.lower()
                for s in x.replace("{", "").replace("}", "").replace('"', "").split(",")
            ]
        )
    )

    df["host_verifications"] = df.host_verifications.apply(try_join_str)

    col_types["text"] += ["amenities", "host_verifications"]
    col_types["cat"].remove("host_verifications")

    return df


class AirbnbMelbourneTabPreparer:
    def __init__(
        self,
        dataset_name: str = "melbourne_airbnb",
        cols_to_drop: list[str] = DROP_COLS,
        replace_cat_nans: bool = False,
        replace_num_nans: bool = False,
    ):
        self.dataset_name = dataset_name
        self.cols_to_drop = cols_to_drop
        self.replace_cat_nans = replace_cat_nans
        self.replace_num_nans = replace_num_nans

        self.col_types = load_col_types(dataset_name)

    def fit(self, df: pd.DataFrame) -> "AirbnbMelbourneTabPreparer":
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
    preparer = AirbnbMelbourneTabPreparer()
    df_prepared = preparer.fit_transform(df)
