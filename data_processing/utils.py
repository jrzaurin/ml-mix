import pickle
import warnings
from typing import Any, Dict, List, Union
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = "raw_data"
COL_TYPES_FNAME = "selected_datasets_col_types.pkl"


def load_col_types(dataset_name: str) -> Dict[str, List[str]]:
    with open(Path(DATA_DIR) / COL_TYPES_FNAME, "rb") as f:
        return pickle.load(f)[dataset_name]


def pd_read_csv_or_parquet(dataset_name: str, split: str = "train") -> pd.DataFrame:
    try:
        df = pd.read_csv(Path(DATA_DIR) / f"{dataset_name} / {split}.pq")
    except UnicodeDecodeError:
        df = pd.read_parquet(Path(DATA_DIR) / f"{dataset_name} / {split}.pq")
    return df


def infer_column_type(
    df: pd.DataFrame,
    cat_threshold: int = 100,
    text_threshold: float = 0.5,
    n_word_threshold: int = 3,
) -> Dict[str, List[str]]:
    text_cols = find_text_columns(df, text_threshold, n_word_threshold)

    cat_cols = find_categorical_columns(df, cat_threshold, exclude_columns=text_cols)

    num_cols = [column for column in df.columns if column not in text_cols + cat_cols]

    return {"text": text_cols, "cat": cat_cols, "num": num_cols}


def join_colnames_to_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["_".join(column.lower().split()) for column in df.columns]
    return df


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    unnamed_columns = [column for column in df.columns if "unnamed" in column.lower()]
    return df.drop(columns=unnamed_columns)


def find_categorical_columns(
    df: pd.DataFrame, cat_threshold: int, exclude_columns: List[str] = []
) -> List[str]:
    categorical_columns = []
    for column in df.columns:
        if (
            df[column].dtype == "object" or df[column].nunique() < cat_threshold
        ) and column not in exclude_columns:
            categorical_columns.append(column)
    return categorical_columns


def find_numerical_columns(
    df: pd.DataFrame, cat_threshold: int, exclude_columns: List[str] = []
) -> List[str]:
    numerical_columns = []
    for column in df.columns:
        if (
            df[column].dtype != "object"
            and df[column].nunique() < cat_threshold
            and column not in exclude_columns
        ):
            numerical_columns.append(column)
    return numerical_columns


def find_text_columns(
    df: pd.DataFrame, text_threshold: float, n_word_threshold: int
) -> List[str]:
    is_text = [
        text_condition(df[column], text_threshold, n_word_threshold)
        for column in df.columns
    ]

    text_cols = df.columns[is_text].to_list()

    return text_cols


def text_condition(
    X: Union[pd.Series, Any],  # Any to avoid some type issues down the line
    unique_threshold: float,
    n_word_threshold: int,
) -> bool:
    is_object = X.dtype == "object"

    # bool to avoid type conflict with np.bool_
    many_unique = bool(X.nunique() / (~X.isna()).sum() > unique_threshold)

    n_words_l: List[int] = []
    for val in X:
        try:
            n_words_l.append(len(val.split()))
        except AttributeError:
            pass
    median_num_words = np.median(n_words_l)
    more_than_n_words = bool(median_num_words >= n_word_threshold)

    return is_object and many_unique and more_than_n_words


def cols_to_lowercase_and_strip(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = df[c].str.lower().str.strip()
    return df


def drop_quasi_constant_cols(df: pd.DataFrame, frac=0.9) -> pd.DataFrame:
    constant_cols: List[str] = [
        c
        for c in df.columns
        if (df[c].value_counts() / df[c].value_counts().sum()).values[0] >= frac
    ]
    df = df.drop(constant_cols, axis=1)
    return df


def _get_columns_with_nan(df: pd.DataFrame) -> List[str]:
    columns_with_nan = df.isna().sum()[df.isna().sum() > 0].index.tolist()
    return columns_with_nan


def drop_high_nan_cols(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    n_rows = df.shape[0]

    has_nan = _get_columns_with_nan(df)

    nan_fraction_per_col = {c: df[c].isna().sum() / n_rows for c in has_nan}

    high_nan_percentage_cols = [
        k for k, v in nan_fraction_per_col.items() if v > threshold
    ]

    df = df.drop(high_nan_percentage_cols, axis=1)

    return df


def nan_with_unknown_imputer(
    df: pd.DataFrame, columns: List[str], replace_str: str | None = None
) -> pd.DataFrame:
    for c in columns:
        if replace_str is None:
            replace_str = (
                "unknown" if "unknown" not in df[c].unique() else "xxx_unknown_xxx"
            )
        try:
            df[c] = df[c].fillna(replace_str)
        except ValueError:
            df[c] = df[c].astype("float").fillna(replace_str)
    return df


def nan_with_number_imputer(
    df: pd.DataFrame, columns: List[str], number: float = -1.0
) -> pd.DataFrame:
    for c in columns:
        if (df[c] < 0).sum() > 1 and number < 0:
            warnings.warn(f"Column {c} has negative values and number is negative. ")
        df[c] = df[c].astype("float").fillna(number)
    return df


def drop_highly_correlated_columns(
    df: pd.DataFrame, cont_cols: List[str], crosscorr_val: float = 0.95
) -> pd.DataFrame:
    corr_matrix = df[cont_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype("bool"))
    to_drop = [column for column in upper.columns if any(upper[column] > crosscorr_val)]

    return df.drop(to_drop, axis=1)
