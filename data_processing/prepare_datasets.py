from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

__all__ = ["infer_column_type", "join_colnames_to_lowercase", "drop_unnamed_columns"]


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
