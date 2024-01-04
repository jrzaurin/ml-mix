from typing import Dict, List
import pickle

from pathlib import Path

import pandas as pd

from data_processing.prepare_datasets import (
    infer_column_type,
    join_colnames_to_lowercase,
    drop_unnamed_columns,
)

if __name__ == "__main__":
    raw_data_path = Path("raw_data")

    datasets_path = raw_data_path / "datasets"

    df_stats = pd.read_csv(raw_data_path / "dataset_stats.csv")

    selected_datasets = df_stats[
        (df_stats["#Cat."] > 1) & (df_stats["#Num."] > 0) & (df_stats["#Train"] > 1000)
    ].Name.to_list() + ["jigsaw_unintended_bias"]

    # the jigsaw_unintended_bias100K is identical to jigsaw_unintended_bias but smaller
    selected_datasets.remove("jigsaw_unintended_bias100K")

    col_type_per_dataset: Dict[str, Dict[str, List[str]]] = {}
    for dataset_name in selected_datasets:
        print(dataset_name)

        if dataset_name == "melbourne_airbnb":
            dataset_name = "airbnb_melbourne"

        if dataset_name == "bookprice_prediction":
            dataset_name = "bookprice"

        try:
            df = pd.read_csv(datasets_path / f"{dataset_name}/train.pq")
        except UnicodeDecodeError:
            df = pd.read_parquet(datasets_path / f"{dataset_name}/train.pq")

        df = drop_unnamed_columns(join_colnames_to_lowercase(df))
        col_type_per_dataset[dataset_name] = infer_column_type(df)

    # json would probably be better...but for now pickle is fine
    with open(raw_data_path / "selected_datasets_col_types.pkl", "wb") as f:
        pickle.dump(col_type_per_dataset, f)
