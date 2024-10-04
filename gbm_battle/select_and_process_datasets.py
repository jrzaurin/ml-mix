import os
import pickle
from typing import Dict, List
from pathlib import Path

import pandas as pd

from data_acquisition.datasets import create_dataset
from data_processing.utils import (
    infer_column_type,
    drop_unnamed_columns,
    join_colnames_to_lowercase,
)

if __name__ == "__main__":
    os.environ["AUTO_MM_BENCH_HOME"] = "raw_data"

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

        dataset = create_dataset(dataset_name)
        # this can be done by simply reading the train set, but I will use
        # the 'create_dataset' function
        # functionality like 'create_dataset' are not very well coded in the
        # original code (imo...)
        df = dataset.data

        df = drop_unnamed_columns(join_colnames_to_lowercase(df))
        col_types = infer_column_type(df)

        assert len(dataset.label_columns) == 1
        target_col = "_".join(dataset.label_columns[0].split()).lower()
        col_types["target"] = [target_col]  # List to avoid type error

        col_type_per_dataset[dataset_name] = col_types

    # json would probably be better...but for now pickle is fine
    with open(raw_data_path / "selected_datasets_col_types.pkl", "wb") as f:
        pickle.dump(col_type_per_dataset, f)
