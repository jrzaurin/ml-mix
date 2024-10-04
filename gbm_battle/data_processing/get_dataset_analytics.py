import os

import numpy as np
import pandas as pd
import sklearn.model_selection
from autogluon.core.utils import infer_problem_type
from autogluon.multimodal import constants as _C
from autogluon.multimodal.data.infer_types import infer_column_types

from data_acquisition.datasets import dataset_registry

os.environ["AUTO_MM_BENCH_HOME"] = "raw_data"


def get_stats():
    dataset_stats = {
        "Name": [],
        "#Cat.": [],
        "#Num.": [],
        "#Text": [],
        "Problem Type": [],
        "#Train": [],
        "#Test": [],
        "#Competition": [],
        "Metric": [],
    }

    train_dataset_l = []
    test_dataset_l = []
    for dataset_name in dataset_registry.list_keys():
        print("Processing:", dataset_name)
        if dataset_name == "google_qa_label":
            print("Skip google_qa_label")
            continue
        dataset_stats["Name"].append(dataset_name)
        dataset_cls = dataset_registry.get(dataset_name)
        if "competition" in dataset_cls.splits():
            competition_dataset = dataset_cls(split="competition")
            competition_num = len(competition_dataset.data)
        else:
            competition_num = 0
        dataset_stats["#Competition"].append(competition_num)
        train_dataset = dataset_cls(split="train")
        test_dataset = dataset_cls(split="test")

        train_dataset_l.append(train_dataset)
        test_dataset_l.append(test_dataset)

    for train_dataset, test_dataset in zip(train_dataset_l, test_dataset_l):
        problem_type = train_dataset.problem_type
        metric = train_dataset.metric
        train_num = len(train_dataset.data)
        test_num = len(test_dataset.data)
        train_df, valid_df = sklearn.model_selection.train_test_split(
            train_dataset.data, test_size=0.05, random_state=np.random.RandomState(100)
        )

        column_types = infer_column_types(
            train_df,
            valid_df,
            label_columns=train_dataset.label_columns,
            problem_type=problem_type,
        )

        inferred_problem_type = infer_problem_type(
            y=train_df[train_dataset.label_columns[0]],
        )

        try:
            assert inferred_problem_type == problem_type
        except AssertionError:
            print(
                f"Problem type mismatch: {inferred_problem_type} (inferred) vs {problem_type} (specified)"
            )

        problem_type = inferred_problem_type

        for label, gt_label_type in zip(
            train_dataset.label_columns, train_dataset.label_types
        ):
            assert column_types[label] == gt_label_type

        feature_col_types = [column_types[col] for col in train_dataset.feature_columns]
        num_categorical = sum(
            [col_type == _C.CATEGORICAL for col_type in feature_col_types]
        )
        num_numerical = sum(
            [col_type == _C.NUMERICAL for col_type in feature_col_types]
        )
        num_text = sum([col_type == _C.TEXT for col_type in feature_col_types])
        dataset_stats["#Cat."].append(num_categorical)
        dataset_stats["#Num."].append(num_numerical)
        dataset_stats["#Text"].append(num_text)
        dataset_stats["Problem Type"].append(problem_type)
        dataset_stats["#Train"].append(train_num)
        dataset_stats["#Test"].append(test_num)
        dataset_stats["Metric"].append(metric)

    dataset_stats["#Competition"] = np.array(
        dataset_stats["#Competition"], dtype=np.int64
    )
    dataset_stats = pd.DataFrame(dataset_stats)
    return dataset_stats


if __name__ == "__main__":
    dataset_stats = get_stats()
