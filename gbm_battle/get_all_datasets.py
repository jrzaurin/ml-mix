import os
from typing import List

import pandas as pd

from data_acquisition.datasets import create_dataset, TEXT_BENCHMARK_ALIAS_MAPPING


if __name__ == "__main__":
    os.environ["AUTO_MM_BENCH_HOME"] = "raw_data"

    datasets_l: List[pd.DataFrame] = []
    for dataset_name in list(TEXT_BENCHMARK_ALIAS_MAPPING.values()):
        dataset = create_dataset(dataset_name)
