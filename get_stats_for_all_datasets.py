import os

from data_processing.get_dataset_analytics import get_stats


if __name__ == "__main__":
    os.environ["AUTO_MM_BENCH_HOME"] = "raw_data"

    dataset_stats = get_stats()
    dataset_stats.to_csv("raw_data/dataset_stats.csv", index=False)
