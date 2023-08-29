import os
from pathlib import Path

from geobench.benchmark.dataset_converters import crop_type_south_africa, eurosat
from geobench.dataset import GeobenchDataset
from geobench.io import bandstats


def create_classification_test_benchmark():
    """Create classificaiton test benchmark."""
    new_benchmark_dir = os.path.join(os.getcwd(), "geobench-classification-test")
    dataset_dir = Path(new_benchmark_dir, eurosat.DATASET_NAME)
    eurosat.convert(max_count=5, dataset_dir=Path(dataset_dir))
    dataset = GeobenchDataset(dataset_dir, band_names=["red", "green", "blue"], partition_name="default")
    bandstats.produce_band_stats(dataset)


def create_segmentation_test_benchmark():
    new_benchmark_dir = os.path.join(os.getcwd(), "geobench-segmentation-test")
    dataset_dir = Path(new_benchmark_dir, crop_type_south_africa.DATASET_NAME)
    crop_type_south_africa.convert(max_count=5, dataset_dir=Path(dataset_dir))
    dataset = GeobenchDataset(dataset_dir, band_names=["red", "green", "blue"], partition_name="default")
    bandstats.produce_band_stats(dataset)


if __name__ == "__main__":
    create_classification_test_benchmark()
    create_segmentation_test_benchmark()
