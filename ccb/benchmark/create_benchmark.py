from math import ceil, floor
import random
from typing import Dict, List, Tuple
from ccb import io
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def make_subsampler(max_sizes):
    def _subsample(partition, task_specs, rng=np.random):
        return subsample(partition=partition, max_sizes=max_sizes, rng=rng)

    return _subsample


def subsample(partition: io.Partition, max_sizes: Dict[str, int], rng=np.random) -> io.Partition:
    """Randomly subsample `partition` to satisfy `max_sizes`."""
    new_partition = io.Partition()

    for split_name, sample_names in partition.partition_dict.items():
        if len(sample_names) > max_sizes[split_name]:
            subset = list(rng.choice(sample_names, max_sizes[split_name], replace=False))
        else:
            subset = sample_names[:]  # create a copy to avoid potential issues
        new_partition.partition_dict[split_name] = subset
    return new_partition


def _make_split_label_maps(label_map: Dict[int, List[str]], partition_dict: Dict[str, List[str]]):
    """Organize label map into 'train', 'valid' and 'test'."""
    split_label_maps = {}
    reverse_label_map = {}
    for label, sample_names in label_map.items():
        for sample_name in sample_names:
            reverse_label_map[sample_name] = label
    for split, sample_names in partition_dict.items():
        split_label_maps[split] = defaultdict(list)
        for sample_name in sample_names:
            label = reverse_label_map[sample_name]
            split_label_maps[split][label].append(sample_name)
    return split_label_maps


def _filter_for_min_size(split_label_maps, min_class_sizes: Dict[str, int]):
    """Makes sure each class has statisfies `min_class_sizes`."""
    new_split_label_maps = defaultdict(dict)
    for label in split_label_maps["train"].keys():

        ok = True
        for split, min_class_size in min_class_sizes.items():
            if len(split_label_maps[split].get(label, ())) < min_class_size:
                ok = False
        if ok:
            for split in ("train", "valid", "test"):
                new_split_label_maps[split][label] = split_label_maps[split][label][:]

    return new_split_label_maps


def assert_no_overlap(split_label_maps: Dict[str, Dict[int, List[str]]]):
    """Asser that label map is a partition and that no sample are common across splits."""
    sample_set = set()
    total_count = 0
    for label_map in split_label_maps.values():
        for sample_names in label_map.values():
            sample_set.update(sample_names)
            total_count += len(sample_names)

    assert len(sample_set) == total_count


def make_resampler(max_sizes, min_class_sizes={"train": 10, "valid": 1, "test": 1}):
    """Matrialize a resampler with the required interface."""

    def _resample(partition, task_specs, rng=np.random):
        label_map = task_specs.label_map
        return resample(
            partition=partition, label_map=label_map, max_sizes=max_sizes, min_class_sizes=min_class_sizes, rng=rng
        )

    return _resample


def resample(
    partition: io.Partition,
    label_map: Dict[int, List[str]],
    max_sizes: Dict[str, int],
    min_class_sizes: Dict[str, int],
    verbose=True,
    rng=np.random,
) -> io.Partition:

    """Reduce class imbalance in `partition` based on information in `label_map`."""
    split_label_maps = _make_split_label_maps(label_map, partition_dict=partition.partition_dict)
    assert_no_overlap(split_label_maps)
    new_split_label_maps = _filter_for_min_size(split_label_maps, min_class_sizes)
    assert_no_overlap(new_split_label_maps)
    partition_dict = defaultdict(list)
    for split, max_size in max_sizes.items():
        label_map = new_split_label_maps[split]
        max_sample_per_class = floor(max_size / len(label_map))
        for label, sample_names in label_map.items():
            if len(sample_names) > max_sample_per_class:
                label_map[label] = rng.choice(sample_names, size=max_sample_per_class, replace=False)

            partition_dict[split].extend(label_map[label])

    for sample_names in partition_dict.values():
        rng.shuffle(sample_names)  # shuffle in place the mutable sequence

    if verbose:
        print("Class rebalancing:")
        for split, label_map in split_label_maps.items():
            print(f"{split}")
            for label, sample_names in label_map.items():
                new_sample_names = new_split_label_maps[split].get(label, ())
                print(f"  class {label} size: {len(sample_names)} -> {len(new_sample_names)}.")
        print()
    return io.Partition(partition_dict=partition_dict)


def make_resampler_from_stats(max_sizes):
    """Matrialize a resampler with the required interface."""

    def _resample(partition, task_specs, rng=np.random):
        label_stats = task_specs.label_stats
        return resample_from_stats(partition=partition, label_stats=label_stats, max_sizes=max_sizes, rng=rng)

    return _resample


def resample_from_stats(
    partition: io.Partition,
    label_stats: Dict[str, List[float]],
    max_sizes: Dict[str, int],
    verbose=True,
    rng=np.random,
    return_prob=False,
) -> io.Partition:

    partition_dict = defaultdict(list)
    prob_dict = {}
    for split, sample_names in partition.partition_dict.items():

        if len(sample_names) > max_sizes[split]:

            stats = np.array([label_stats[sample_name] for sample_name in sample_names])
            cum_stats = np.sum(stats, axis=0, keepdims=True)
            weight_factors = 1 / (cum_stats + 1)
            prob = np.sum(stats * weight_factors, axis=1)
            prob /= prob.sum()

            partition_dict[split] = list(rng.choice(sample_names, size=max_sizes[split], replace=False, p=prob))
            prob_dict[split] = prob

        else:
            print(f"Split {split} unchanged since {len(sample_names)} <= {max_sizes[split]}.")
            partition_dict[split] = sample_names

    new_partition = io.Partition(partition_dict=partition_dict)

    if return_prob:
        return new_partition, prob_dict
    else:
        return new_partition


def transform_dataset(
    dataset_dir: Path,
    new_benchmark_dir: Path,
    partition_name: str,
    resampler=None,
    sample_converter=None,
    delete_existing=False,
    hdf5=True,
):

    dataset = io.Dataset(dataset_dir, partition_name=partition_name)
    task_specs = dataset.task_specs
    task_specs.benchmark_name = dataset_dir.parent.name
    new_dataset_dir = new_benchmark_dir / dataset_dir.name

    if new_dataset_dir.exists() and delete_existing:
        print(f"Deleting exising dataset {new_dataset_dir}.")
        shutil.rmtree(new_dataset_dir)

    new_dataset_dir.mkdir(parents=True, exist_ok=True)

    if resampler is not None:
        new_partition = resampler(
            partition=dataset.load_partition(partition_name),
            task_specs=task_specs,
        )
    else:
        new_partition = dataset.load_partition(partition_name)

    task_specs.benchmark_name = new_benchmark_dir.name
    task_specs.save(new_dataset_dir, overwrite=True)

    for split_name, sample_names in new_partition.partition_dict.items():
        print(f"  Converting {len(sample_names)} samples from {split_name} split.")
        for sample_name in tqdm(sample_names):

            if sample_converter is None:
                if hdf5:
                    sample_name += ".hdf5"
                    shutil.copyfile(dataset_dir / sample_name, new_dataset_dir / sample_name)
                else:
                    shutil.copytree(dataset_dir / sample_name, new_dataset_dir / sample_name, dirs_exist_ok=True)
            else:
                raise NotImplementedError()

    new_partition.save(new_dataset_dir, "default")


def _make_benchmark(new_benchmark_name, specs, src_benchmark_name="converted"):

    for dataset_name, (resampler, sample_converter) in specs.items():
        print(f"Transforming {dataset_name}.")
        transform_dataset(
            dataset_dir=io.CCB_DIR / src_benchmark_name / dataset_name,
            new_benchmark_dir=io.CCB_DIR / new_benchmark_name,
            partition_name="default",
            resampler=resampler,
            sample_converter=sample_converter,
            delete_existing=True,
        )


def make_classification_benchmark():
    max_sizes = {"train": 3000, "valid": 1000, "test": 1000}
    default_resampler = make_resampler(max_sizes=max_sizes)
    specs = {
        "eurosat": (default_resampler, None),
        "brick_kiln_v1.0": (default_resampler, None),
        # "so2sat": (default_resampler, None),
        # "pv4ger_classification": (default_resampler, None),
        "geolifeclef-2021": (make_resampler(max_sizes={"train": 10000, "valid": 5000, "test": 5000}), None),
        # "bigearthnet": (make_subsampler(max_sizes), None),
    }
    _make_benchmark("classification_v0.4", specs)


def make_segmentation_benchmark():
    max_sizes = {"train": 3000, "valid": 1000, "test": 1000}
    # default_resampler = make_subsampler(max_sizes=max_sizes)
    resampler_from_stats = make_resampler_from_stats(max_sizes=max_sizes)
    specs = {
        # "pv4ger_segmentation": (resampler_from_stats, None),
        # "xview2": (resampler_from_stats, None),
        # # "forestnet_v1.0": (resampler_from_stats, None),
        "cvpr_chesapeake_landcover": (resampler_from_stats, None),
        # "nz_cattle_segmentation": (resampler_from_stats, None),
        # "NeonTree_segmentation": (resampler_from_stats, None),
    }
    _make_benchmark("segmentation_v0.2", specs)


if __name__ == "__main__":
    # make_classification_benchmark()
    make_segmentation_benchmark()
