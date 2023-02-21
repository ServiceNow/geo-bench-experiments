"""So2Sat dataset."""
# So2Sat will be automatically downloaded by TorchGeo (https://github.com/microsoft/torchgeo)

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torchgeo.datasets import So2Sat
from tqdm import tqdm

from geobench import io
from geobench.io.dataset import Sample
from geobench.io.task import TaskSpecifications

DATASET_NAME = "so2sat"
SRC_DATASET_DIR = io.CCB_DIR / "source" / DATASET_NAME  # type: ignore
DATASET_DIR = io.CCB_DIR / "converted" / DATASET_NAME  # type: ignore


class GeoSo2Sat(So2Sat):

    all_s1_band_names = (
        "S1_B1",
        "S1_B2",
        "S1_B3",
        "S1_B4",
        "S1_B5",
        "S1_B6",
        "S1_B7",
        "S1_B8",
    )
    all_s2_band_names = (
        "S2_B02",
        "S2_B03",
        "S2_B04",
        "S2_B05",
        "S2_B06",
        "S2_B07",
        "S2_B08",
        "S2_B8A",
        "S2_B11",
        "S2_B12",
    )
    all_band_names = all_s1_band_names + all_s2_band_names

    rgb_bands = ["S2_B04", "S2_B03", "S2_B02"]

    BAND_SETS = {
        "all": all_band_names,
        "s1": all_s1_band_names,
        "s2": all_s2_band_names,
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new So2Sat dataset instance.
        Args:
            root: root directory where dataset can be found
            split: one of "train", "validation", or "test"
            bands: a sequence of band names to use where the indices correspond to the
                array index of combined Sentinel 1 and Sentinel 2
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        super().__init__(root, split, bands, transforms, checksum)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        with h5py.File(self.fn, "r") as f:
            s1 = f["sen1"][index].astype(np.float64)  # convert from <f8 to float64
            s1 = np.take(s1, indices=self.s1_band_indices, axis=2)
            s2 = f["sen2"][index].astype(np.float64)  # convert from <f8 to float64
            s2 = np.take(s2, indices=self.s2_band_indices, axis=2)

            # convert one-hot encoding to int64 then torch int
            label = torch.tensor(f["label"][index].argmax())

            s1 = np.rollaxis(s1, 2, 0)  # convert to CxHxW format
            s2 = np.rollaxis(s2, 2, 0)  # convert to CxHxW format

            s1 = torch.from_numpy(s1)
            s2 = torch.from_numpy(s2)

        sample = {"image": torch.cat([s1, s2]).float(), "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


def make_sample(
    images: "np.typing.NDArray[np.int_]", label: int, sample_name: str, task_specs: TaskSpecifications
) -> Sample:
    """Create a sample from images and label.

    Args:
        images: image array to be contained in sample
        label: label to be contained in sample
        sample_name: name of sample
        task_specs: task specifications of this datasets

    Returns:
        sample
    """
    n_bands, _height, _width = images.shape

    transform = None  # TODO can't find the GPS coordinates from torch geo.
    crs = None

    bands = []
    for band_idx in range(n_bands):
        band_data = images[band_idx, :, :]

        band_info = task_specs.bands_info[band_idx]
        band_data = band_data.astype(np.float32)
        band = io.Band(
            data=band_data,
            band_info=band_info,
            spatial_resolution=task_specs.spatial_resolution,
            transform=transform,
            crs=crs,
            convert_to_int16=False,
        )
        bands.append(band)

    return io.Sample(bands, label=label, sample_name=sample_name)


def convert(max_count: int = None, dataset_dir: Path = DATASET_DIR) -> None:
    """Convert So2Sat dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)
    partition = io.dataset.Partition()

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(32, 32),
        n_time_steps=1,
        bands_info=io.sentinel1_8_bands + io.sentinel2_13_bands[1:9] + io.sentinel2_13_bands[-2:],  # type: ignore
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.Classification(17, class_names=So2Sat.classes),
        spatial_resolution=10,
    )
    task_specs.save(str(dataset_dir), overwrite=True)
    n_samples = 0
    for split_name in ["train", "validation", "test"]:
        so2sat_dataset = So2Sat(root=SRC_DATASET_DIR, split=split_name, transforms=None, checksum=False)
        for tg_sample in tqdm(so2sat_dataset):
            sample_name = f"id_{n_samples:04d}"

            images = np.array(tg_sample["image"])
            label = tg_sample["label"]

            sample = make_sample(images, int(label), sample_name, task_specs)
            sample.write(str(dataset_dir))

            partition.add(split_name.replace("validation", "valid"), sample_name)

            n_samples += 1
            if max_count is not None and n_samples >= max_count:
                break

        if max_count is not None and n_samples >= max_count:
            break

    partition.save(str(dataset_dir), "original", as_default=True)


if __name__ == "__main__":
    convert()
