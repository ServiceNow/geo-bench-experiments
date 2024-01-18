"""geobench.dataset Datamodule."""


from typing import Any, Callable, Dict, Sequence

import kornia.augmentation as K
import numpy as np
import torch
from geobench.dataset import Band, Sample
from geobench.task import TaskSpecifications
from kornia.augmentation import ImageSequential
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.transforms import AugmentationSequential


def get_transform(task_specs, config, train):
    """Decide which transforms to get."""
    if "classification" in task_specs.benchmark_name:
        return get_classification_transform(task_specs, config, train)
    elif "segmentation" in task_specs.benchmark_name:
        return get_segmentation_transform(task_specs, config, train)
    else:
        raise NotImplementedError
    
def get_desired_input_sizes(model_name: str) -> int:
    """Define input sizes for models."""
    input_size_dict = {
        "resnet18": 224,
        "resnet50": 224,
        "convnext_base": 224,
        "vit_tiny_patch16_224": 224,
        "vit_small_patch16_224": 224,
        "swinv2_tiny_window16_256": 256,
    }
    return input_size_dict[model_name]

def get_classification_transform(task_specs, config: Dict[str, Any], train=True) -> Callable[[Sample], Dict[str, Any]]:
    """Define data transformations specific to the models generated.

    Args:
        task_specs: task specs to retrieve dataset
        config: config file for dataset specifics
        train: train mode true or false

    Returns:
        callable function that applies transformations on input data
    """

    mean, std = task_specs.get_dataset(
        split="train",
        format=config["datamodule"]["format"],
        band_names=tuple(config["datamodule"]["band_names"]),
        partition_name=config["experiment"]["partition_name"],
    ).normalization_stats()

    desired_input_size = get_desired_input_sizes(config["model"]["model"])

    if train:
        t = ImageSequential(
            K.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.Resize((desired_input_size, desired_input_size)),
        )
    else:
        t = ImageSequential(
            K.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
            K.Resize((desired_input_size, desired_input_size)),
        )

    def transform(sample: Sample):
        x: "np.typing.NDArray[np.float_]" = sample.pack_to_3d(band_names=config["datamodule"]["band_names"])[0].astype(
            "float32"
        )
        x = t(torch.from_numpy(x).permute(2, 0, 1)).squeeze(0)
        return {"input": x, "label": sample.label}

    return transform


def get_segmentation_transform(
    task_specs: TaskSpecifications,
    config: Dict[str, Any],
    train=True,
):
    """Define data transformations specific to the models generated.

    Args:
        task_specs: task specs to retrieve dataset
        hparams: model hyperparameters
        train: train mode true or false

    Returns:
        callable function that applies transformations on input data
    """
    c, h, w = config["model"]["input_size"]
    patch_h, patch_w = task_specs.patch_size
    if h != w or patch_h != patch_w:
        raise (RuntimeError("Only square patches are supported in this version"))
    h32 = w32 = int(32 * (h // 32))  # make input res multiple of 32

    mean, std = task_specs.get_dataset(
        split="train",
        format=config["dataset"]["format"],
        band_names=tuple(config["dataset"]["band_names"]),
        # benchmark_dir=config["experiment"]["benchmark_dir"],
        partition_name=config["experiment"]["partition_name"],
    ).rgb_stats()
    band_names = config["dataset"]["band_names"]

    if train:
        t = AugmentationSequential(
            K.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.Resize((h32, w32)),
            data_keys=["image", "mask"],
        )
    else:
        t = AugmentationSequential(
            K.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
            K.Resize((h32, w32)),
            data_keys=["image", "mask"],
        )

    def transform(sample: Sample):
        x = sample.pack_to_3d(band_names=band_names)[0].astype("float32")

        if isinstance(sample.label, Band):
            # kornia expects channel first and label to have a channel
            x, y = torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(
                sample.label.data.astype("float32")
            ).unsqueeze(0)
            transformed = t({"image": x, "mask": y})

        return {
            "input": transformed["image"].squeeze(0),
            "label": transformed["mask"].squeeze(0).to(dtype=torch.long),
        }

    return transform


class DataModule(LightningDataModule):
    """Data Module.

    Define a
    `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html>`_
    that provides dataloaders from task_specs.
    """

    def __init__(
        self,
        task_specs: TaskSpecifications,
        benchmark_dir: str,
        partition_name: str,
        batch_size: int,
        num_workers: int,
        val_batch_size: int = None,
        train_transform=None,
        eval_transform=None,
        collate_fn=None,
        band_names: Sequence[str] = ("red", "green", "blue"),
        format: str = "hdf5",
    ) -> None:
        """Initialize new instance of DataModule .

        Args:
            task_specs: TaskSpecifications object to call get_dataset.
            benchmark_dir: path to benchmark directory that contains datasets
            partition_name: name of partition to load
            batch_size: The size of the mini-batch.
            num_workers: The number of parallel workers for loading samples from the hard-drive.
            val_batch_size: Tes size of the batch for the validation set and test set. If None, will use batch_size.
            transform: Callable transforming a Sample. Executed on a worker and the output will be provided to collate_fn.
            collate_fn: A callable passed to the DataLoader. Maps a list of Sample to dictionnary of stacked torch tensors.
            band_names: multi spectral bands to select
            file_format: 'hdf5' or 'tif'
        """
        super().__init__()
        self.task_specs = task_specs
        self.benchmark_dir = benchmark_dir
        self.partition_name = partition_name
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.collate_fn = collate_fn
        self.band_names = band_names
        self.format = format

    def train_dataloader(self) -> DataLoader:
        """Create the train dataloader."""
        return DataLoader(
            self.task_specs.get_dataset(
                split="train",
                partition_name=self.partition_name,
                transform=self.train_transform,
                band_names=self.band_names,
                format=self.format,
                # benchmark_dir=self.benchmark_dir,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader."""
        return (
            DataLoader(
                self.task_specs.get_dataset(
                    split="valid",
                    partition_name=self.partition_name,
                    transform=self.eval_transform,
                    band_names=self.band_names,
                    format=self.format,
                    # benchmark_dir=Path(self.benchmark_dir),
                ),
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            ),
            DataLoader(
                self.task_specs.get_dataset(
                    split="test",
                    partition_name=self.partition_name,
                    transform=self.eval_transform,
                    band_names=self.band_names,
                    format=self.format,
                    # benchmark_dir=Path(self.benchmark_dir),
                ),
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader."""
        return DataLoader(
            self.task_specs.get_dataset(
                split="test",
                partition_name=self.partition_name,
                transform=self.eval_transform,
                band_names=self.band_names,
                format=self.format,
                # benchmark_dir=self.benchmark_dir,
            ),
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
