"""Timm Model Generator."""

import random
from typing import Any, Callable, Dict

import albumentations as A
import numpy as np
import timm
import torch
from albumentations.pytorch import ToTensorV2
from ccb import io
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import (
    BackBone,
    Model,
    ModelGenerator,
    eval_metrics_generator,
    head_generator,
    test_metrics_generator,
    train_loss_generator,
    train_metrics_generator,
)
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as tt


def modify_beyond_rgb_layer(new_in_channels, current_layer, task_specs, config):
    """Modify input layer for Timm Models to work beyond RGB."""
    # Creating new Conv2d layer
    new_layer = torch.nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=current_layer.out_channels,
        kernel_size=current_layer.kernel_size,
        stride=current_layer.stride,
        padding=current_layer.padding,
    )

    # new_layer.bias.data = current_layer.bias  # type: ignore

    return _initialize_additional_in_channels(
        current_layer=current_layer, new_layer=new_layer, task_specs=task_specs, config=config
    )


def beyond_rgb(backbone, task_specs, config):
    """Modify the 1st layer to accept all the new bands instead of just RGB if required."""
    new_in_channels = len(config["dataset"]["band_names"])

    if new_in_channels == 3:
        return

    if new_in_channels < 3:
        raise ValueError(f"Expecting 3 bands or more, got: {new_in_channels}")

    # if we go beyond RGB channels need to initialize other layers, otherwise keep the same
    if config["model"]["backbone"] in ["resnet18", "resnet50"]:
        backbone.conv1 = modify_beyond_rgb_layer(new_in_channels, backbone.conv1, task_specs=task_specs, config=config)

    elif config["model"]["backbone"] in ["convnext_base"]:
        backbone.stem[0] = modify_beyond_rgb_layer(
            new_in_channels, backbone.stem[0], task_specs=task_specs, config=config
        )

    elif config["model"]["backbone"] in [
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "swinv2_tiny_window16_256",
    ]:

        backbone.patch_embed.proj = modify_beyond_rgb_layer(
            new_in_channels, backbone.patch_embed.proj, task_specs=task_specs, config=config
        )


def _initialize_additional_in_channels(
    current_layer: torch.nn.Conv2d,
    new_layer: torch.nn.Conv2d,
    task_specs: TaskSpecifications,
    config: Dict[str, Any],
) -> torch.nn.Conv2d:
    """Initialize new additional input channels.

    By default RGB channels are copied and new input channels randomly initialized

    Args:
        current_layer: current Conv2d backbone layer
        new_layer: newly initialized layer to which to copy weights
        task_specs: task specs to retrieve dataset
        config: config file for dataset specifics

    Returns:
        newly initialized input Conv2d layer
    """
    method = config["model"]["new_channel_init_method"]

    dataset = task_specs.get_dataset(
        split="train",
        band_names=config["dataset"]["band_names"],
        format=config["dataset"]["format"],
        benchmark_dir=config["experiment"]["benchmark_dir"],
        partition_name=config["experiment"]["partition_name"],
    )
    alt_band_names = dataset.alt_band_names
    band_names = dataset.band_names

    # find index of the rgb bands
    new_rgb_indices = []
    full_rgb_names = []
    for rgb_name in ["red", "green", "blue"]:
        rgb_full_name = alt_band_names[rgb_name]
        new_rgb_indices.append(band_names.index(rgb_full_name))
        full_rgb_names.append(rgb_full_name)

    non_rgb_names = list(set(band_names) - set(full_rgb_names))
    non_rgb_indices = [band_names.index(band) for band in non_rgb_names]

    # how rgb is ordered in current layer
    current_rgb_indices = [0, 1, 2]
    # need to check that this order matches with how the data of all bands is retrieved
    for new_idx, old_idx in zip(new_rgb_indices, current_rgb_indices):
        with torch.no_grad():
            new_layer.weight[:, new_idx : new_idx + 1, :, :] = current_layer.weight[:, old_idx : old_idx + 1, :, :]

    ## can define different approaches here about how to initialize other channels
    if method == "clone_random_rgb_channel":
        # Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
        for channel in non_rgb_indices:
            # index of existing channel weights
            # Here will initialize the weights in new channel by randomly cloning one pretrained channel and adding gaussian noise
            rand_rgb_idx = random.randint(0, 2)
            # find respective location of rgb pands in old and new
            current_rgb_idx = current_rgb_indices.index(rand_rgb_idx)

            with torch.no_grad():
                new_layer.weight[:, channel : channel + 1, :, :] = current_layer.weight[
                    :, current_rgb_idx : current_rgb_idx + 1, ::
                ].clone() + random.gauss(0, 1 / new_layer.in_channels)

            new_layer.weight = torch.nn.Parameter(new_layer.weight)

    return new_layer


class TIMMGenerator(ModelGenerator):
    """Timm Model Generator.

    The Timm Model Generator lets you define a model with any available
    `Timm models <https://rwightman.github.io/pytorch-image-models/>`_ as a backbone, and
    attaches a classification head according to the task. Additionally, it
    supports pretrained models from
    `TorchGeo <https://torchgeo.readthedocs.io/en/stable/api/models.html>`_.
    """

    def __init__(self) -> None:
        """Initialize a new instance of Timm model generator."""
        super().__init__()

    def generate_model(self, task_specs: TaskSpecifications, config: dict) -> Model:
        """Return a ccb.torch_toolbox.model.Model instance from task specs and hparams.

        Args:
            task_specs: object with task specs
            hparams: dictionary containing hparams
            config: dictionary containing config

        Returns:
            configured model
        """
        backbone = timm.create_model(
            config["model"]["backbone"], pretrained=config["model"]["pretrained"], features_only=False
        )
        setattr(backbone, backbone.default_cfg["classifier"], torch.nn.Identity())
        config["model"]["default_input_size"] = backbone.default_cfg["input_size"]

        beyond_rgb(backbone, task_specs, config)

        test_input_for_feature_dim = (
            len(config["dataset"]["band_names"]),
            256 if config["model"]["backbone"] == "swinv2_tiny_window16_256" else 224,
            256 if config["model"]["backbone"] == "swinv2_tiny_window16_256" else 224,
        )

        with torch.no_grad():
            backbone.eval()
            features = torch.zeros(test_input_for_feature_dim).unsqueeze(0)
            features = backbone(features)
        shapes = [tuple(features.shape[1:])]  # get the backbone's output features

        config["model"]["n_backbone_features"] = shapes[0][0]

        head = head_generator(task_specs, shapes, config)
        loss = train_loss_generator(task_specs, config)
        train_metrics = train_metrics_generator(task_specs, config)
        eval_metrics = eval_metrics_generator(task_specs, config)
        test_metrics = test_metrics_generator(task_specs, config)

        return Model(
            backbone=backbone,
            head=head,
            loss_function=loss,
            config=config,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            test_metrics=test_metrics,
        )

    def get_transform(
        self, task_specs, config: Dict[str, Any], train=True, scale=None, ratio=None
    ) -> Callable[[io.Sample], Dict[str, Any]]:
        """Define data transformations specific to the models generated.

        Args:
            task_specs: task specs to retrieve dataset
            config: config file for dataset specifics
            train: train mode true or false
            scale: define image scale
            ratio: define image ratio range

        Returns:
            callable function that applies transformations on input data
        """
        mean, std = task_specs.get_dataset(
            split="train",
            format=config["dataset"]["format"],
            band_names=tuple(config["dataset"]["band_names"]),
            benchmark_dir=config["experiment"]["benchmark_dir"],
            partition_name=config["experiment"]["partition_name"],
        ).normalization_stats()

        desired_input_size = config["model"]["default_input_size"][1]

        t = []
        if train:
            t.append(A.RandomRotate90(0.5))
            t.append(A.HorizontalFlip(0.5))
            t.append(A.VerticalFlip(0.5))
            t.append(A.Transpose(0.5))

        t.append(A.Resize(desired_input_size, desired_input_size))

        # max_pixel_value = 1 is essential for us
        t.append(A.Normalize(mean=mean, std=std, max_pixel_value=1))
        t.append(ToTensorV2())
        transform_comp = A.Compose(t)

        def transform(sample: io.Sample):
            x: "np.typing.NDArray[np.float_]" = sample.pack_to_3d(band_names=config["dataset"]["band_names"])[0].astype(
                "float32"
            )
            x = transform_comp(image=x)["image"]
            return {"input": x, "label": sample.label}

        return transform

    def generate_model_name(self, config: Dict[str, Any]) -> str:
        """Generate a model name that can be used throughout to the pipeline.

        Args:
            config: config file
        """
        model_name = config["model"]["backbone"]
        if not config["model"]["pretrained"]:
            model_name = "scratch_" + model_name
        return model_name


def model_generator() -> TIMMGenerator:
    """Initializ Timm Generator.

    Returns:
        segmentation model generator
    """
    return TIMMGenerator()