"""Seco Model Generator."""

import random
from argparse import ArgumentParser
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision
from albumentations.pytorch import ToTensorV2
from pl_bolts.metrics import precision_at_k
from pytorch_lightning import LightningModule
from torch import nn, optim
from torchvision import transforms as tt

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


class SeCoGenerator(ModelGenerator):
    """SeCo Model Generator.

    The SeCo Model Generator lets you define a Resnet18 or 50 architecture
    pretrained with the seasonal contrast method introduced in this
    `paper <https://arxiv.org/abs/2103.16607>`_.
    """

    def __init__(self) -> None:
        """Initialize a new instance of SeCo model generator."""
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
        if "resnet50" in config["model"]["backbone"]:
            ckpt_path = "/mnt/data/experiments/nils/seco_weights/seco_resnet50_1m_encoder.ckpt"
        elif "resnet18" in config["model"]["backbone"]:
            ckpt_path = "/mnt/data/experiments/nils/seco_weights/seco_resnet18_1m_encoder.ckpt"
        else:
            raise ValueError("Only support 'resnet18' or 'resnet50' as model.")

        backbone = timm.create_model(
            config["model"]["backbone"], pretrained=config["model"]["pretrained"], features_only=False
        )
        setattr(backbone, backbone.default_cfg["classifier"], torch.nn.Identity())
        backbone.load_state_dict(torch.load(ckpt_path))

        test_input_for_feature_dim = (len(config["dataset"]["band_names"]), 224, 224)

        with torch.no_grad():
            backbone.eval()
            features = torch.zeros(test_input_for_feature_dim).unsqueeze(0)
            features = backbone(features)
        shapes = [tuple(features.shape[1:])]  # get the backbone's output features

        # replace head for the task at hand
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

    def get_transform(self, task_specs, config: Dict[str, Any], train=True) -> Callable[[io.Sample], Dict[str, Any]]:
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
            format=config["dataset"]["format"],
            band_names=tuple(config["dataset"]["band_names"]),
            benchmark_dir=config["experiment"]["benchmark_dir"],
            partition_name=config["experiment"]["partition_name"],
        ).normalization_stats()
        t = []
        if train:
            t.append(A.RandomRotate90(0.5))
            t.append(A.HorizontalFlip(0.5))
            t.append(A.VerticalFlip(0.5))
            t.append(A.Transpose(0.5))

        t.append(A.Resize(224, 224))

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
        return "seco_" + config["model"]["backbone"]


def model_generator() -> SeCoGenerator:
    """Return SeCo Generator.

    Returns:
        SeCo model generator
    """
    return SeCoGenerator()
