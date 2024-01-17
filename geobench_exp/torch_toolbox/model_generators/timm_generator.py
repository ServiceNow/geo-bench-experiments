"""Timm Model Generator."""

import random
from typing import Any, Callable, Dict

import kornia.augmentation as K
import numpy as np
import timm
import torch
from geobench.dataset import Sample
from geobench.task import TaskSpecifications
from kornia.augmentation import ImageSequential
from torchgeo.models import get_weight
from torchgeo.trainers.utils import load_state_dict

from geobench_exp.torch_toolbox.model import (
    Model,
    ModelGenerator,
    eval_metrics_generator,
    head_generator,
    train_loss_generator,
)


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
        """Return a geobench_exp.torch_toolbox.model.Model instance from task specs and hparams.

        Args:
            task_specs: object with task specs
            hparams: dictionary containing hparams
            config: dictionary containing config

        Returns:
            configured model
        """
        backbone = timm.create_model(
            config["model"]["backbone"],
            pretrained=config["model"]["pretrained"],
            features_only=False,
            in_chans=config["model"]["in_chans"],
        )
        weight_name = config["model"].get("weights", None)
        if weight_name is not None:
            weights = get_weight(weight_name)
            state_dict = weights.get_state_dict(progress=True)
            backbone = load_state_dict(backbone, state_dict)

        setattr(backbone, backbone.default_cfg["classifier"], torch.nn.Identity())
        config["model"]["default_input_size"] = backbone.default_cfg["input_size"]

        test_input_for_feature_dim = (
            config["model"]["in_chans"],
            256 if config["model"]["backbone"] == "swinv2_tiny_window16_256" else 224,
            256 if config["model"]["backbone"] == "swinv2_tiny_window16_256" else 224,
        )

        with torch.no_grad():
            backbone.eval()
            features = torch.zeros(test_input_for_feature_dim).unsqueeze(0)
            features = backbone(features)
        shapes = [tuple(features.shape[1:])]  # get the backbone's output features

        config["model"]["n_backbone_features"] = shapes[0][0]

        head = head_generator(task_specs, shapes)
        loss = train_loss_generator(task_specs, config)

        return Model(
            backbone=backbone,
            head=head,
            loss_function=loss,
            config=config,
            train_metrics=eval_metrics_generator(task_specs, config),
            eval_metrics=eval_metrics_generator(task_specs, config),
            test_metrics=eval_metrics_generator(task_specs, config),
        )

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
