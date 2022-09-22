"""SSL MOCO pretrained model from ZhuLab."""

import copy
import random
from typing import Any, Callable, Dict

import numpy as np
import timm
import torch
from torch.utils.data.dataloader import default_collate
from torchvision import models
from torchvision import transforms as tt

from ccb import io
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import (
    Model,
    ModelGenerator,
    eval_metrics_generator,
    head_generator,
    test_metrics_generator,
    train_loss_generator,
    train_metrics_generator,
)


class SSLMocoGenerator(ModelGenerator):
    """SSL Moco Generator.

    SSL Moco models on RGB data pretrained taken from:
    `Zhu Lab <https://github.com/zhu-xlab/SSL4EO-S12>`_.
    """

    def __init__(self) -> None:
        """Initialize a new instance of SSL MOCO model."""
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
        # this part comes from model loading from their script
        if "resnet50" in config["model"]["backbone"]:
            backbone = models.resnet50(weights=None)
            backbone.fc = torch.nn.Linear(2048, 19)
            shapes = [(2048, 1, 1)]
            ckpt_path = "/mnt/data/experiments/nils/ssl_checkpoints_zhu/B3_rn50_moco_0099_ckpt.pth"
        elif "resnet18" in config["model"]["backbone"]:
            backbone = models.resnet18(weights=None)
            backbone.fc = torch.nn.Linear(512, 19)
            shapes = [(512, 1, 1)]
            ckpt_path = "/mnt/data/experiments/nils/ssl_checkpoints_zhu/B3_rn18_moco_0199_ckpt.pth"

        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint["state_dict"]

        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
                # pdb.set_trace()
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        previous_backbone = copy.deepcopy(backbone)
        prev_sample_weight = previous_backbone.layer4[0].conv1.weight
        backbone.load_state_dict(state_dict, strict=False)

        post_sample_weight = backbone.layer4[0].conv1.weight

        # make sure there are new weights loaded
        assert torch.equal(prev_sample_weight, post_sample_weight) is False

        # replace head for the task at hand
        backbone.fc = torch.nn.Identity()
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

    def get_collate_fn(self, task_specs: TaskSpecifications, config: dict):
        """Define a collate function to batch input tensors.

        Args:
            task_specs: task specs to retrieve dataset
            config: model hyperparameters

        Returns:
            collate function
        """
        return default_collate

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
        t = []
        t.append(tt.ToTensor())
        t.append(tt.Normalize(mean=mean, std=std))
        if train:
            t.append(tt.RandomApply(torch.nn.ModuleList([tt.RandomRotation((90, 90))]), p=0.5))
            t.append(tt.RandomHorizontalFlip())
            t.append(tt.RandomVerticalFlip())
            t.append(tt.ColorJitter(0.1))
            t.append(tt.RandomGrayscale(0.1))

        # all convolutional architectures
        if task_specs.patch_size[0] <= 224:
            t.append(tt.Resize((224, 224)))
        elif task_specs.patch_size[0] > 224:
            t.append(tt.RandomCrop((224, 224)))

        transform_comp = tt.Compose(t)

        def transform(sample: io.Sample):
            x: "np.typing.NDArray[np.float_]" = sample.pack_to_3d(band_names=config["dataset"]["band_names"])[0].astype(
                "float32"
            )
            x = transform_comp(x)
            return {"input": x, "label": sample.label}

        return transform


def model_generator() -> SSLMocoGenerator:
    """Return SSL Moco Generator.

    Returns:
        SSL Moco model generator
    """
    return SSLMocoGenerator()
