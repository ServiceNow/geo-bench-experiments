"""Segmentation Model Generator."""

from typing import Any, Dict

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.functional as TF
from albumentations.pytorch.transforms import ToTensorV2
from ccb import io
from ccb.io.dataset import Band
from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.model import (
    Model,
    ModelGenerator,
    eval_metrics_generator,
    test_metrics_generator,
    train_loss_generator,
    train_metrics_generator,
)
from ccb.torch_toolbox.modules import ClassificationHead
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as tt


class SegmentationGenerator(ModelGenerator):
    """SegmentationGenerator.

    This ModelGenerator uses
    `segmentation_models.pytorch <https://github.com/qubvel/segmentation_models.pytorch>`_
    as an architecture choice and allows any of these
    `TIMM encoders <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_
    """

    def __init__(self) -> None:
        """Initialize a new instance of segmentation generator.

        Args:
            hparams: set of hyperparameters

        """
        super().__init__()

    def generate_model(self, task_specs: TaskSpecifications, config: dict) -> Model:
        """Return model instance from task specs and hyperparameters.

        Args:
            task_specs: object with task specs
            config: config: dictionary containing config

        Returns:
            model specified by task specs and hyperparameters
        """
        encoder_type = config["model"]["encoder_type"]
        decoder_type = config["model"]["decoder_type"]
        encoder_weights = config["model"].get("encoder_weights", None)
        # in_ch, *other_dims = features_shape[-1]
        out_ch = task_specs.label_type.n_classes

        # Load segmentation backbone from py-segmentation-models
        backbone = getattr(smp, decoder_type)(
            encoder_name=encoder_type,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=len(
                config["dataset"]["band_names"]
            ),  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=out_ch,
        )  # model output channels (number of classes in your dataset))

        # For timm models, we can extract the mean and std of the pre-trained backbone
        # hparams.update({"mean": backbone.default_cfg["mean"]})
        # hparams.update({"std": backbone.default_cfg["std"]})
        config["model"]["input_size"] = (
            len(config["dataset"]["band_names"]),
            config["model"]["desired_input_size"],
            config["model"]["desired_input_size"],
        )

        with torch.no_grad():
            backbone.eval()
            features = torch.zeros(config["model"]["input_size"]).unsqueeze(0)
            features = backbone.encoder(features)

        class Noop(torch.nn.Module):
            def forward(self, x):
                return x

        head = ClassificationHead(
            num_classes=1, in_ch=1, ret_identity=True
        )  # pytorch image models already adds a classifier on top of the UNETs
        # head = head_generator(task_specs, shapes, hparams)
        loss = train_loss_generator(task_specs, config)
        train_metrics = train_metrics_generator(task_specs, config)
        eval_metrics = eval_metrics_generator(task_specs, config)
        test_metrics = test_metrics_generator(task_specs, config)
        return Model(backbone, head, loss, config, train_metrics, eval_metrics, test_metrics)

    def get_transform(
        self,
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
            benchmark_dir=config["experiment"]["benchmark_dir"],
            partition_name=config["experiment"]["partition_name"],
        ).rgb_stats()
        band_names = config["dataset"]["band_names"]

        t = []
        if h < patch_h:
            t.append(A.SmallestMaxSize(max_size=h))
        if patch_h < h32:
            t.append(A.Resize(h32, w32))

        t.append(A.RandomCrop(h32, w32))
        if train:
            t.append(A.RandomRotate90(0.5))
            t.append(A.HorizontalFlip(0.5))
            t.append(A.VerticalFlip(0.5))
            t.append(A.Transpose(0.5))

        # max_pixel_value = 1 is essential for us
        t.append(A.Normalize(mean=mean, std=std, max_pixel_value=1))
        t.append(ToTensorV2())
        t_comp = A.Compose(t)

        def transform(sample: io.Sample):
            x = sample.pack_to_3d(band_names=band_names)[0].astype("float32")

            if isinstance(sample.label, Band):
                x, y = x, sample.label.data.astype("float32")
                transformed = t_comp(image=x, mask=y)

            return {"input": transformed["image"], "label": transformed["mask"].squeeze(-1).long()}

        return transform

    def generate_model_name(self, config: Dict[str, Any]) -> str:
        """Generate a model name that can be used throughout to the pipeline.

        Args:
            config: config file
        """
        model_name = config["model"]["encoder_type"] + "_" + config["model"]["decoder_type"]
        if config["model"]["pretrained"] is False:
            model_name = "scratch_" + model_name
        return model_name


def model_generator() -> SegmentationGenerator:
    """Initialize Segmentation Generator.

    Returns:
        segmentation model generator
    """
    return SegmentationGenerator()