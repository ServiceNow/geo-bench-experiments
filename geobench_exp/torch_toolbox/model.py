"""Model."""

import os
import time
from typing import Callable, Dict, List, Optional, Union

import lightning
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn.functional as F
import torchmetrics
from geobench.dataset import SegmentationClasses
from geobench.label import Classification, MultiLabelClassification
from geobench.task import TaskSpecifications
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor
from torchgeo.models import get_weight
from torchgeo.trainers import utils
from torchvision.models._api import WeightsEnum


class GeoBenchBaseModule(LightningModule):
    """GeoBench Base Lightning Module."""

    def __init__(
        self,
        task_specs: TaskSpecifications,
        in_channels,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        """Initialize a new ClassificationTask instance.

        Args:
            task_specs: an object describing the task to be performed
            model: Name of the `timm
                <https://huggingface.co/docs/timm/reference/models>`__ model to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            freeze_backbone: Freeze the backbone network to linear probe
                the classifier head.
            optimizer: Optimizer to use for training
            lr_scheduler: Learning rate scheduler to use for training
        """
        super().__init__()
        self.task_specs = task_specs

        self.loss_fn = train_loss_generator(task_specs)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_metrics = eval_metrics_generator(task_specs)
        self.eval_metrics = eval_metrics_generator(task_specs)
        self.test_metrics = eval_metrics_generator(task_specs)

        self.configure_the_model()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (N, C, H, W)

        Returns:
            Tensor (N, num_classes)
        """
        return self.model(x)

    def configure_the_model(self) -> None:
        """Initialize the model."""
        raise NotImplementedError("Necessary to define a model.")

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx=0) -> Dict[str, Tensor]:  # type: ignore
        """Define steps taken during training mode.

        Args:
            batch: input batch
            batch_idx: index of batch

        Returns:
            training step outputs
        """
        inputs, target = batch["input"], batch["label"]
        output = self(inputs)
        loss_train = self.loss_fn(output, target)
        self.train_metrics(output, target)
        self.log("train_loss", loss_train, logger=True)
        self.log("current_time", time.time(), logger=True)

        return loss_train

    def on_train_epoch_end(self, *arg, **kwargs) -> None:  # type: ignore
        """Define actions after a training epoch.

        Args:
            outputs: outputs from :meth:`__training_step`
        """
        self.log_dict({f"train_{k}": v.mean() for k, v in self.train_metrics.compute().items()}, logger=True)
        self.train_metrics.reset()

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int):
        """Define steps taken during validation mode.

        Args:
            batch: input batch
            batch_idx: index of batch
            dataloader_idx: index of dataloader

        Returns:
            validation step outputs
        """
        self.prefix = ["val", "test"][dataloader_idx]
        inputs, target = batch["input"], batch["label"]
        output = self(inputs)
        loss = self.loss_fn(output, target)
        self.log(f"{self.prefix}_loss", loss)
        if self.prefix == "val":
            self.eval_metrics(output, target)
        else:
            self.test_metrics(output, target)

        return loss

    def on_validation_epoch_end(self):
        """Define actions after a validation epoch."""

        # eval metrics
        eval_metrics = self.eval_metrics.compute()
        self.log_dict({f"val_{k}": v.mean() for k, v in eval_metrics.items()}, logger=True)
        self.eval_metrics.reset()

        # test metrics
        test_metrics = self.test_metrics.compute()
        self.log_dict({f"test_{k}": v.mean() for k, v in test_metrics.items()}, logger=True)
        self.test_metrics.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Define steps taken during test mode.

        Args:
            batch: input batch
            batch_idx: index of batch

        Returns:
            test step outputs
        """
        inputs, target = batch["input"], batch["label"]
        output = self(inputs)
        loss = self.loss_function(output, target)
        self.log("test_loss", loss)
        self.test_metrics(output, target)
        return loss

    def on_test_epoch_end(self, *arg, **kwargs):
        """Define actions after a test epoch.

        Args:
            outputs: outputs from :meth:`__test_step`
        """
        test_metrics = self.test_metrics.compute()
        self.log_dict({f"test_{k}": v.mean() for k, v in test_metrics.items()}, logger=True)
        self.test_metrics.reset()

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class GeoBenchClassifier(GeoBenchBaseModule):
    def __init__(
        self,
        task_specs: TaskSpecifications,
        model: str,
        in_channels: int,
        weights: Union[WeightsEnum, str, bool, None] = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        self.save_hyperparameters(ignore=["loss_fn", "task_specs"])
        self.hparams["model"] = model
        self.weights = weights
        super().__init__(task_specs, in_channels, freeze_backbone, optimizer, lr_scheduler)

    def configure_the_model(self) -> None:
        """Configure classification model."""
        # Create model
        self.model = timm.create_model(
            self.hparams["model"],
            num_classes=self.task_specs.label_type.n_classes,
            in_chans=self.hparams["in_channels"],
            pretrained=self.weights is True,
        )

        # Load weights
        if self.weights and self.weights is not True:
            if isinstance(self.weights, WeightsEnum):
                state_dict = self.weights.get_state_dict(progress=True)
            elif os.path.exists(self.weights):
                _, state_dict = utils.extract_backbone(self.weights)
            else:
                state_dict = get_weight(self.weights).get_state_dict(progress=True)

            utils.load_state_dict(self.model, state_dict)

        # Freeze backbone and unfreeze classifier head
        if self.hparams["freeze_backbone"]:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True


class GeoBenchSegmentation(GeoBenchBaseModule):
    def __init__(
        self,
        task_specs: TaskSpecifications,
        encoder_type: str,
        decoder_type: str,
        in_channels: int,
        encoder_weights: Union[WeightsEnum, str, bool, None] = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        self.save_hyperparameters(ignore=["loss_fn", "task_specs"])

        super().__init__(task_specs, in_channels, freeze_backbone, optimizer, lr_scheduler)

    def configure_the_model(self) -> None:
        """Configure segmentation model."""
        # Load segmentation backbone from py-segmentation-models
        self.model = getattr(smp, self.hparams["decoder_type"])(
            encoder_name=self.hparams["encoder_type"],
            encoder_weights=self.hparams["encoder_weights"],
            in_channels=self.hparams["in_channels"],
            classes=self.task_specs.label_type.n_classes,
        )  # model output channels (number of cl


def eval_metrics_generator(task_specs: TaskSpecifications) -> List[torchmetrics.MetricCollection]:
    """Return the appropriate eval function depending on the task_specs.

    Args:
        task_specs: an object describing the task to be performed
        hyperparams: dictionary containing hyperparameters of the experiment

    Returns:
        metric collection used during evaluation
    """
    metrics: List[torchmetrics.MetricCollection] = {  # type: ignore
        Classification: torchmetrics.MetricCollection(
            {"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=task_specs.label_type.n_classes)}
        ),
        SegmentationClasses: torchmetrics.MetricCollection(
            {
                "Jaccard": torchmetrics.JaccardIndex(task="multiclass", num_classes=task_specs.label_type.n_classes),
                "FBeta": torchmetrics.FBetaScore(
                    task="multiclass",
                    num_classes=task_specs.label_type.n_classes,
                    beta=2.0,
                    multidim_average="samplewise",
                ),
            }
        ),
        MultiLabelClassification: torchmetrics.MetricCollection(
            {"F1Score": torchmetrics.F1Score(task="multilabel", num_labels=task_specs.label_type.n_classes)}
        ),
    }[task_specs.label_type.__class__]

    return metrics


def _balanced_binary_cross_entropy_with_logits(outputs: Tensor, targets: Tensor) -> Tensor:
    """Compute balance binary cross entropy for multi-label classification.

    Args:
        outputs: model outputs
        targets: targets to compute binary cross entropy on
    """
    classes = targets.shape[-1]
    outputs = outputs.view(-1, classes)
    targets = targets.view(-1, classes).float()
    loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    loss = loss[targets == 0].mean() + loss[targets == 1].mean()
    return loss


def train_loss_generator(task_specs: TaskSpecifications) -> Callable[[Tensor], Tensor]:
    """Return the appropriate loss function depending on the task_specs.

    Args:
        task_specs: an object describing the task to be performed

    Returns:
        available loss functions for training
    """
    loss = {
        Classification: F.cross_entropy,
        MultiLabelClassification: _balanced_binary_cross_entropy_with_logits,
        SegmentationClasses: F.cross_entropy,
    }[task_specs.label_type.__class__]

    return loss  # type: ignore
