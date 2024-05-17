import os
import random
import string

from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger


def generate_trainer(config: dict, job) -> Trainer:
    """Configure a pytroch lightning Trainer.

    Args:
        config: dictionary containing config
        job: job being executed to let logger know directory

    Returns:
        lightning Trainer with configurations from config file.
    """
    run_id = "".join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(8))

    loggers = [
        CSVLogger(str(job.dir), name="csv_logs"),
    ]
    if "wandb" in config:
        config["wandb"]["wandb_run_id"] = run_id
        loggers.append(
            WandbLogger(
                save_dir=str(job.dir),
                project=config["wandb"]["project"],
                entity=config["wandb"]["entity"],
                id=run_id,
                group=config["wandb"].get("wandb_group", None),
                name=config["wandb"].get("name", None),
                resume="allow",
                config=config["model"],
                mode=config["wandb"].get("mode", "online"),
            )
        )

    job.save_config(config, overwrite=True)

    ckpt_dir = os.path.join(job.dir, "checkpoint")

    ds_name = job.task_specs.dataset_name

    if ds_name in [
        "m-eurosat",
        "m-brick-kiln",
        "m-pv4ger",
        "m-so2sat",
        "m-forestnet",
    ]:
        track_metric = "val_Accuracy"
        mode = "max"
    elif ds_name == "m-bigearthnet":
        track_metric = "val_F1Score"
        mode = "max"
    elif ds_name in [
        "m-pv4ger-seg",
        "m-nz-cattle",
        "m-SA-crop-type",
        "m-seasonet",
        "m-chesapeake-landcover",
        "m-NeonTree",
        "m-cashew-plantation",
    ]:
        track_metric = "val_Jaccard"
        mode = "max"

    if "early_stopping_metric" in config["model"]:
        track_metric = config["model"]["early_stopping_metric"]
        mode = "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir, save_top_k=1, monitor=track_metric, mode=mode, every_n_epochs=1
    )
    patience = int((1 / config["trainer"]["val_check_interval"]) * (config["trainer"]["max_epochs"] / 6))
    early_stopping_callback = EarlyStopping(
        monitor=track_metric,
        mode=mode,
        patience=patience,
        min_delta=1e-5,
    )

    trainer = instantiate(
        config.trainer,
        default_root_dir=job.dir,
        callbacks=[
            early_stopping_callback,
            checkpoint_callback,
        ],
        logger=loggers,
    )

    return trainer
