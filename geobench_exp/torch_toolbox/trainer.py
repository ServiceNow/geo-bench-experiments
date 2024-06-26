#!/usr/bin/env python
"""Train the model using job information contained in the current directory."""

import argparse
import os

from dataset import get_transform
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from model_utils import generate_trainer
from ruamel.yaml import YAML
from torch.utils.data.dataloader import default_collate

from geobench_exp.experiment.experiment import Job


def train(job_dir: str) -> None:
    """Train a model from the model generator on datamodule.

    Args:
        job_dir: job directory that contains task_specs and hparams.json
    """
    job = Job(job_dir)
    config = job.config
    task_specs = job.task_specs
    seed = config["model"].get("seed", None)
    if seed is not None:
        seed_everything(seed, workers=True)

    if config["dataset"]["band_names"] == "all":
        config["dataset"]["band_names"] = [band_info.name for band_info in task_specs.bands_info]

    # Load the user-specified model generator
    # TODO should also be done with hydra
    model = instantiate(config.model, task_specs=task_specs)

    # TODO this should be done with hydra
    trainer = generate_trainer(config=config, job=job)

    # load config new because there might have been additions in generate_trainer function
    config = job.config

    # TODO this should be done with hydra
    datamodule = instantiate(
        config.datamodule,
        task_specs=task_specs,
        train_transform=get_transform(task_specs=task_specs, config=config, train=True),
        eval_transform=get_transform(task_specs=task_specs, config=config, train=False),
        collate_fn=default_collate,
    )
    # datamodule = DataModule(
    #     task_specs=task_specs,
    #     benchmark_dir=config["experiment"]["benchmark_dir"],
    #     partition_name=config["experiment"]["partition_name"],
    #     batch_size=config["dataloader"]["batch_size"],
    #     num_workers=config["dataloader"]["num_workers"],
    #     train_transform=model_gen.get_transform(task_specs=task_specs, config=config, train=True),
    #     eval_transform=model_gen.get_transform(task_specs=task_specs, config=config, train=False),
    #     collate_fn=default_collate,
    #     band_names=config["dataset"]["band_names"],
    #     format=config["dataset"]["format"],
    # )

    trainer.log_every_n_steps = min(len(datamodule.train_dataloader()), config["pl"]["log_every_n_steps"])  # type: ignore[attr-defined]

    ckpt_path = config["model"].get("ckpt_path", None)
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    trainer.test(model, datamodule)

    # save updated configs in csv_logger one directories are created
    # TODO this should be done with OmegaConf
    yaml = YAML()
    with open(os.path.join(trainer.loggers[0].log_dir, "config.yaml"), "w") as fd:
        yaml.dump(config, fd)

    with open(os.path.join(trainer.loggers[0].log_dir, "status.txt"), "w") as fd:
        fd.write("Done")


def start() -> None:
    """Start training."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="trainer.py", description="Trains the model using job information contained in the current directory."
    )

    parser.add_argument("--job_dir", help="Path to the job.", required=True)

    args = parser.parse_args()

    train(args.job_dir)


if __name__ == "__main__":
    start()
