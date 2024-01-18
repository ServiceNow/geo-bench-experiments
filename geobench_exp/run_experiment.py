import argparse

import os

from geobench_exp.torch_toolbox.dataset import get_transform
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from geobench_exp.torch_toolbox.model_utils import generate_trainer
from torch.utils.data.dataloader import default_collate

from geobench_exp.experiment.experiment import Job
from omegaconf import OmegaConf


def run(job_dir: str) -> None:
    """Run the experiment.
    
    Args:
        job_dir: job directory that contains task_specs
    """
    job = Job(job_dir)
    config = OmegaConf.create(job.config)
    task_specs = job.task_specs

    seed = config["model"].get("seed", None)
    if seed is not None:
        seed_everything(seed, workers=True)

    if config["datamodule"]["band_names"] == "all":
        config["datamodule"]["band_names"] = [band_info.name for band_info in task_specs.bands_info]

    # instantiate the model
    model = instantiate(config.model, task_specs=task_specs)

    # instantiate the trainer
    trainer = generate_trainer(config=config, job=job)
    # load config new because there might have been additions in generate_trainer function
    config = OmegaConf.create(job.config)

    datamodule = instantiate(
        config.datamodule,
        task_specs=task_specs,
        benchmark_dir=config["experiment"]["benchmark_dir"],
        partition_name=config["experiment"]["partition_name"],
        train_transform=get_transform(task_specs=task_specs, config=config, train=True),
        eval_transform=get_transform(task_specs=task_specs, config=config, train=False),
        collate_fn=default_collate,
    )

    # for small partition sizes
    trainer.log_every_n_steps = min(len(datamodule.train_dataloader()), config["trainer"]["log_every_n_steps"])  # type: ignore[attr-defined]

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)

    # save config and status
    OmegaConf.save(config, os.path.join(trainer.loggers[0].log_dir, "config.yaml"))
    with open(os.path.join(trainer.loggers[0].log_dir, "status.txt"), "w") as fd:
        fd.write("Done")


def start() -> None:
    """Start training."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="run_experiment.py", description="Trains the model using job information contained in the specified directory."
    )

    parser.add_argument("--job_dir", help="Path to the job.", required=True)

    args = parser.parse_args()

    run(args.job_dir)


if __name__ == "__main__":
    start()