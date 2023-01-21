"""Generate experiment directory structure.

Usage: experiment_generator.py --model-generator path/to/my/model/generator.py  --experiment-dir path/to/my/experiments
"""
import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from ccb import io
from ccb.experiment.experiment import Job, get_model_generator


def experiment_generator(
    config_filepath: str,
) -> Path:
    """Generate the directory structure for every tasks.

    According to model_generator.hp_search.

    Args:
        config_filepath: path to config file that defines experiment
    Returns:
        Name of the experiment directory.

    Raises:
        FileNotFoundError if path to config file or hparam file does not exist
    """
    config_file_path = Path(config_filepath)

    # check that specified paths exists
    if config_file_path.is_file():
        with config_file_path.open() as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file at path {config_file_path} does not exist.")

    benchmark_dir = config["experiment"]["benchmark_dir"]

    experiment_prefix = f"{config['experiment']['experiment_name'] or 'experiment'}_{os.path.basename(benchmark_dir)}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    if config["experiment"]["experiment_name"] is not None:
        experiment_dir: Path = Path(config["experiment"]["generate_experiment_dir"]) / experiment_prefix
    else:
        experiment_dir: Path = Path(config["experiment"]["generate_experiment_dir"])  # type: ignore[no-redef]

    for task_specs in io.task_iterator(
        benchmark_dir=benchmark_dir, task_filter=config["experiment"].get("tasks", None)
    ):
        print(task_specs.dataset_name)
        experiment_type = config["experiment"]["experiment_type"]
        task_config = copy.deepcopy(config)

        if (
            experiment_type == "sweep"
        ):  # TODO(nils) sweep and seeded_runs code should be the same with a flag making minor modifications.

            print(len(task_config["dataset"]["band_names"]))
            model_generator = get_model_generator(config["model"]["model_generator_module_name"])

            # use wandb sweep for hyperparameter search
            model = model_generator.generate_model(task_specs, task_config)

            # there might be other params added during the generate process,
            # continue with hyperparameters from initialized model
            task_config = model.config
            task_config["model"]["model_name"] = model_generator.generate_model_name(config)

            # append model name to experiment dir
            experiment_dir = Path(str(experiment_dir) + f"_{task_config['model']['model_name']}")
            # create and fill experiment directory
            job_dir = experiment_dir / task_specs.dataset_name
            job = Job(job_dir)
            job.save_config(task_config)
            job.save_task_specs(task_specs)

            # sweep name that will be seen on wandb
            wandb_name = "_".join(str(job_dir).split("/")[-2:]) + "_" + task_config["model"]["model_name"]

            job.write_wandb_sweep_cl_script(
                task_config["model"]["model_generator_module_name"],
                job_dir=job_dir,
                base_sweep_config=task_config["wandb"]["sweep"]["sweep_config_path"],
                name=wandb_name,
            )

        elif experiment_type == "seeded_runs":

            with open(config["experiment"]["best_hparam_config"], "r") as f:
                seed_run_dict = json.load(f)

            part_name = task_config["experiment"]["partition_name"].split("_partition.json")[0]

            model_generator = get_model_generator(config["model"]["model_generator_module_name"])

            # use wandb sweep for hyperparameter search
            model = model_generator.generate_model(task_specs, task_config)

            model_name = model.generate_model_name(task_config)

            ds_dict = seed_run_dict[model_name][part_name]
            exp_dir = ds_dict[task_specs.dataset_name]

            # load best config file
            with open(os.path.join(exp_dir, "config.yaml")) as f:
                best_config = yaml.safe_load(f)

            best_config["wandb"]["wandb_group"] = task_specs.dataset_name + "/" + model_name + "/" + experiment_prefix

            for i in range(config["experiment"]["num_seeds"]):
                # set seed to be used in experiment
                best_config["experiment"]["seed"] = i

                job_dir = experiment_dir / task_specs.dataset_name / f"seed_{i}"
                job = Job(job_dir)
                job.save_config(best_config)
                job.save_task_specs(task_specs)
                job.write_script(job_dir=str(job_dir))

    return experiment_dir


def start() -> None:
    """Start generating experiments."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="experiment_generator.py",
        description="Generate experiment directory structure based on user-defined model generator",
    )
    parser.add_argument(
        "--config_filepath",
        help="The path to the configuration file.",
        required=True,
    )

    args = parser.parse_args()

    experiment_generator(config_filepath=args.config_filepath)


if __name__ == "__main__":
    start()
