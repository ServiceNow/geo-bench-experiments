"""Experiment."""

import csv
import glob
import json
import os
import pickle
import stat
import sys
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Union

from geobench.task import TaskSpecifications
from omegaconf import OmegaConf


class Job:
    """Job.

    Helper class to organize running of experiments.
    """

    def __init__(self, dir: Union[str, Path]) -> None:
        """Initialize new instance of Job.

        Args:
            dir: path to directory where job should be created
        """
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def task_specs(self) -> TaskSpecifications:
        """Return task specifications."""
        with open(self.dir / "task_specs.pkl", "rb") as fd:
            load_task_specs: TaskSpecifications = pickle.load(fd)
            return load_task_specs

    def save_task_specs(self, task_specs: TaskSpecifications, overwrite: bool = False) -> None:
        """Save task specifications in job directory.

        Args:
            task_specs: task specifications
            overwrite: whether to overwrite existing task specs
        """
        task_specs.save(str(self.dir), overwrite=overwrite)

    @cached_property
    def config(self) -> Dict[str, Any]:
        """Return config file."""
        loaded_config = OmegaConf.load(str(self.dir / "config.yaml"))
        return loaded_config

    def save_config(self, config: Dict[str, Any], overwrite: bool = False) -> None:
        """Save config file in job directory.

        Args:
            config: config file for experiment logistics and training
            overwrite: whether to overwrite existing config file
        """
        config_path = self.dir / "config.yaml"
        if config_path.exists() and not overwrite:
            raise Exception("config already exists and overwrite is set to False.")
        OmegaConf.save(config=config, f=str(config_path))
        self.config = config

    def get_metrics(self) -> Dict[str, Any]:
        """Retrieve the metrics after training from job directory."""
        if "wandb" in self.config["experiment"].get("loggers", ""):
            import wandb

            wandb.finish()
            summary = glob.glob(str(self.dir / "wandb" / "latest-run" / "*" / "wandb-summary.json"))

            with open(summary[0], "r") as infile:
                data: Dict[str, Any] = json.load(infile)
            return data
        else:
            try:
                with open(self.dir / "csv_logs" / "version_0" / "metrics.csv", "r") as fd:
                    data: Dict[str, Any] = {}  # type: ignore[no-redef]
                    # FIXME: This would be more efficient if done backwards
                    for entry in csv.DictReader(fd):
                        data.update({k: v for k, v in entry.items() if v != ""})
                return data
            except FileNotFoundError as e:
                stderr = self.get_stderr()
                if stderr is not None:
                    raise Exception(stderr)
                else:
                    raise e

    def write_script(self, job_dir: str) -> None:
        """Write bash scrip that can be executed to run job.

        Args:
            job_dir: job directory from which to run job
        """
        script_path = self.dir / "run.sh"
        with open(script_path, "w") as fd:
            fd.write("#!/bin/bash\n")
            fd.write("# Usage: sh run.sh path/to/model_generator.py\n\n")
            fd.write(f"geobench_exp-run_exp --job_dir {job_dir}")
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    def write_wandb_sweep_cl_script(
        self, model_generator_module_name: str, job_dir: Union[str, Path], base_sweep_config: str, name: str
    ) -> None:
        """Write final sweep_config.yaml that can be used to initialize sweep.

        Args:
            model_generator_module_name: what model_generator to use
            job_dir: job directory from which to run job
            base_sweep_config: path to base sweep config yaml file for wandb
            name: wandb sweep experiment name
        """
        yaml = YAML()
        with open(base_sweep_config, "r") as yamlfile:
            base_yaml = yaml.load(yamlfile)

        base_yaml["command"] = [  # commands needed to run actual training script
            "${program}",
            "--job_dir",
            str(job_dir),
        ]
        config = Job(job_dir).config

        base_yaml["name"] = config["model"]["model_name"]

        save_path = os.path.join(job_dir, "sweep_config.yaml")
        yaml.indent(sequence=4, offset=2)
        with open(save_path, "w") as yamlfile:
            yaml.dump(base_yaml, yamlfile)

    def get_stderr(self) -> Union[str, None]:
        """Return error output from executing script."""
        try:
            with open(self.dir / "err.out", "r") as fd:
                return fd.read()
        except FileNotFoundError:
            return None

    def get_stdout(self) -> str:
        """Return log output from executing script."""
        with open(self.dir / "log.out", "r") as fd:
            return fd.read()