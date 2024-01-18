"""Experiment."""

import os
import pickle
import stat
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Union

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
        return OmegaConf.load(self.dir / "config.yaml")

    def save_config(self, config: Dict[str, Any], overwrite: bool = False) -> None:
        """Save config file in job directory.

        Args:
             config: config file for experiment logistics and training
             overwrite: whether to overwrite existing config file
        """
        config_path = self.dir / "config.yaml"
        if config_path.exists() and not overwrite:
            raise Exception("config alread exists and overwrite is set to False.")
        OmegaConf.save(config, self.dir / "config.yaml")

    def write_script(self, job_dir: str) -> None:
        """Write bash scrip that can be executed to run job.

        Args:
            job_dir: job directory from which to run job
        """
        script_path = self.dir / "run.sh"
        with open(script_path, "w") as fd:
            fd.write("#!/bin/bash\n")
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
        base_yaml = OmegaConf.load(base_sweep_config)

        base_yaml["command"] = [  # commands needed to run actual training script
            "${program}",
            "--job_dir",
            str(job_dir),
        ]
        config = Job(job_dir).config

        base_yaml["name"] = config["model"]["model_name"]

        save_path = os.path.join(job_dir, "sweep_config.yaml")
        OmegaConf.save(base_yaml, save_path)

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
