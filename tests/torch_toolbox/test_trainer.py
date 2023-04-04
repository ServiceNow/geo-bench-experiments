"""Test trainer.py"""

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

from ruamel.yaml import YAML

import geobench_exp
from geobench_exp.experiment.experiment import Job


def test_trainer_start():
    with open(
        os.path.join("tests", "data", "geobench-segmentation-test", "southAfricaCropType", "task_specs.pkl"), "rb"
    ) as fd:
        task_specs = pickle.load(fd)

    yaml = YAML()
    with open(os.path.join("tests", "configs", "base_segmentation.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile)

    toolbox_dir = Path(geobench_exp.torch_toolbox.__file__).absolute().parent

    with tempfile.TemporaryDirectory(prefix="test") as job_dir:
        # job_dir = f"{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
        # os.makedirs(job_dir, exist_ok=True)

        job = Job(job_dir)
        task_specs.save(job.dir)

        job.save_config(config)

        cmd = [sys.executable, str(toolbox_dir / "trainer.py"), "--job_dir", job_dir]
        subprocess.call(cmd)
