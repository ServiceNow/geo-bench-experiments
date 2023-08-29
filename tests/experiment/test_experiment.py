import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from ruamel.yaml import YAML

import geobench_exp
from geobench_exp.experiment.experiment import Job, get_model_generator
from geobench_exp.experiment.sequential_dispatcher import sequential_dispatcher


def test_unexisting_path():
    """
    Test trying to load from an unexisting module path.

    """
    try:
        get_model_generator("geobench_exp.torch_toolbox.model_generators.foobar")
    except Exception as e:
        assert isinstance(e, ModuleNotFoundError)


@pytest.mark.slow
@pytest.mark.parametrize(
    "config_filepath",
    [
        ("tests/configs/base_classification.yaml"),
        ("tests/configs/base_segmentation.yaml"),
    ],
)
def test_experiment_generator_on_benchmark(config_filepath: str, tmp_path: Path):
    experiment_generator_dir = Path(geobench_exp.experiment.__file__).absolute().parent

    # change experiment dir to tmp path
    with open(config_filepath, "r") as yamlfile:
        config = yaml.load(yamlfile, yaml.Loader)

    config["experiment"]["generate_experiment_dir"] = str(tmp_path)

    new_config_filepath = os.path.join(tmp_path, "config.yaml")
    with open(new_config_filepath, "w") as fd:
        yaml.dump(config, fd)

    print(f"Generating experiments in {tmp_path}.")
    cmd = [
        sys.executable,
        str(experiment_generator_dir / "experiment_generator.py"),
        "--config_filepath",
        new_config_filepath,
    ]

    subprocess.check_call(cmd)

    os.remove(new_config_filepath)
    exp_dir = os.path.join(tmp_path, os.listdir(tmp_path)[-1])
    sequential_dispatcher(exp_dir=exp_dir, prompt=False)
    for ds_dir in Path(exp_dir).iterdir():
        assert (ds_dir / "config.yaml").exists()


if __name__ == "__main__":
    test_experiment_generator_on_benchmark()
