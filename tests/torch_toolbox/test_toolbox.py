import os
import pickle
import tempfile

import pytest
from ruamel.yaml import YAML

from geobench_exp.experiment.experiment import Job
from geobench_exp.torch_toolbox import trainer


def train_job_on_task(config, task_specs, threshold, check_logs=True, metric_name="Accuracy", **kwargs):
    """Based on a job train model_generator on task.

    Args:
        model_generator: model_generator that has been instantiated and called with desired hparams
        config: config file
        task_specs: task specifications which to train model on
    """
    with tempfile.TemporaryDirectory(prefix="test") as job_dir:
        job = Job(job_dir)
        task_specs.save(job.dir)

        job.save_config(config)

        trainer.train(job_dir=job_dir)
        print(task_specs.benchmark_name)
        if check_logs:
            metrics = job.get_metrics()
            print(metrics)
            print(task_specs.benchmark_name)
            # assert (
            #     float(metrics[f"test_{metric_name}"]) > threshold
            # )  # has to be better than random after seeing 20 batches
            return metrics

        return None


def test_toolbox_segmentation():
    with open(
        os.path.join("tests", "data", "geobench-segmentation-test", "southAfricaCropType", "task_specs.pkl"), "rb"
    ) as fd:
        task_specs = pickle.load(fd)
        task_specs.benchmark_name = "testing"

    yaml = YAML()
    with open(os.path.join("tests", "configs", "base_segmentation.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile)

    train_job_on_task(config=config, task_specs=task_specs, threshold=0.04, metric_name="JaccardIndex")


@pytest.mark.parametrize(
    "backbone, model_generator_module_name",
    [
        ("resnet18", "geobench_exp.torch_toolbox.model_generators.timm_generator"),
    ],
)
def test_toolbox_classification(backbone, model_generator_module_name):
    with open(os.path.join("tests", "data", "geobench-classification-test", "eurosat", "task_specs.pkl"), "rb") as fd:
        task_specs = pickle.load(fd)
        task_specs.benchmark_name = "testing"

    yaml = YAML()
    with open(os.path.join("tests", "configs", "base_classification.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile)

    config["model"]["backbone"] = backbone
    config["model"]["model_generator_module_name"] = model_generator_module_name

    train_job_on_task(config=config, task_specs=task_specs, threshold=0.40)
