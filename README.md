# GEO-Bench: Toward Foundation Models for Earth Monitoring

GEO-Bench is a [ServiceNow Research](https://www.servicenow.com/research) project. 

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.9%2B-green?logo=python&logoColor=green)](https://www.python.org)

GEO-Bench is a General Earth Observation benchmark for evaluating the performances of large pre-trained models on geospatial data. Read the [full paper](https://arxiv.org/abs/2306.03831) for usage details and evaluation of existing pre-trained vision models. This repo contains code to exectute the experiments reported in the paper.

# Installation


```console
$ https://github.com/ServiceNow/geo-bench-experiments.git
$ cd geo-bench-experiments
$ pip install .
```

# Create and run Experiments

Running experiments is a two-step process.

1. In the first step, we create directories that hold the necessary files that are needed to actually run an experiment. This can be achieved with the script command

```console
$ geobench_exp-gen_exp --task_config_path geobench_exp/configs/segmentation_task.yaml --model_config_path geobench_exp/configs/model_configs/segmentation/unet_resnet18.yaml
```

where the `task_config_path` flag should either point to `segmentation_task.yaml` or `classification_task.yaml`. The `model_config_path` points to the configuration of the model you want to run, for example a Unet with a ResNet18 backbone. This will create experiment directories under the specified `generate_experiment_dir` directory specified in the task config. Among other files, those subdirectory will hold a bash script called `run.sh` with a command to execute the run for that particular experiment.

The `geobench_exp-gen_exp` command is a shortcut for the `geobench_exp/generate_experiment.py` script which is solely controlled by the task config you write. The task config is annotated with comments to give an idea what controls what.

2. To execute an experiment, run the command contained in one of the `run.sh` scripts of the experiment you are interested in. For example,

```console
$ geobench_exp-run_exp --job_dir experiments/0.05x_train_segmentation_v1.0_01-18-2024_13:38:27resnet18_Unet/m-chesapeake/seed_0
```

The `geobench_exp-run_exp` command is a shortcut for the `geobench_exp/run_experiment.py` script which executes the training.


