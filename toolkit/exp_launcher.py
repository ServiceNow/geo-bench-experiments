"""Experiment Launcher with toolkit."""
# Convenient code for bypassing commandlines
# This will create experiments and lanch on toolkit
# This file serves as a template and will be remove from the repo once we're done with experiments.
# Please don't commit the changes related to your personnal experiments

import os
import sys

# from ..geobench_exp.experiment.experiment_generator import experiment_generator
from pathlib import Path

import dispatch_toolkit

from geobench_exp.experiment.experiment_generator import experiment_generator

sys.path.append("/mnt/home/geo-bench-experiments/geobench_exp")

os.environ["GEO_BENCH_DIR"] = "/mnt/data/ccb/"

experiment_dir = experiment_generator(
    # config_filepath=str(Path(__file__).parent.parent / "geobench/configs/classification_config.yaml"),
    config_filepath="/mnt/home/geo-bench-experiments/geobench_exp/configs/seed_segmentation_config.yaml"
)

dispatch_toolkit.push_code(Path(__file__).parent.parent)

# you may want to change to your WANDB_API_KEY."
os.environ["WANDB_API_KEY"] = "def8d0fad10d1479d79ab4c7e68530d59be04cf5"
dispatch_toolkit.toolkit_dispatcher(
    experiment_dir,
    env_vars=(f"WANDB_API_KEY={os.environ['WANDB_API_KEY']}", f"GEO_BENCH_DIR={os.environ['GEO_BENCH_DIR']}"),
)
