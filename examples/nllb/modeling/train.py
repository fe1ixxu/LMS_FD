# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import asyncio
import glob
import logging
import os
import time

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from examples.nllb.modeling.utils import execute_in_shell
from fb_sweep.sweep.slurm import copy_all_python_files

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training pipeline")


def generate_data_prepare_cmd(data_cfg_full_path, snapshot_dir, output_dir):
    logger.info("generate_data_prepare_cmd")
    data_config_path, data_config_name = os.path.split(data_cfg_full_path)

    cmd = [
        "python",
        f"{snapshot_dir}/examples/nllb/modeling/prepare_data.py",
        "--config-path",
        f"{data_config_path}",
        "--config-name",
        f"{data_config_name}",
        "--output-dir",
        f"{output_dir}",
    ]
    logger.info(f"generated data preparation command:\n{cmd}\n")
    return cmd


def execute_data_prepare(data_cfg_full_path, snapshot_dir, output_dir, dry_run=False):
    logger.info("execute_data_prepare")

    prepare_out_dir = f"{output_dir}/data_preparation"
    prepare_cmd = generate_data_prepare_cmd(
        data_cfg_full_path, snapshot_dir, prepare_out_dir
    )
    if dry_run:
        os.makedirs(f"{prepare_out_dir}/data_bin/shard000", exist_ok=True)
    execute_in_shell(prepare_cmd, shell=False, dry_run=dry_run, quiet=False)

    data_bins = glob.glob(f"{prepare_out_dir}/data_bin/shard*/")
    return data_bins


def traverse_values_from_cfg(prefix, cfg):
    """
    Recursively traverse all config values
        connect hierarchical values with dots, in "xxx.xxx.xxx" format
    """
    kv_list = []
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == "_name" and not isinstance(v, dict):
                kv_list.extend([f"{prefix}={v}"])
            else:
                kv_list.extend(
                    traverse_values_from_cfg(f"{prefix}.{k}" if prefix else f"{k}", v)
                )
    else:
        kv_list = [f"{prefix}={cfg}"]
    return kv_list


def generate_train_cmd(data_bins, train_cfg_full_path, snapshot_dir, output_dir):
    """
    Generate commands to call hydra-train with multirun
        read hp to be swept & overridden from yaml file
        generate hydra multirun commands
    """
    logger.info("generate_train_cmd")
    train_config_path, train_config_name = os.path.split(train_cfg_full_path)

    with initialize(config_path=train_config_path):
        train_cfg = compose(config_name=train_config_name)
        train_cfg = OmegaConf.to_container(train_cfg, resolve=True, enum_to_str=True)
        logger.info(f"\ntraining config\n{OmegaConf.to_yaml(train_cfg)}\n")

    cmd = [
        "python",
        f"{snapshot_dir}/fairseq_cli/hydra_train.py",
        "--multirun",
        f"hydra.sweep.dir={output_dir}",
    ]
    cmd.extend(traverse_values_from_cfg("", train_cfg))
    if data_bins:
        cmd.append(f"task.data={','.join(data_bins)}")
        assert train_cfg["task"].get("data", "") == "", "redundant data path"
    else:
        assert train_cfg["task"].get("data", "") != "", "missing data path"
        assert os.path.isdir(train_cfg["task"]["data"]), "invalid data path"

    logger.info(f"generated hydra_train command:\n{cmd}\n")
    return cmd


def main(
    data_cfg_path, train_cfg_path, output_dir, skip_preparation=False, dry_run=False
):
    """
    The main function controlling the whole process
        get prepared data
        read training config and train
    """
    logger.info("start training pipeline")
    logger.info(f"data config full path: {data_cfg_path}")
    logger.info(f"training config full path: {train_cfg_path}")
    logger.info(f"output_dir: {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    snapshot_dir = copy_all_python_files(
        source=".",
        snapshot_main_dir=output_dir,
        code_snapshot_hash="snapshot_training_pipeline",
        recurse_dirs="fairseq,fairseq_cli,examples/nllb",
    )
    os.environ["PYTHONPATH"] = snapshot_dir + ":" + os.environ.get("PYTHONPATH", "")

    if skip_preparation:
        logger.info(
            "skip data preparation and use the binarized specified in train_config.task.data"
        )
        data_bins = []
    else:
        data_bins = execute_data_prepare(
            data_cfg_full_path=data_cfg_path,
            snapshot_dir=snapshot_dir,
            output_dir=f"{output_dir}/data_preparation",
            dry_run=dry_run,
        )

    train_cmd = generate_train_cmd(
        data_bins=data_bins,
        train_cfg_full_path=train_cfg_path,
        snapshot_dir=snapshot_dir,
        output_dir=f"{output_dir}/training_output",
    )
    logger.info("start training")
    execute_in_shell(train_cmd, shell=False, dry_run=dry_run, quiet=False)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-config",
        default="baselines_conf/data_configs/miniset.yaml",
    )
    data_group.add_argument(
        "--skip-prepare-data",
        action="store_true",
        help="skip data preparation step and use the binarized specified in train_config.task.data",
    )
    parser.add_argument(
        "--train-config", default="baselines_conf/training_configs/train_inherited.yaml"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory under default hydra.run.dir",
    )
    parser.add_argument("--log-file", default="training_pipeline.log")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="see what commands will be executed (in simulation mode)",
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else f"/checkpoint/{os.getenv('USER')}/outputs/{os.path.basename(args.train_config).rsplit('.', 1)[0]}_{int(time.time())}"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logger.info(f"output dir: {output_dir}")

    fh = logging.FileHandler(filename=os.path.join(output_dir, args.log_file))
    logger.addHandler(fh)

    main(
        data_cfg_path=args.data_config,
        train_cfg_path=args.train_config,
        output_dir=output_dir,
        skip_preparation=args.skip_prepare_data,
        dry_run=args.dry_run,
    )
