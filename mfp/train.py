import os
import sys
from dm_control import suite

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["HYDRA_FULL_ERROR"]="1"
os.environ["LAZY_LEGACY_OP"] = "0"
import warnings

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["WANDB_MODE"] = "offline"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import hydra
from termcolor import colored

from mfp.common.parser import parse_cfg
from mfp.common.seed import set_seed
from mfp.common.buffer import Buffer
from mfp.envs import make_env
from mfp.mfp_alg import MFP
from mfp.trainer.online_trainer import OnlineTrainer
from mfp.common.logger import Logger

torch.backends.cudnn.benchmark = True
import os

@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
    """
    Script for training MFP agents.
    """
    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    trainer_cls = OnlineTrainer
    trainer = trainer_cls(
        cfg=cfg,
        env=make_env(cfg),
        agent=MFP(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
