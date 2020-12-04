# encoding=utf-8
import tasks.random_samping as random_samping 
import tasks.deep_seg as deep_seg
from utils.config import load_config
# import torch


if __name__ == "__main__":
    # random_sampling(load_config("configs/demo_cfg.json"))
    # loss_prediction(load_config("configs/loss_pred.json"))
    # random_sampling(load_config("configs/demo_cfg.json"))
    # random_sampling(load_config("configs/demo_cfg.json"))
    deep_seg.pipeline(load_config("configs/deep_seg.json"))
