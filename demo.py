# encoding=utf-8
from tasks.random_samping import random_sampling
from utils.config import load_config
# import torch


if __name__ == "__main__":
    random_sampling(load_config("configs/demo_cfg.json"))
    loss_prediction(load_config("configs/loss_pred.json"))
    random_sampling(load_config("configs/demo_cfg.json"))
    random_sampling(load_config("configs/demo_cfg.json"))
