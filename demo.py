# encoding=utf-8
import sys
import tasks.random_samping as random_samping 
import tasks.deep_seg as deep_seg
import tasks.fed_seg as fed_seg
from utils.config import load_config
from utils.log import Logger
import datetime
import os
# import torch

log_name = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H-%M-%S") + ".log"
sys.stdout = Logger(filename=os.path.join("logs", log_name))


if __name__ == "__main__":
    # random_sampling(load_config("configs/demo_cfg.json"))
    # loss_prediction(load_config("configs/loss_pred.json"))
    # random_sampling(load_config("configs/demo_cfg.json"))
    # random_sampling(load_config("configs/demo_cfg.json"))
    # deep_seg.pipeline(load_config("configs/deep_seg.json"))
    fed_seg.pipeline(load_config("configs/fed_seg.json"))
