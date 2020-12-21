import sys
import datetime
import os

from utils.config import load_config, load_configs, global_conf
from utils.log import Logger
from utils.args import parse_args
from utils.dataset import datasetUtils
from utils.task import taskUtils
from utils.data import load_seg_data_paths, load_cla_data_paths_and_labels_specific


args = parse_args()
log_name = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H-%M-%S") + ".log"
sys.stdout = Logger(filename=os.path.join("logs", log_name), write_log=global_conf["write_log"])


def run_task(task):
    task.start()
    task.run()
    task.end()


def main():
    # Check if the args are valid.
    assert args.task_type == "seg" or args.task_type == "cla", "Only support segmentation and classification currently."
    assert args.mode in ("deep", "active", "fed", "lefal", "tefal"), "Mode not supported."
    print("Trail Mode: ", args.mode)
    print("Task Type: ", args.task_type)
    print("Active Strategy: ", args.strategy)
    print("Data Root: ", global_conf["data_root"])
    conf_dir = ["configs/deep/seg.json"] if args.task_type == "seg" else ["configs/deep/cla.json"]
    data_load_func = load_seg_data_paths if args.task_type == "seg" else load_cla_data_paths_and_labels_specific
    # Load conf
    if args.mode == "active":
        conf_dir.append(os.path.join("configs/active", args.strategy + ".json"))
    elif args.mode == "fed":
        conf_dir.append("configs/fed/fed.json")
    elif args.mode == "lefal":
        conf_dir.extend([os.path.join("configs/active", args.strategy + ".json"), "configs/fed/fed.json"])
    else:
        conf_dir.append("configs/fed/tefal.json")
    conf = load_configs(conf_dir)
    # Create a dataset according to the task type.
    sep = "\\" if global_conf["os"] == "windows" else "/"
    x_info, y_info = data_load_func(global_conf["data_root"], sep=sep)
    dataset = getattr(datasetUtils, args.task_type)(x_info, y_info)
    # Create a task
    task = getattr(taskUtils, args.mode)(dataset, conf, args.strategy)
    run_task(task)


if __name__ == "__main__":
    main()
