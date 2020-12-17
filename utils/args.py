import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="lefal", help="deep, active, fed, lefal, tefal")
    parser.add_argument('--strategy', type=str, default="no_active", help="active learning strategy")
    parser.add_argument('--task_type', type=str, default="seg", help="segmentation or classification")
    args = parser.parse_args()
    return args
