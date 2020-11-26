from sampler import SamplerUtils
from dataset import DatasetUtils
from data import DataUtils


class Utils:
    def __init__(self):
        self.samplers = SamplerUtils()
        self.datasets = DatasetUtils()
        self.dataloaders = DataUtils()
        self.optimizers = OptimizerUtils()
        self.losses = LossUtils()

