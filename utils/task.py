from sampler import *
from tasks.deep import DeepPipeline
from tasks.fed import FedPipeline, LefalPipeline, TefalPipeline
from tasks.active import activeUtils
from utils.data import image_resize, process_masks


class TaskUtils:
    @staticmethod
    def deep(dataset, conf, strategy):
        assert strategy == "no_active", "The mode and the strategy are not matched."
        return DeepPipeline(dataset, conf)

    @staticmethod
    def active(dataset, conf, strategy, data_num=None):
        assert strategy != "no_active", "The mode and the strategy are not matched."
        return getattr(activeUtils, strategy)(dataset, conf, data_num)

    @staticmethod
    def fed(dataset, conf, strategy):
        assert strategy == "no_active", "The mode and the strategy are not matched."
        return FedPipeline(dataset, conf)

    @staticmethod
    def lefal(dataset, conf, strategy):
        assert strategy != "no_active", "The mode and the strategy are not matched."
        return LefalPipeline(dataset, conf, strategy)

    @staticmethod
    def tefal(dataset, conf, strategy):
        assert strategy == "no_active", "The mode and the strategy are not matched."
        return TefalPipeline(dataset, conf, strategy)


taskUtils = TaskUtils()


