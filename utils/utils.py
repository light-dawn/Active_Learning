from torch import nn, optim
from models import unet, resnet
from utils.dataset import *


class ModelUtils:
    @staticmethod
    def unet(conf):
        return unet.UNet(conf["n_channels"], conf["n_classes"])

    @staticmethod
    def unet_feat(conf):
        return unet.UNet_FM(conf["n_channels"], conf["n_classes"])

    @staticmethod
    def resnet18(conf):
        return resnet.ResNet18(n_channels=conf["n_channels"], num_classes=conf["n_classes"])

    def resnet18_feat(conf):
        return resnet.ResNet18_FM(n_channels=conf["n_channels"], num_classes=conf["n_classes"])

    
class LossUtils:
    @staticmethod
    def cross_entropy():
        return nn.CrossEntropyLoss()

    @staticmethod
    def cross_entropy_reduction_none():
        return nn.CrossEntropyLoss(reduction="none")


class OptimUtils:
    @staticmethod
    def rms_prop(params, conf):
        return optim.RMSprop(params, lr=conf.get("lr", 1e-2), weight_decay=conf.get("weight_decay", 0), 
                             momentum=conf.get("momentum", 0))
    
    @staticmethod
    def adam(params, conf):
        return optim.Adam(params, lr=conf.get("lr", 1e-2), weight_decay=conf.get("weight_decay", 0), 
                             momentum=conf.get("momentum", 0))
    
    @staticmethod
    def sgd(params, conf):
        return optim.SGD(params, lr=conf.get("lr", 1e-2), weight_decay=conf.get("weight_decay", 0), 
                             momentum=conf.get("momentum", 0))


modelUtils = ModelUtils()
lossUtils = LossUtils()
optimUtils = OptimUtils()
datasetUtils = DatasetUtils()