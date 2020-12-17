from torch.utils.data import Dataset
from PIL import Image

from utils.data import image_resize, process_masks


class DatasetUtils:
    @staticmethod
    def cla(x_info, y_info):
        return ImageDataset(x_info, y_info, image_resize)

    @staticmethod
    def seg(x_info, y_info):
        return ImageSegDataset(x_info, y_info, image_resize, process_masks)


# 图像数据集，根据文件路径和标签构造数据集
class ImageDataset(Dataset):
    def __init__(self, data_paths, labels, transform=None, target_transform=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform

    @staticmethod
    def image_loader(path):
        image = Image.open(path)
        image = image.convert("RGB")
        return image

    def __getitem__(self, index):
        path = self.data_paths[index]
        label = self.labels[index]
        image = self.image_loader(path)
        if self.transform:
            image = self.transform(image)
        # 额外返回index，便于主动学习采样
        return image, label, index

    def __len__(self):
        return len(self.data_paths)

    
class ImageSegDataset(Dataset):
    def __init__(self, data_paths, mask_paths, transform=None, target_transform=None):
        self.data_paths = data_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def image_loader(path):
        image = Image.open(path)
        image = image.convert("RGB")
        return image

    @staticmethod
    def mask_loader(path):
        mask = Image.open(path)
        return mask

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        mask_path = self.mask_paths[index]
        image = self.image_loader(data_path)
        mask = self.mask_loader(mask_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        # 额外返回index，便于主动学习采样
        return image, mask, index

    def __len__(self):
        return len(self.data_paths)


# TODO: VolumeDataset

# TODO: TwoPhaseDataset

datasetUtils = DatasetUtils()

    
