from torch.utils.data import Dataset
from PIL import Image


# 图像数据集，根据文件路径和标签构造数据集
class ImageDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    @staticmethod
    def image_loader(path):
        image = Image.open(path)
        image = image.convert("RGB")
        return image

    def __getitem__(self, index):
        path = self.filepaths[index]
        label = self.labels[index]
        image = self.image_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.filepaths)

    
