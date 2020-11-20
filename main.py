from dataset import ImageDataset
import utils.data as data
from config import config
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import resnet
import sampler


def deep_active_learn():
    # 获取文件和标签
    train_filepaths, train_labels = data.load_labeled_data_paths(config["data_root"]["train"])
    test_filepaths, test_labels = data.load_labeled_data_paths(config["data_root"]["test"])

    # 构建数据池，通过将labeled_indices传入sampler来模拟数据被oracle打标签
    data_pool = ImageDataset(train_filepaths, train_labels, data.image_resize)

    # 构造所有data的索引
    indices = list(range(len(data_pool)))
    random.shuffle(indices)

    # 初始化labeled数据
    k = config["active_learn"]["initial_labeled_nums"]
    labeled_indices = indices[:k]
    unlabeled_indices = indices[k:]

    # 通过labeled_indices构造Train DataLoader
    train_sampler = SubsetRandomSampler(labeled_indices)
    train_loader = DataLoader(data_pool, batch_size=config["deep_learn"]["batch_size"], sampler=train_sampler,
                              shuffle=True, pin_memory=False) # 设备性能好时设置为True，加快数据转到GPU的速度

    # 定义任务模型
    task_model = resnet.ResNet18(num_classes=3)

    active_sampler = sampler.HybridSampler

if __name__ == "__main__":
    deep_active_learn()