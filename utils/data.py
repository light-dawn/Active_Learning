# encoding=utf-8
import sys
sys.path.append("./")
import os
from config import config
from pathlib import Path
import torchvision.transforms as transforms

# 获取当前目录的绝对路径
cur_path = os.path.realpath(os.curdir)


# 加载分类任务数据，支持加载的文件结构: root -> categories -> data 
def load_labeled_data_paths(data_root):
    paths, labels = [], []
    for index, category in enumerate(sorted(os.listdir(data_root))):
        category_path = os.path.join(data_root, category)
        labels.extend([index for _ in range(len(os.listdir(category_path)))])
        for file_path in sorted(os.listdir(category_path)):
            paths.append(os.path.join(category_path, file_path))
    return paths, labels


def load_seg_data_paths(root, image_suffix=".jpg", mask_suffix=".png"):
    """
    mask_suffix: 掩膜文件的后缀名，用来拼接路径
    """
    data_paths, mask_paths = [], []
    # print(f"cur_path: {cur_path}, root: {root}")
    data_root, mask_root = os.path.join(cur_path, root, "image"), os.path.join(cur_path, root, "mask")
    # print(f"数据根目录: {data_root}, 掩模根目录: {mask_root}")
    # print(os.listdir(data_root))
    all_mask_paths = [str(item) for item in Path(mask_root).rglob("*" + mask_suffix)]
    for item in Path(data_root).rglob("*" + image_suffix):
        image_path = "/".join(str(item).split("/")[-2:])
        target_mask_path = os.path.join(mask_root, image_path.split(".")[0] + mask_suffix)
        if target_mask_path in all_mask_paths:
            data_paths.append(str(item))
            mask_paths.append(target_mask_path)
    return data_paths, mask_paths

    
# 读取yaml配置文件
def read_yaml_config(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as file:
        file_data = file.read()
    return yaml.load(file_data, Loader=yaml.SafeLoader)


def image_resize(image):
    t = transforms.Compose(
        [
            transforms.Resize(config["preprocess"]["image_resized_shape"]),
            transforms.ToTensor()
        ]
    )
    image_tensor = t(image)
    return image_tensor


def process_masks(mask):
    t = transforms.Compose(
        [
            transforms.Resize(config["preprocess"]["image_resized_shape"]),
            transforms.ToTensor(),
        ]
    )
    mask_tensor = t(mask)
    mask_tensor = mask_tensor.squeeze(0)
    return mask_tensor


if __name__ == "__main__":
    # print(path)
    # print(os.path.join(path, "raw_data/CC-CCII/ct_lesion_seg"))
    data_paths, mask_paths = load_seg_data_paths(os.path.join(path, "raw_data/CC-CCII/ct_lesion_seg"), mask_suffix=".png")
    print(f"加载到的数据量: {len(data_paths)}")
    print(f"加载到的掩膜量: {len(mask_paths)}")
    print(data_paths[0])
    print(mask_paths[0])