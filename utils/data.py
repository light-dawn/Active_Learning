# encoding=utf-8
import os
import yaml
from config import config

# 加载分类任务数据，支持加载的文件结构: root -> categories -> data 
def load_labeled_data_paths(data_root):
    paths, labels = [], []
    for index, category in enumerate(sorted(os.listdir(data_root))):
        category_path = os.path.join(data_root, category)
        labels.extend([index for _ in range(len(os.listdir(category_path)))])
        for file_path in sorted(os.listdir(category_path)):
            paths.append(os.path.join(category_path, file_path))
    return paths, labels

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


if __name__ == "__main__":
    yaml_data = read_yaml_config("config.yaml")
    print(type(yaml_data))
    print(yaml_data)
    

