# encoding=utf-8
import sys
sys.path.append("./")
import os
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm


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
    assert len(paths) == len(labels), "The number of data paths and labels not match."
    assert len(paths) > 0 and len(labels) > 0, "No paths and labels are loaded."
    return paths, labels


# sep is "/" for MacOS and "\\" for Windows
def load_seg_data_paths(root, image_suffix=".jpg", mask_suffix=".png", sep="/"):
    """
    mask_suffix: 掩膜文件的后缀名，用来拼接路径
    """
    data_paths, mask_paths = [], []
    # print(f"cur_path: {cur_path}, root: {root}")
    data_root, mask_root = os.path.join(cur_path, root, "image"), os.path.join(cur_path, root, "mask")
    # print(f"data dir: {data_root}, mask dir: {mask_root}")
    # print(os.listdir(data_root))
    # Handle the different seperation on different operation systems.
    all_mask_paths = [str(item).replace("\\", sep) for item in Path(mask_root).rglob("*" + mask_suffix)]
    all_mask_paths = [str(item).replace("/", sep) for item in all_mask_paths]
    # print("all_mask_paths: ", all_mask_paths[0])
    for item in Path(data_root).rglob("*" + image_suffix):
        image_path = sep.join(str(item).split(sep)[-2:])
        # print("image path: ", image_path)
        target_mask_path = os.path.join(mask_root, image_path.split(".")[0] + mask_suffix).replace("\\", sep)
        target_mask_path = target_mask_path.replace("/", sep)
        # print("target mask path: ", target_mask_path)
        if target_mask_path in all_mask_paths:
            data_paths.append(str(item))
            mask_paths.append(target_mask_path)
    assert len(data_paths) == len(mask_paths), "The number of data paths and mask paths not match."
    assert len(data_paths) > 0 and len(mask_paths) > 0, "No data paths are loaded."
    return data_paths, mask_paths


# Scan target data types recursively
def load_all_paths_recursive(data_root, data_types):
    result = []
    for data_type in data_types:
        print("Data type: ", data_type)
        result.extend([str(path).strip() for path in list(Path(data_root).rglob("*."+data_type))])
    return sorted(result)


def get_label_names(data_root):
    label_names = sorted(item.name for item in Path(data_root).glob("*/") if item.is_dir())
    return label_names


def read_csv(csv):
    with open(csv, "r") as f:
        csv_content = [line.strip() for line in f.readlines()]
    return csv_content


# This is a function specific to loading CC-CCII classification data
def load_cla_data_paths_and_labels_specific(data_root, nc_limit=300, sep="/", desc_csv="raw_data/ct_cla/lesions_slices.csv"):
    all_paths = load_all_paths_recursive(data_root, data_types=["jpg", "png"])
    label_names = get_label_names(data_root)
    print("Label names: ", label_names)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print("Label indices: ", label_to_index)

    # Control the volume of NC images
    nc_count = 0
    desc_csv = os.path.join(data_root, "lesions_slices.csv")
    paths_in_csv = read_csv(desc_csv)[1:]
    print("csv path example: ", paths_in_csv[0])
    data_paths, labels = [], []
    for path in tqdm(all_paths):
        # print("short_path: ", sep.join(path.split(sep)[-4:]))
        if "/".join(path.split(sep)[-4:]) in paths_in_csv:
            # print("path: ", path)
            data_paths.append(path)
            # print("label: ", path.split(sep)[-4])
            labels.append(path.split(sep)[-4])
        else:
            if nc_count >= nc_limit:
                continue
            # print("path: ", path)
            data_paths.append(path)
            nc_count += 1
            labels.append("NC")
    
    print("Labels: ", set(labels))
    return data_paths, [label_to_index.get(label) for label in labels]


def indices_train_test_split(indices, test_size=0.2, seed=2020):
    random.seed(seed)
    random.shuffle(indices)
    # print(indices)
    train_indices, test_indices = indices[:int((1-test_size)*len(indices))], indices[int((1-test_size)*len(indices)):]
    return train_indices, test_indices


def image_resize(image):
    t = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ]
    )
    image_tensor = t(image)
    return image_tensor


def process_masks(mask):
    t = transforms.Compose(
        [
            transforms.Resize([224, 224]),
        ]
    )
    mask = np.array(t(mask))
    mask_tensor = torch.tensor(mask, dtype=torch.uint8)
    mask_tensor = mask_tensor.squeeze(0)
    return mask_tensor


def seg_pred_to_mask(pred):
    pred = torch.argmax(pred, axis=0)
    return pred


def load_and_preprocess_single_image(image_dir):
    image = Image.Open(image_dir)
    return self.image_resize(image)


def create_visual_anno(anno):
    """"""
    assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 248, 220],  # cornsilk
        2: [100, 149, 237],  # cornflowerblue
        3: [102, 205, 170],  # mediumAquamarine
        4: [205, 133, 63],  # peru
        5: [160, 32, 240],  # purple
        6: [255, 64, 64],  # brown1
        7: [139, 69, 19],  # Chocolate4
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]
    anno_mask = Image.fromarray(visual_anno)
    return anno_mask
        

# 读取yaml配置文件
def read_yaml_config(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as file:
        file_data = file.read()
    return yaml.load(file_data, Loader=yaml.SafeLoader)





if __name__ == "__main__":
    # print(path)
    # print(os.path.join(path, "raw_data/CC-CCII/ct_lesion_seg"))
    # data_paths, mask_paths = load_seg_data_paths(os.path.join(path, "raw_data/ct_lesion_seg"), mask_suffix=".png")
    data_paths, labels = load_cla_data_paths_and_labels_specific("raw_data/ct_cla", desc_csv="raw_data/ct_cla/lesions_slices.csv", sep="\\")
    print(len(data_paths))
    print(len(labels))
    print("CP num: ", len(list(filter(lambda x: x == 0, labels))))
    print("NC num: ", len(list(filter(lambda x: x == 1, labels))))
    print("NCP num: ", len(list(filter(lambda x: x == 2, labels))))