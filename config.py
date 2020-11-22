import sys

sys.path.append("./")

config = {
    "data_root": {
        "active_learn": {
            "train": "raw_data/CC-CCII/active/train",
            "test": "raw_data/CC-CCII/active/test"
        },
        "segmentation": "raw_data/CC-CCII/ct_lesion_seg"
    },
    "preprocess": {
        "image_resized_shape": (224, 224),
        "pretrained_model_path": "checkpoints/pretrained/resnet18.pth"
    },
    "active_learn": {
        "initial_budget": 100,
        "budget": 50,
        "cycles": 20
    },
    "deep_learn": {
        "batch_size": 12,
        "task_model": {
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 5e-4,
        },
        "loss_module": {
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 5e-4,
        }
    },
    "segmentation": {
        "batch_size": 12,
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 5e-4
    }
}