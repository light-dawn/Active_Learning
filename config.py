config = {
    "data_root": {
        "active_learn": {
            "train": "data/CC-CCII/active/train",
            "test": "data/CC-CCII/active/test"
        },
        "segmentation": {
            "train": "data/CC-CCII/seg/train",
            "test": "data/CC-CCII/seg/test",
        }
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
            "momentum": ,
            "weight_decay": ,
        },
        "loss_module": {
            "lr": 1e-3,
            "momentum": ,
            "weight_decay": ,
        }
    },
    "segmentation": {
        "batch_size": 12,
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 5e-4
    }
}