config = {
    "data_root": {
        "train": "data/CC-CCII/train",
        "test": "data/CC-CCII/test"
    },
    "preprocess": {
        "image_resized_shape": (224, 224)
    },
    "active_learn": {
        "initial_labeled_nums": 100
    },
    "deep_learn": {
        "batch_size": 12
    }
}