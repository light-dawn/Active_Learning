import json

def load_config(config_dir):
    with open(config_dir, "r") as f:
        config = json.loads(f.read())
    return config