import json

def load_config(config_dir):
    with open(config_dir, "r") as f:
        config = json.loads(f.read())
    return config

def load_configs(config_dir_list):
    total_conf = {}
    for conf_dir in config_dir_list:
        conf = load_config(conf_dir)
        total_conf.update(conf)
    return total_conf


global_conf = load_config("configs/global.json")