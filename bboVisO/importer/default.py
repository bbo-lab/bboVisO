from bbo import yaml
from pathlib import Path


def load(yml_path, opts):
    data = yaml.load(yml_path)
    data["videos"] = [Path(v) for v in data["videos"]]
    return data, opts
