from bbo import yaml


def load(yml_path, opts):
    data = yaml.load(yml_path)

    return data, opts