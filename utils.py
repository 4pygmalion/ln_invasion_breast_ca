import yaml

def open_yaml(filepath:str) -> dict:
    with open(filepath, 'r') as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)