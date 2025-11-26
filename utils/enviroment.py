import os

from utils.paths import PathsEnum

def get_path(path_enum: PathsEnum, base_path: str = os.getenv("BASE_PATH")):
    return os.path.join(base_path, path_enum.value)