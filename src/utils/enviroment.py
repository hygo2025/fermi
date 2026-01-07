import yaml
from pathlib import Path
from typing import Any

_config_cache = None

def _load_config() -> dict:
    """Load project configuration from YAML file (cached)."""
    global _config_cache
    if _config_cache is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "project_config.yaml"
        with open(config_path, 'r') as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache

def get_config(key_path: str = None) -> Any:
    """
    Get configuration value by dot-separated path.
    Example: get_config('raw_data.events_path')
    """
    config = _load_config()
    if key_path is None:
        return config
    
    keys = key_path.split('.')
    value = config
    for key in keys:
        value = value[key]
    return value
