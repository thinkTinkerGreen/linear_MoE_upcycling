import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
