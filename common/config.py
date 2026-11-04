"""Configuration and explicit extension loading; no deployment defaults."""
from importlib import import_module
from pathlib import Path


def load_config(path):
    import yaml

    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a mapping")
    return config


def load_extension(spec):
    """Load a user-supplied Python callable, never a network service."""
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("Supply an extension as python_module:callable")
    module, name = spec.split(":")
    value = getattr(import_module(module), name)
    if not callable(value):
        raise TypeError("The extension must be callable")
    return value


def required(config, name):
    value = config.get(name)
    if value is None or value == "":
        raise ValueError(f"Configure {name}; no private default is included")
    return value


def check_output_path(output, *inputs):
    """Do not put generated files inside a read-only input/model directory."""
    target = Path(output).expanduser().resolve()
    for item in inputs:
        if item is None:
            continue
        source = Path(item).expanduser().resolve()
        if target == source or source in target.parents:
            raise ValueError("Output must be separate from input/model paths")
    return target
