# src/tools/__init__.py
from importlib import import_module

submodules = ["wikipedia", "arxiv"]

# Collect all tool functions in one list
tools = []

for mod_name in submodules:
    mod = import_module(f"src.tools.{mod_name}")
    for attr in getattr(mod, "__all__", []):
        obj = getattr(mod, attr)
        globals()[attr] = obj  # make it importable directly
        tools.append(obj)

__all__ = [name for name in globals() if not name.startswith("_")]
