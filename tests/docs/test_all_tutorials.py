import importlib
from types import ModuleType


def test_all_tutorials(melusine_tutorial: str) -> None:
    """Test tutorial code by importing it"""
    # Import parent module just to have the __init__ in the coverage
    parent_module_path, _, _ = melusine_tutorial.rpartition(".")
    parent_module = importlib.import_module(melusine_tutorial)
    assert isinstance(parent_module, ModuleType)

    mod = importlib.import_module(melusine_tutorial)
    assert isinstance(mod, ModuleType)
