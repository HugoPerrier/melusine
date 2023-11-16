import importlib
import re
from types import ModuleType

# Tutorials not tested by this module
exception_list = [
    "BasicClassification",
    "Configurations",
]


def test_all_tutorials(add_docs_to_pythonpath, melusine_tutorial: str) -> None:
    """Test tutorial code by importing it"""
    for exception in exception_list:
        if re.search(exception, melusine_tutorial):
            return

    # Import parent module just to have the __init__ in the coverage
    parent_module_path, _, _ = melusine_tutorial.rpartition(".")
    parent_module = importlib.import_module(melusine_tutorial)
    assert isinstance(parent_module, ModuleType)

    mod = importlib.import_module(melusine_tutorial)
    assert isinstance(mod, ModuleType)
