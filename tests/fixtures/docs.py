import sys
from pathlib import Path
from typing import Any, Generator, List

import pytest

# Package source root
docs_folder = Path(__file__).parents[2] / "docs"
# package_source_folder = Path(__file__).parents[2] / "docs_src"
docs_source_folder = docs_folder / "docs_src"

# Modules excluded for the melusine_tutorial fixture
melusine_tutorial_exclude_list: List[str] = ["__init__"]


def get_tutorial_modules() -> List[str]:
    """
    Get all tutorial modules.

    Returns:
        _: List of modules found
    """
    # List modules in folder
    tutorial_list = docs_source_folder.rglob("**/*.py")

    # Convert file path (path/to/module.py) to module path (path.to.module)
    module_list = [
        str(tutorial_file.relative_to(docs_folder)).replace(".py", "").replace("/", ".")
        for tutorial_file in tutorial_list
        if "__init__" not in str(tutorial_file)
    ]

    return module_list


@pytest.fixture(
    params=get_tutorial_modules(),
)
def melusine_tutorial(request: Any) -> str:
    """Parametrized fixture for tutorials in the folder to test"""
    # Add docs to python path
    return request.param


@pytest.fixture
def add_docs_to_pythonpath() -> Generator[None, None, None]:
    """Parametrized fixture for tutorials in the folder to test"""
    # Add docs to python path
    sys.path.insert(0, str(docs_folder))
    yield None
