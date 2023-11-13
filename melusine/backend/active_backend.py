"""
Melusine transformation can operate on different data structures such as dict or pandas.DataFrame.
Different transformation backends are used to process different data structures.
The ActiveBackend class stores an instance of the activated backend.

Implemented classes: [
    ActiveBackend,
]
"""

import logging
import os
from typing import Callable, List, Optional, Union

from melusine.backend.base_backend import Any, BaseTransformerBackend
from melusine.backend.dict_backend import DictBackend

logger = logging.getLogger(__name__)


class ActiveBackend(BaseTransformerBackend):
    """
    Class storing the active backend used by Melusine.
    """

    PANDAS_BACKEND: str = "pandas"
    DICT_BACKEND: str = "dict"

    def __init__(self, new_backend: Union[str, BaseTransformerBackend]):
        """

        Parameters
        ----------
        new_backend: Union[str, BaseTransformerBackend]
            Backend to be used
        """
        super().__init__()
        # self._backend: Union[BaseTransformerBackend, None] = None
        self._switch_backend(new_backend)

    def _switch_backend(self, new_backend: Union[BaseTransformerBackend, str]) -> None:
        """
        Method to switch from current backend to specified backend.

        Parameters
        ----------
        new_backend: Union[BaseTransformerBackend, str]
            New backend to be used
        """

        if isinstance(new_backend, BaseTransformerBackend):
            self._backend = new_backend

        elif new_backend == self.PANDAS_BACKEND:
            # Importing in local scope to prevent hard dependencies
            from melusine.backend.pandas_backend import PandasBackend

            self._backend = PandasBackend()

        elif new_backend == self.DICT_BACKEND:
            self._backend = DictBackend()

        else:
            raise ValueError(f"Backend {new_backend} is not supported")

        logger.info(f"Using backend '{new_backend}' for Data transformations")

    def apply_transform(
        self,
        data: Any,
        func: Callable,
        output_columns: Optional[List[str]] = None,
        input_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Method to apply a transform on a Dataset using current backend.

        Parameters
        ----------
        data: Dataset
            Data to be transformed
        func: Callable
            Transform function to apply to the input data
        output_columns: Sequence[str]
            List of output columns
        input_columns: Sequence[str]
            List of input columns
        kwargs

        Returns
        -------
        _: Dataset
            Transformed data
        """
        return self._backend.apply_transform(
            data=data,
            func=func,
            output_columns=output_columns,
            input_columns=input_columns,
            **kwargs,
        )

    def copy(self, data: Any, fields: List[str] = None) -> Any:
        """
        Method to make a copy of the input dataset.

        Parameters
        ----------
        data: Dataset
            MelusineDataset object
        fields: List[str]
            List of fields to include in the copy (by default copy all fields)

        Returns
        -------
        _: Dataset
            Copy of original object
        """
        return self._backend.copy(data, fields=fields)

    def get_fields(self, data: Any) -> List[str]:
        """
        Method to get the list of fields available in the input dataset.

        Parameters
        ----------
        data: Dataset
            MelusineDataset object

        Returns
        -------
        _: List[str]
            List of dataset fields
        """
        return self._backend.get_fields(data=data)

    def add_fields(self, left: Any, right: Any, fields: List[str] = None) -> Any:
        """
        Method to add fields from the right object to the left object

        Parameters
        ----------
        left: Dataset
            MelusineDataset object
        right: Dataset
            Melusine Dataset object
        fields: List[str]
            List of fields to be added

        Returns
        -------
        _: Dataset
            Left object with added fields
        """
        return self._backend.add_fields(left=left, right=right, fields=fields)

    def check_debug_flag(self, data: Any) -> bool:
        """
        Method to check if debug_mode is activated.

        Parameters
        ----------
        data: Dataset
            MelusineDataset object

        Returns
        -------
        _: bool
            True if debug mode is activated
        """
        return self._backend.check_debug_flag(data=data)

    def setup_debug_dict(self, data: Any, dict_name: str) -> Any:
        """
        Method to check if debug_mode is activated

        Parameters
        ----------
        data: Dataset
            MelusineDataset object
        dict_name: str
            Name of the debug dict field to be added

        Returns
        -------
        _: Dataset
            MelusineDataset object
        """
        return self._backend.setup_debug_dict(data=data, dict_name=dict_name)


def switch_backend(new_backend: Union[BaseTransformerBackend, str]) -> None:
    """
    Function to switch current backend to a new backend globally.

    Parameters
    ----------
    new_backend: Union[BaseTransformerBackend, str]
        New backend to be used
    """
    global backend

    backend._switch_backend(new_backend)


def reset_backend() -> None:
    """
    Function to reset the global backend to the default backend.
    """
    switch_backend(default_backend)


# Specify a default backend
default_backend = ActiveBackend.PANDAS_BACKEND

# Use the default backend unless a backend is specified as ENV variable
backend = ActiveBackend(new_backend=os.getenv("MELUSINE_BACKEND", default_backend))
