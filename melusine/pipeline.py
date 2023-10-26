"""
This module contains classes for the MelusinePipeline object.

Implemented classes: [PipelineStepConf, PipelineConf, MelusinePipeline, PipelineHandler]
"""
from __future__ import annotations

import copy
import importlib
from typing import Dict, Iterable, List, Optional, Set, Tuple, TypeVar, cast

from pydantic import BaseModel, root_validator, validator
from sklearn.pipeline import Pipeline

from melusine.backend import backend
from melusine.backend.base_backend import Any
from melusine.base import MelusineTransformer
from melusine.conf import config
from melusine.io import IoMixin

T = TypeVar("T")


class PipelineStepConf(BaseModel):
    """
    Pydantic model for pipeline steps config.
    """

    class_name: str
    module: str
    name: Optional[str] = None
    config_key: Optional[str] = None
    parameters: Optional[Dict] = None

    @root_validator(pre=True)
    @classmethod
    def check_config_key_or_args(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make sure that the step conf contains either:
        - a config_key field
        - a name field and a parameter field

        Args:
        values: Values of the step configuration.

        Returns:
        _: Values of the step configuration.
        """
        if not values.get("config_key"):
            if not (values.get("name") and values.get("parameters")):
                raise ValueError("abc")

        return values

    @validator("name", always=True)
    @classmethod
    def validate_name(cls, name: str, values: Dict[str, Any]) -> str:
        """
        If a config_key is specified, use it as name.
        Otherwise, use the name field.

        Parameters
        ----------
        name: Optional[str]
            Step parameters
        values: Dict[Any]
            Values of the step configuration.

        Returns
        -------
        _: str
            The founded key.
        """
        config_key = values.get("config_key")

        # Get parameters from config
        if config_key:
            return config_key

        return name

    @validator("parameters", always=True)
    @classmethod
    def validate_parameters(cls, parameters: str, values: Dict[str, Any]) -> str:
        """
        If a config_key is specified, get the step parameters from the Melusine config.
        Otherwise, use the parameters field.

        Parameters
        ----------
        parameters: Optional[Dict[str, Any]]
            Step parameters
        values: Dict[Any]
            Values of the step configuration.

        Returns
        -------
        _: str
            The founded key.
        """
        config_key = values.get("config_key")

        # Get parameters from config
        if config_key:
            return config[config_key]

        return parameters


class PipelineConf(BaseModel):
    """
    Pydantic model for pipelines config.
    """

    steps: List[PipelineStepConf]


class MelusinePipeline(Pipeline):
    """
    This class defines and executes data transformation.

    The MelusinePipeline is built on top of sklearn Pipelines.
    """

    OBJ_NAME: str = "name"
    OBJ_KEY: str = "config_key"
    OBJ_PARAMS: str = "parameters"
    STEPS_KEY: str = "steps"
    OBJ_CLASS: str = "class_name"
    OBJ_MODULE: str = "module"

    def __init__(
        self,
        steps: List[Tuple[str, MelusineTransformer]],
        memory: Optional[bool] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize attributes.

        Parameters
        ----------
        steps: List[Tuple[str, MelusineTransformer]]
            List of the pipeline steps.
        memory: bool
            If True, cache invariant transformers when running grid searches.
        verbose: bool
            Verbose mode.
        """
        Pipeline.__init__(self, steps=steps, memory=memory, verbose=verbose)

        self.memory = memory
        self.verbose = verbose

    @property
    def input_columns(self) -> List[str]:
        """
        Input fields of the Pipeline.

        Returns
        -------
        _: List[str]
            List of input fields.
        """
        column_set: Set[str] = set()
        for _, step in self.steps:
            # UNION between sets
            column_set |= set(step.input_columns)

        return list(column_set)

    @property
    def output_columns(self) -> List[str]:
        """
        Output fields of the Pipeline.

        Returns
        -------
        _: List[str]
            List of output fields.
        """
        column_set: Set[str] = set()
        for _, step in self.steps:
            column_set |= set(step.output_columns)

        return list(column_set)

    @classmethod
    def get_obj_class(cls, obj_params: Dict[str, Any]) -> Any:
        """
        Get the class object of an instance.

        Parameters
        ----------
        obj_params: Dict[str, Any].

        Returns
        -------
        _: Any
            Class object.
        """
        obj_class_name = obj_params.pop(cls.OBJ_CLASS)
        obj_module = obj_params.pop(cls.OBJ_MODULE)

        obj_class = MelusinePipeline.import_class(obj_class_name, obj_module)

        return obj_class

    @staticmethod
    def import_class(obj_class_name: str, obj_module: str) -> Any:
        """
        Method to import a class dynamically.

        Parameters
        ----------
        obj_class_name: str
            Name of the object to be imported.
        obj_module: str
            Name of the module containing the object to be imported.

        Returns
        -------
        _: Any
            Class object.
        """
        # Import object class from name and module
        module = importlib.import_module(obj_module)
        if not hasattr(module, obj_class_name):
            raise AttributeError(f"Object `{obj_class_name}` cannot be loaded from module `{module}`.")
        obj_class = getattr(module, obj_class_name)
        return obj_class

    @classmethod
    def flatten_pipeline_config(cls, conf: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested Melusine Pipelines.

        This makes it easier for the rest of the processing.

        Parameters
        ----------
        conf: Dict[str, Any]
            Base pipeline conf possibly containing nested pipelines.

        Returns
        -------
        _: Dict[str, Any]
            Flattened conf.
        """
        new_conf: List[Any] = list()
        for step in conf[cls.STEPS_KEY]:
            if step.get(cls.OBJ_CLASS, "") == cls.__name__:
                subpipeline_conf = cls.flatten_pipeline_config(step["parameters"])
                new_conf.extend(subpipeline_conf[cls.STEPS_KEY])
            else:
                new_conf.append(step)
        conf[cls.STEPS_KEY] = new_conf

        return conf

    @classmethod
    def from_config(
        cls, config_key: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> MelusinePipeline:
        """
        Instantiate a MelusinePipeline from a config key.

        Parameters
        ----------
        config_key: str
            Key of the pipeline configuration.
        config_dict: dict
            Dict containing the pipeline configuration.

        Returns
        -------
        _: MelusinePipeline
            Pipeline instance.
        """
        init_params = dict()

        # Get config dict
        if config_key and not config_dict:
            raw_config_dict = config[config_key]
            config_dict = cls.parse_pipeline_config(raw_config_dict)

        elif config_dict and not config_key:
            config_dict = cls.parse_pipeline_config(config_dict)
        else:
            raise ValueError("You should specify one and only one of 'config_key' and 'config_dict'")

        # Prepare step list
        steps = list()

        # Load steps meta data
        steps_meta = config_dict.pop(cls.STEPS_KEY)

        # Instantiate transformers
        for obj_meta in steps_meta:
            # Step name
            step_name: str = obj_meta.pop(cls.OBJ_NAME, None)

            # Step class
            obj_class = cls.get_obj_class(obj_meta)

            # Step arguments
            obj_params = obj_meta[cls.OBJ_PARAMS]

            if issubclass(obj_class, IoMixin):
                obj = obj_class.from_config(config_dict=obj_params)
            else:
                raise TypeError(f"Object {obj_class} does not inherit from the SaverMixin class")  # pragma: no cover

            # Add step to pipeline
            steps.append((step_name, obj))

        # Init params
        init_params.update(config_dict)
        init_params.update(kwargs)

        # Instantiate MelusinePipeline object
        return cls(steps=steps, **init_params)

    @classmethod
    def parse_pipeline_config(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse config dict to replace config key by the associated configurations.

        Parameters
        ----------
        config_dict: Dict[str, Any]
            Initial config.

        Returns
        -------
        _: Dict[str, Any]
            Parsed config.
        """
        config_dict = copy.deepcopy(config_dict)

        # Validate raw pipeline conf
        config_dict = PipelineConf(**config_dict).dict(exclude_unset=True)
        config_dict = cast(Dict[str, Any], config_dict)

        steps = []
        for step in config_dict[cls.STEPS_KEY]:
            # Step defined from the config
            config_key = step.get(cls.OBJ_KEY)
            if config_key:
                # Use config key as step name
                step[cls.OBJ_NAME] = config_key
                _ = step.pop(cls.OBJ_KEY)
                # Update step parameters
                step[cls.OBJ_PARAMS] = config[config_key]

            # Nested pipeline
            if step[cls.OBJ_CLASS] == cls.__name__:
                raw_nested_pipeline_config = step[cls.OBJ_PARAMS]
                step[cls.OBJ_PARAMS] = cls.parse_pipeline_config(raw_nested_pipeline_config)

            # Add parsed step config to step list
            steps.append(step)

        config_dict[cls.STEPS_KEY] = steps

        return MelusinePipeline.flatten_pipeline_config(config_dict)

    @classmethod
    def get_config_from_key(cls, config_key: str) -> Dict[str, Any]:
        """
        Parse config dict to replace config key by the associated configurations.

        Parameters
        ----------
        config_key: Pipeline configuration key

        Returns
        -------
        _: Dict[str, Any]
            Parsed config.
        """
        return cls.parse_pipeline_config(config_dict=config[config_key])

    def validate_input_fields(self, data: Any) -> None:
        """
        Validate input fields prior of executing the pipeline.
        Use the input_columns and output_columns attributes of each step.

        Parameters
        ----------
        data: Any
            Input data.
        """
        active_fields: Set[str] = set(backend.get_fields(data))

        for step_name, step in self.steps:
            difference = set(step.input_columns).difference(active_fields)
            if difference:
                raise ValueError(
                    f"Error at step '{step_name}'.\n"
                    f"Fields {difference} should be either:\n"
                    "- Present in input fields or\n"
                    "- Created by a previous pipeline step"
                )

            active_fields |= set(step.output_columns)

    def transform(self, X: Iterable[Any]) -> Iterable[Any]:  # NOSONAR
        """
        Transform input dataset.

        Parameters
        ----------
        X: Dataset
            Input Dataset.

        Returns
        -------
        _: Dataset
            Output Dataset.
        """
        self.validate_input_fields(X)
        return super().transform(X)
