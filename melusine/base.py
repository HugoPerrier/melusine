"""
Base classes of the Melusine framework.

Implemented classes: [
    MelusineTransformer,
    MelusineDetector,
    MelusineModel,
    BaseLabelProcessor,
    MissingModelInputFieldError,
    MissingFieldError,
    MelusineFeatureEncoder
]
"""
from __future__ import annotations

import copy
import inspect
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from melusine.backend import backend
from melusine.io import IoMixin

logger = logging.getLogger(__name__)

# Dataset types supported by Melusine : pandas DataFrame and dicts
MelusineDataset = Union[Dict[str, Any], pd.DataFrame]

# Corresponding items are:
# - Dataset : Pandas DataFrame => Item : Pandas Series
# - Dataset Dict => Item Dict
MelusineItem = Union[Dict[str, Any], pd.Series]
Transformer = TypeVar("Transformer", bound="MelusineTransformer")


class TransformError(Exception):
    """
    Exception raised when an error occurs during the transform operation.
    """


class MelusineTransformer(BaseEstimator, TransformerMixin, IoMixin):
    """
    Define a MelusineTransformer object.

    Is an abstract class.

    It can be a Processor or a Detector.
    """

    def __init__(
        self,
        input_columns: Union[str, Iterable[str]],
        output_columns: Union[str, Iterable[str]],
        func: Optional[Callable] = None,
    ) -> None:
        """
        Attribute initialization.

        Parameters
        ----------
        input_columns: Union[str, Iterable[str]]
            List of input columns
        output_columns: Union[str, Iterable[str]]
            List of output columns
        func: Callable
            Transform function to be applied
        """
        IoMixin.__init__(self)

        self.input_columns: List[str] = self.parse_column_list(input_columns)
        self.output_columns: List[str] = self.parse_column_list(output_columns)
        self.func = func

    @staticmethod
    def parse_column_list(columns: Union[str, Iterable[str]]) -> List[str]:
        """
        Transform a string into a list with a single element.

        Parameters
        ----------
        columns: Union[str, Iterable[str]]
            String or list of strings with column name(s).

        Returns
        -------
        _: List[str]
            A list of column names.
        """
        # Change string into list of strings if necessary
        # "body" => ["body]
        if isinstance(columns, str):
            columns = [columns]
        return list(columns)

    def fit(self, df: MelusineDataset, y: Optional[pd.Series] = None) -> MelusineTransformer:
        """
        Fit a transformer.

        Parameters
        ----------
        df: MelusineDataset
            Input data.
        y: pd.Series
            Target data.

        Returns
        -------
        _: MelusineTransformer
            Fitted instance.
        """
        return self

    def transform(self, data: MelusineDataset) -> MelusineDataset:
        """
        Transform input data.

        Parameters
        ----------
        data: MelusineDataset
            Input data.

        Returns
        -------
        _: MelusineDataset
            Transformed data (output).
        """
        if self.func is None:
            raise AttributeError(f"Attribute func of MelusineTransformer {type(self).__name__} should not be None")
        try:
            return backend.apply_transform(
                data=data, input_columns=self.input_columns, output_columns=self.output_columns, func=self.func
            )

        except Exception as exception:
            func_name = self.func.__name__
            class_name = type(self).__name__
            input_columns = self.input_columns
            raise TransformError(
                f"Error in class: '{class_name}' "
                f"with method '{func_name}' "
                f"input_columns: {input_columns}\n"
                f"{str(exception)}"
            ).with_traceback(exception.__traceback__) from exception


class BaseMelusineDetector(MelusineTransformer, ABC):
    """
    Used to define detectors.

    Template Method str based on the MelusineTransformer class.
    """

    def __init__(
        self,
        name: str,
        input_columns: List[str],
        output_columns: List[str],
    ):
        """
        Attributes initialization.

        Parameters
        ----------
        name: str
            Name of the detector.
        input_columns:
            Detector input columns.
        output_columns:
            Detector output columns.
        """
        #  self.name needs to be set before the super class init
        #  Name is used to build the output_columns
        self.name = name

        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
        )

    @property
    def debug_dict_col(self) -> str:
        """
        Standard name for the column containing the debug info.

        Typically, a detector may return the following outputs:
        - output_result_col: bool
          > Ex: thanks_result: True
        - output_value_col: Any
          > Ex: thanks_output: "Remerciement plat"
        - output_score_col: float
          > Ex: thanks_score: 0.95
        - (debug) debug_dict_col: Dict[str, Any]
          > Ex: debug_thanks: {"thanks_text": "Merci"}
        """
        return f"debug_{self.name}"

    @property
    @abstractmethod
    def transform_methods(self) -> List[Callable]:
        """
        Specify the sequence of methods to be called by the transform method.

        Returns
        -------
        _: List[Callable]
            List of  methods to be called by the transform method.
        """

    def transform(self, df: MelusineDataset) -> MelusineDataset:
        """
        Re-definition of super().transform() => specific detector's implementation

        Transform input data.

        Parameters
        ----------
        df: MelusineDataset
            Input data.

        Returns
        -------
        _: MelusineDataset
            Transformed data (output).
        """
        # Debug mode ON?
        debug_mode: bool = backend.check_debug_flag(df)

        # Validate fields of the input data
        self.validate_input_fields(df)

        # Work on a copy of the DataFrame and limit fields to effective input columns
        # data_ = backend.copy(data, fields=self.input_columns)

        # Work on a copy of the DataFrame and keep all columns
        # (too complex to handle model input columns)
        data_ = backend.copy(df)

        # Get list of new columns created by the detector
        return_cols = copy.deepcopy(self.output_columns)

        # Create debug data dict
        if debug_mode:
            data_ = backend.setup_debug_dict(data_, dict_name=self.debug_dict_col)
            return_cols.append(self.debug_dict_col)

        for method in self.transform_methods:
            first_arg_name: str = list(inspect.signature(method).parameters)[0]

            if first_arg_name == "row":
                # Run row-wise method
                data_ = backend.apply_transform(
                    data=data_, input_columns=None, output_columns=None, func=method, debug_mode=debug_mode
                )
            else:
                data_ = method(data_, debug_mode=debug_mode)

        # Add new fields to the original MelusineDataset
        data = backend.add_fields(left=df, right=data_, fields=return_cols)

        return data

    def validate_input_fields(self, data: MelusineDataset) -> None:
        """
        Make sure that all the required input fields are present.

        Parameters
        ----------
        data: MelusineDataset
            Input data.
        """
        input_fields: List[str] = backend.get_fields(data)
        missing_fields: List[str] = [x for x in self.input_columns if x not in input_fields]
        if missing_fields:
            raise MissingFieldError(f"Fields {missing_fields} are missing from the input data")


class MelusineDetector(BaseMelusineDetector, ABC):
    """
    Defines an interface for detectors.
    All detectors used in a MelusinePipeline should inherit from the MelusineDetector class and
    implement the abstract methods.
    This ensures homogeneous coding style throughout the application.
    Alternatively, melusine user's can define their own Interface (inheriting from the BaseMelusineDetector)
    to suit their needs.
    """

    @property
    def transform_methods(self) -> List[Callable]:
        """
        Specify the sequence of methods to be called by the transform method.

        Returns
        -------
        _: List[Callable]
            List of  methods to be called by the transform method.
        """
        return [self.pre_detect, self.detect, self.post_detect]

    @abstractmethod
    def pre_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """What needs to be done before detection."""

    @abstractmethod
    def detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """Run detection."""

    @abstractmethod
    def post_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """What needs to be done after detection (e.g., mapping columns)."""


class MelusineModel(IoMixin, ABC):
    """
    Base class for all Melusine models.

    Is an abstract class.

    Implement generic methods such as:
    [predict, train, load, save, label encoding, etc.]
    """

    # Class constants
    INFERENCE_MODE_KEY: str = "_inference_mode"
    INFERENCE_MODE_LOCAL: str = "LOCAL"
    STANDARD_CONFIG_KEY: str = "melusine_model"

    def __init__(self, label_processor: BaseLabelProcessor) -> None:
        """
        Attributes initialization.

        Parameters
        ----------
        label_processor: BaseLabelProcessor
            Object to process supervised learning labels.
        """

        super().__init__()

        # Attributes not to be pickled
        self.pkl_exclude_list: List[str] = ["model"]
        self.json_exclude_list.append("model")

        self.label_processor = label_processor

        self.model: Any
        self.inference_mode: str

    def __getstate__(self) -> Dict[str, Any]:
        """
        Override __getstate__ to prevent pickling model objects.

        Model objects should be saved separately.

        Returns
        -------
        save_dict: Dict[str, Any]
            Filtered dict.
        """
        save_dict: Dict[str, Any] = self.__dict__.copy()
        return {k: v for k, v in save_dict.items() if k not in self.pkl_exclude_list}

    @property
    @abstractmethod
    def model_input_columns(self) -> List[str]:
        """
        List all columns required as input to use the model.

        Returns
        -------
        _: List[str]
            List of required input columns.
        """

    @abstractmethod
    def pre_train(self, x: Union[np.ndarray, pd.DataFrame], y: Optional[pd.Series] = None) -> None:
        """
        Method to run everything that needs to be run prior to the training and that requires access to the train data.

        Parameters
        ----------
        x: pd.DataFrame
            Train features.
        y: pd.Series
            Supervised learning labels.
        """

    @abstractmethod
    def prepare_input(self, x: pd.DataFrame) -> Union[List[np.ndarray], np.ndarray]:
        """
        Run transformation of the input data.
        Return data in a format ready to be fed to the ML/DL model.

        Parameters
        ----------
        x: pd.DataFrame
            Train features.

        Returns
        -------
        _: Union[List[np.ndarray], np.ndarray]
            Prepared train features.
        """

    @abstractmethod
    def build_network(self) -> None:
        """
        Assemble the deep learning network.
        """

    @abstractmethod
    def train_model(self, x: Union[List[np.ndarray], np.ndarray], y: Optional[np.ndarray] = None, **kwargs: Any) -> Any:
        """
        Method to launch the model training.

        Parameters
        ----------
        x: Union[List[np.ndarray], np.ndarray]
            Model inputs.
        y: np.ndarray
            Labels.
        kwargs: Any
            Keyword arguments.

        Returns
        -------
        _: Any
            Can be everything as keras 'history' for example.
        """

    def train(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        """
        Train model.

        Parameters
        ----------
        x: pandas.DataFrame
            DataFrame with raw features.
        y: pandas.Series
            Series with raw categories.
        kwargs: Any
            Keyword arguments.
        """

        # Run pre-training step
        self.pre_train(x, y)

        # Fit label encoder
        self.label_processor.fit(y)

        # Create neural network
        self.build_network()

        # Transform train features and labels
        x_input = self.prepare_input(x)
        y_cat = self.label_processor.transform(y)

        # Prepare validation data if there are some
        validation_data = kwargs.pop("validation_data", None)
        if validation_data:
            validation_data = self.prepare_validation_data(validation_data)
            kwargs["validation_data"] = validation_data

        # Train model
        self.train_model(x_input, y_cat, **kwargs)

    def prepare_validation_data(self, validation_data: Sequence) -> Sequence:
        """
        Prepare validation data for the model.

        Parameters
        ----------
        validation_data: Iterable
            Training data as 1st argument,
            Labels as second argument.

        Returns
        -------
        prepared_validation_data: Iterable
            Transformed training data as 1st argument,
            Transformed labels as second argument
        """
        if len(validation_data) < 2:
            raise ValueError(
                "'validation_data' should be an Iterable with:"
                "- training data as 1st argument"
                "- labels as second argument"
            )

        x_val_prepared = self.prepare_input(validation_data[0])
        y_val_cat = self.label_processor.transform(validation_data[1])
        prepared_validation_data = (x_val_prepared, y_val_cat, *validation_data[2:])

        return prepared_validation_data

    def validate_input_fields(self, x: pd.DataFrame) -> None:
        """
        Make sure that the required columns are present in the input DataFrame.

        Parameters
        ----------
        x: pd.DataFrame
            Input DataFrame.
        """

        missing_fields = set(self.model_input_columns).difference(x.columns)
        if missing_fields:
            raise MissingModelInputFieldError(f"Error for model '{type(self)}'.\nFields {missing_fields} are required.")

    def predict(self, x: pd.DataFrame, **kwargs: Any) -> Any:
        """
        Run model predict on input data.

        Parameters
        ----------
        x: pd.DataFrame
            DataFrame with raw features.
        kwargs: Any
            Keyword arguments.

        Returns
        -------
        _: Any
            Prediction results.
        """
        self.validate_input_fields(x)
        x_input = self.prepare_input(x)

        return self.model.predict(x_input, **kwargs)

    def predict_labels(self, x: pd.DataFrame, **kwargs: Any) -> Iterable[Tuple[str, float]]:
        """
        Run model predict on input data and compute the label with the maximum probability.
        Return a tuple (best_label, label_proba).

        Parameters
        ----------
        x: pd.DataFrame
            DataFrame with raw features.
        kwargs: Any
            Keyword arguments.

        Returns
        -------
        _: Iterable[Tuple[str, float]]
            Tuple with predicted label and associated probability.
        """
        results: Any = self.predict(x, **kwargs)
        labels, proba = self.label_processor.post_process(results)

        return list(zip(labels, proba))

    @staticmethod
    def get_versioned_model_path(path: str, version: Optional[str]) -> str:
        """
        Method to get the path to the desired version of the model.

        Parameters
        ----------
        path: str
            Base path to the model folder.
        version: Optional[str]
            Model version of interest.

        Returns
        -------
        _: str
            Path to the versioned model if applicable, base path to the model folder otherwise.
        """
        if version is not None:
            if not (Path(path) / version).exists():
                raise FileNotFoundError(f"Could not find version folder {version} at path {path}")
            path = str(Path(path) / version)

        return path

    @staticmethod
    def search_file(filename: str, path: str, filename_prefix: Optional[str] = None) -> str:
        """
        Look for a file at a given path.

        Parameters
        ----------
        filename: str
            Base file name.
        path: str
            Path to the file.
        filename_prefix: Optional[str]
            File name prefix.

        Returns
        -------
        full_path: str
            Full path to the file.
        """
        # Look for candidate files at given path
        if filename_prefix:
            filename = f"{filename_prefix}_{filename}"
        else:
            filename = f"{filename}"

        file_path = os.path.join(path, filename)

        # Nothing found at given path
        if (not os.path.isfile(file_path)) and (not os.path.isdir(file_path)):
            raise FileNotFoundError(f"Could not find files at path {file_path}")

        return file_path

    @classmethod
    def load_json(cls, path: str, filename_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Load data from a json file

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: Optional[str]
            File name prefix.

        Returns
        -------
        data: dict
            Data loaded from the json file
        """
        filename = cls.__name__ + cls.JSON_SUFFIX
        filepath = cls.search_file(filename, path, filename_prefix=filename_prefix)
        with open(filepath, encoding="utf-8") as file:
            data = json.load(file)

        return data

    def __repr__(self) -> str:
        """
        Class display method.

        Returns
        -------
        _: str
            String representation.
        """
        class_repr: str = ""
        for key, value in self.__dict__.items():
            if key in ["json_exclude_list", "pkl_exclude_list"]:
                continue
            class_repr += f"{key}: {value}\n"

        return class_repr


class BaseLabelProcessor(ABC):
    """
    Class to process supervised learning labels.
    """

    @property
    @abstractmethod
    def n_targets(self) -> int:
        """
        Number of class targets.

        Returns
        -------
        n_targets: int
            Number of targets (Ex: number of classes in a classification task).
        """

    @property
    @abstractmethod
    def classes_(self) -> Sequence[str]:
        """
        Get the classes of the fitted encoder

        Returns
        -------
        classes_: Sequence[str]
            All the labels known by the encoder.
        """

    @abstractmethod
    def fit(self, df: pd.series, y: Optional[Any] = None) -> None:
        """
        Fit the label processor.

        Parameters
        ----------
        df: pd.Series
            Input data.
        y: Any
            Supervised learning labels.
        """

    @abstractmethod
    def transform(self, df: pd.Series) -> np.ndarray:
        """
        Transform input data.

        Parameters
        ----------
        df: MelusineDataset
            Input data.

        Returns
        -------
        _: MelusineDataset
            Tranformed data (output).
        """

    @abstractmethod
    def inverse_transform(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Inverse transform labels.

        Parameters
        ----------
        y: pd.Series
            Encoded labels

        Returns
        -------
        _: numpy.ndarray
            Decoded labels
        """

    @abstractmethod
    def post_process(self, y_hat: np.ndarray) -> Tuple[Any, Any]:
        """
        Take as input the predictions of the model and return the labels with its
        associated score.

        Parameters
        ----------
        y_hat: np.ndarray
                The array of predictions given by the model

        Returns
        -------
        labels: Sequence[List[str]]
                List of predicted labels.
        scores: Sequence[List[float]]
                Scores or probabilities associated to the given label.
        """


class MissingModelInputFieldError(Exception):
    """
    Exception raised when a missing field is encountered by a MelusineModel
    """


class MissingFieldError(Exception):
    """
    Exception raised when a missing field is encountered by a MelusineTransformer
    """


class MelusineFeatureEncoder(ABC):
    """
    Base class for all Melusine meta-feature encoders:
    - Ordinal categorical features
    - Non-ordinal categorical features
    """

    ARBITRARY_VALUE: str = "__arbitrary__"

    def __init__(self) -> None:
        self._n_features: Optional[int] = None

    def preprocess_input(self, input_data: MelusineDataset, replace_na: bool = True) -> MelusineDataset:
        """
        Perform preprocessing of the input data before feeding it to the encoder.

        Parameters
        ----------
        input_data: pd.Series
            Series containing the input values to be encoded.
        replace_na:
            If True, invalid values are replaced by None.

        Returns
        -------
        input_data: MelusineDataset
            Value to be encoded.
        """
        return input_data

    @property
    def n_features(self) -> int:
        """
        Number of features created by the encoder.
        Watchout : Feature != Category
        For a variable with 2 categories (binary) there could be just one
        feature created.

        Returns
        -------
        _: int
            Number of features created by the encoder
        """
        if self._n_features is None:
            raise AttributeError("Attribute self._n_features is None. Encoder not fitted?")
        else:
            return self._n_features

    @abstractmethod
    def fit(self, data: pd.Series, y: Optional[pd.Series] = None) -> MelusineFeatureEncoder:
        """
        Fit the feature encoder.

        Parameters
        ----------
        data: pd.Series
            Input data.
        y: pd.Series
            Supervised learning labels.

        Returns
        -------
        _: MelusineFeatureEncoder
            Fitted instance.
        """

    @abstractmethod
    def transform(self, data: pd.Series) -> pd.Series:
        """
        Transform input data.

        Parameters
        ----------
        data: pd.Series
            Input data.

        Returns
        -------
        _: ps.Series
            Tranformed data (output).
        """


class MelusineRegex(ABC):
    """
    Class to standardise text pattern detection using regex.
    """

    REGEX_FLAGS: re.RegexFlag = re.IGNORECASE | re.MULTILINE
    DEFAULT_MATCH_GROUP: str = "DEFAULT"
    DEFAULT_SUBSTITUTION_PATTERN: str = " "

    # Match fields
    MATCH_RESULT: str = "match_result"
    NEUTRAL_MATCH_FIELD: str = "neutral_match_data"
    POSITIVE_MATCH_FIELD: str = "positive_match_data"
    NEGATIVE_MATCH_FIELD: str = "negative_match_data"

    # Match data
    MATCH_START: str = "start"
    MATCH_STOP: str = "stop"
    MATCH_TEXT: str = "match_text"

    @property
    def regex_name(self) -> str:
        """
        Name of the Melusine regex object.
        Defaults to the class name.
        """
        return getattr(self, "_regex_name", type(self).__name__)

    @property
    @abstractmethod
    def positive(self) -> Union[Dict[str, str], str]:
        """
        Define regex patterns required to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """

    @property
    def neutral(self) -> Optional[Union[Dict[str, str], str]]:
        """
        Define regex patterns to be ignored when running detection.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return None

    @property
    def negative(self) -> Optional[Union[Dict[str, str], str]]:
        """
        Define regex patterns prohibited to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return None

    @property
    @abstractmethod
    def match_list(self) -> List[str]:
        """
        List of texts that should activate the MelusineRegex.

        Returns:
            _: List of texts.
        """

    @property
    @abstractmethod
    def no_match_list(self) -> List[str]:
        """
        List of texts that should NOT activate the MelusineRegex.

        Returns:
            _: List of texts.
        """

    def _get_match(
        self, text: str, base_regex: Union[str, Dict[str, str]], regex_group: str = DEFAULT_MATCH_GROUP
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run specified regex on the input text and return a dict with matching group as key.

        Args:
            text: Text to apply regex on.
            base_regex: Regex to apply on text.
            regex_group: Name of the group the regex belongs to.

        Returns:
            Dict of regex matches for each regex group.
        """
        match_data_dict = {}

        if isinstance(base_regex, dict):
            for group, regex in base_regex.items():
                group_match_data = self._get_match(text, regex, group)
                match_data_dict.update(group_match_data)
        else:
            for match in re.finditer(base_regex, text, flags=self.REGEX_FLAGS):
                if not match_data_dict.get(regex_group):
                    match_data_dict[regex_group] = []

                # Get match position
                start, stop = match.span()

                match_data_dict[regex_group].append(
                    {
                        self.MATCH_START: start,
                        self.MATCH_STOP: stop,
                        self.MATCH_TEXT: text[start:stop],
                    }
                )

        return match_data_dict

    def ignore_text(
        self,
        text: str,
        match_data_dict: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """
        Replace neutral regex match text with substitution text to ignore it.

        Args:
            text: Input text.
            match_data_dict: Regex match results.

        Returns:
            _: Text with substituions.
        """
        if len(self.DEFAULT_SUBSTITUTION_PATTERN) > 1:
            raise AttributeError(f"Substitution pattern {self.DEFAULT_SUBSTITUTION_PATTERN} should be 1 character long")

        for _, match_list in match_data_dict.items():
            for match_data in match_list:
                start = match_data[self.MATCH_START]
                stop = match_data[self.MATCH_STOP]

                # Mask text to ignore
                text = text[:start] + self.DEFAULT_SUBSTITUTION_PATTERN * (stop - start) + text[stop:]

        return text

    def direct_match(self, text: str) -> bool:
        """
        Apply MelusineRegex patterns (neutral, negative and positive) on the input text.
        Return a boolean output of the match result.

        Args:
            text: input text.

        Returns:
            _: True if the MelusineRegex matches the input text.
        """
        result = self.match(text)
        return result[self.MATCH_RESULT]

    def match(self, text: str) -> Dict[str, Any]:
        """
        Apply MelusineRegex patterns (neutral, negative and positive) on the input text.
        Return a detailed output of the match results as a dict.

        Args:
            text: input text.

        Returns:
            _: Regex match results.
        """
        match_dict = {
            self.MATCH_RESULT: False,
            self.NEUTRAL_MATCH_FIELD: {},
            self.NEGATIVE_MATCH_FIELD: {},
            self.POSITIVE_MATCH_FIELD: {},
        }

        negative_match = False

        if self.neutral:
            neutral_match_data = self._get_match(text=text, base_regex=self.neutral)
            match_dict[self.NEUTRAL_MATCH_FIELD] = neutral_match_data

            text = self.ignore_text(text, neutral_match_data)

        if self.negative:
            negative_match_data = self._get_match(text=text, base_regex=self.negative)
            negative_match = bool(negative_match_data)
            match_dict[self.NEGATIVE_MATCH_FIELD] = negative_match_data

        positive_match_data = self._get_match(text=text, base_regex=self.positive)
        positive_match = bool(positive_match_data)
        match_dict[self.POSITIVE_MATCH_FIELD] = positive_match_data

        match_dict[self.MATCH_RESULT] = positive_match and not negative_match

        return match_dict

    def describe(self, text: str, position: bool = False) -> None:
        """
        User-friendly description of the regex match results.

        Args:
            text: Input text.
            position: If True, print regex match start and stop positions.
        """

        def _describe_match_field(match_field_data: Dict[str, List[Dict[str, Any]]]) -> None:
            """
            Format and print result description text.

            Args:
                match_field_data: Regex match result for a given field.
            """
            for group, match_list in match_field_data.items():
                for match_dict in match_list:
                    print(f"{indent}({group}) {match_dict[self.MATCH_TEXT]}")
                    if position:
                        print(f"{indent}start: {match_dict[self.MATCH_START]}")
                        print(f"{indent}stop: {match_dict[self.MATCH_STOP]}")

        indent = " " * 4
        match_data = self.match(text)

        if not any(
            [
                match_data[self.NEUTRAL_MATCH_FIELD],
                match_data[self.NEGATIVE_MATCH_FIELD],
                match_data[self.POSITIVE_MATCH_FIELD],
            ]
        ):
            print("The input text did not match anything.")

        if match_data[self.NEUTRAL_MATCH_FIELD]:
            print("The following text was ignored:")
            _describe_match_field(match_data[self.NEUTRAL_MATCH_FIELD])

        if match_data[self.NEGATIVE_MATCH_FIELD]:
            print("The following text matched negatively:")
            _describe_match_field(match_data[self.NEGATIVE_MATCH_FIELD])

        if match_data[self.POSITIVE_MATCH_FIELD]:
            print("The following text matched positively:")
            _describe_match_field(match_data[self.POSITIVE_MATCH_FIELD])

    def test(self) -> None:
        """
        Test the MelusineRegex on the match_list and no_match_list.
        """
        for text in self.match_list:
            match = self.match(text)
            assert match[self.MATCH_RESULT] is True, f"Expected match for text\n{text}\nObtained: {match}"

        for text in self.no_match_list:
            match = self.match(text)
            assert match[self.MATCH_RESULT] is False, f"Expected no match for text:\n{text}\nObtained: {match}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(positive:{self.positive},neutral:{self.neutral},negative:{self.negative})"
