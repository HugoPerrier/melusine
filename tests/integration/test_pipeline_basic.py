"""
Example script to fit a minimal preprocessing pipeline
"""
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
from pydantic import ValidationError

from melusine.conf import config
from melusine.pipeline import MelusinePipeline
from melusine.processors import Normalizer, RegexTokenizer


def test_pipeline_basic(dataframe_basic):
    """
    Train a pipeline by explicitly instatiating all the transformers.
    """
    # Input data
    df = dataframe_basic.copy()

    # Instantiate processors
    normalizer = Normalizer(lowercase=True, form="NFKD")
    tokenizer = RegexTokenizer()

    # Create pipeline
    pipe = MelusinePipeline(steps=[("normalizer", normalizer), ("tokenizer", tokenizer)], verbose=True)

    # Fit the pipeline and transform the data
    df_transformed = pipe.fit_transform(df)

    # Most basic test, check that the pipeline returns a pandas DataFrame
    assert isinstance(df_transformed, pd.DataFrame)


@pytest.mark.usefixtures("use_test_config")
def test_pipeline_from_config(dataframe_basic):
    """
    Train a pipeline using transformers defined in a pipeline config file.
    """
    # Input data
    df = dataframe_basic.copy()

    # Set config keys
    normalizer_key = "test_normalizer"
    tokenizer_key = "test_tokenizer"
    pipeline_key = "test_pipeline"

    # Pipeline configuration
    conf_pipeline_basic = {
        "steps": [
            {
                "class_name": "Normalizer",
                "module": "melusine.processors",
                "config_key": normalizer_key,
            },
            {
                "class_name": "RegexTokenizer",
                "module": "melusine.processors",
                "config_key": tokenizer_key,
            },
        ]
    }

    test_conf_dict = config.dict()
    test_conf_dict[pipeline_key] = conf_pipeline_basic
    config.reset(config_dict=test_conf_dict)

    # Create pipeline from a json config file (using config key "my_pipeline")
    pipe = MelusinePipeline.from_config(config_key=pipeline_key, verbose=True)

    # Fit the pipeline and transform the data
    df_transformed = pipe.fit_transform(df)

    # Make basic tests
    assert isinstance(df_transformed, pd.DataFrame)
    assert normalizer_key in pipe.named_steps
    assert tokenizer_key in pipe.named_steps


@pytest.mark.usefixtures("use_test_config")
def test_pipeline_from_dict(dataframe_basic):
    """
    Train a pipeline using transformers defined in a pipeline config file.
    """
    # Input data
    df = dataframe_basic.copy()

    # Set config keys
    normalizer_name = "normalizer"
    tokenizer_key = "test_tokenizer"

    # Pipeline configuration
    conf_pipeline_basic = {
        "steps": [
            {
                "name": normalizer_name,
                "class_name": "Normalizer",
                "module": "melusine.processors",
                "parameters": {
                    "form": "NFKD",
                    "input_columns": ["text"],
                    "lowercase": True,
                    "output_columns": ["text"],
                },
            },
            {
                "class_name": "RegexTokenizer",
                "module": "melusine.processors",
                "config_key": tokenizer_key,
            },
        ]
    }

    # Create pipeline from a json config file (using config key "my_pipeline")
    pipe = MelusinePipeline.from_config(config_dict=conf_pipeline_basic, verbose=True)

    # Fit the pipeline and transform the data
    df_transformed = pipe.fit_transform(df)

    # Make basic tests
    assert isinstance(df_transformed, pd.DataFrame)
    assert normalizer_name in pipe.named_steps
    assert tokenizer_key in pipe.named_steps


@pytest.mark.usefixtures("use_test_config")
def test_missing_config_key():
    """
    Train a pipeline using transformers defined in a pipeline config file.
    """
    # Set config keys
    normalizer_name = "normalizer"

    # Pipeline configuration
    conf_pipeline_basic = {
        "steps": [
            {
                "name": normalizer_name,
                "class_name": "Normalizer",
                "module": "melusine.processors",
                "parameters": {
                    "form": "NFKD",
                    "input_columns": ["text"],
                    "lowercase": True,
                    "output_columns": ["text"],
                },
            },
            {
                "class_name": "RegexTokenizer",
                "module": "melusine.processors",
            },
        ]
    }

    # Create pipeline from a json config file (using config key "my_pipeline")
    with pytest.raises(ValidationError):
        _ = MelusinePipeline.from_config(config_dict=conf_pipeline_basic, verbose=True)


@pytest.mark.usefixtures("use_test_config")
def test_invalid_config_key():
    """
    Train a pipeline using transformers defined in a pipeline config file.
    """
    incorrect_config_key = "INCORRECT_CONFIG_KEY"

    # Pipeline configuration
    conf_pipeline_basic = {
        "steps": [
            {
                "class_name": "Normalizer",
                "module": "melusine.processors",
                "config_key": incorrect_config_key,
            },
            {
                "class_name": "RegexTokenizer",
                "module": "melusine.processors",
            },
        ]
    }

    # Create pipeline from a json config file (using config key "my_pipeline")
    with pytest.raises(KeyError, match=incorrect_config_key):
        _ = MelusinePipeline.from_config(config_dict=conf_pipeline_basic, verbose=True)


@pytest.mark.usefixtures("use_test_config")
def test_missing_name():
    """
    Train a pipeline using transformers defined in a pipeline config file.
    """
    # Pipeline configuration
    conf_pipeline_basic = {
        "steps": [
            {
                "class_name": "Normalizer",
                "module": "melusine.processors",
                "parameters": {
                    "form": "NFKD",
                    "input_columns": ["text"],
                    "lowercase": True,
                    "output_columns": ["text"],
                },
            },
            {
                "class_name": "RegexTokenizer",
                "module": "melusine.processors",
                "config_key": "test_tokenizer",
            },
        ]
    }

    # Create pipeline from a json config file (using config key "my_pipeline")
    with pytest.raises(ValidationError):
        _ = MelusinePipeline.from_config(config_dict=conf_pipeline_basic, verbose=True)


@pytest.mark.usefixtures("use_test_config")
def test_missing_parameters():
    """
    Train a pipeline using transformers defined in a pipeline config file.
    """
    # Set config keys
    normalizer_name = "normalizer"

    # Pipeline configuration
    conf_pipeline_basic = {
        "steps": [
            {
                "name": normalizer_name,
                "class_name": "Normalizer",
                "module": "melusine.processors",
            },
            {
                "class_name": "RegexTokenizer",
                "module": "melusine.processors",
                "config_key": "test_tokenizer",
            },
        ]
    }

    # Create pipeline from a json config file (using config key "my_pipeline")
    with pytest.raises(ValidationError):
        _ = MelusinePipeline.from_config(config_dict=conf_pipeline_basic, verbose=True)


def test_missing_input_field(dataframe_basic):
    """
    Try to transform with an ill config pipeline.
    (The tokenizer step expects an input field "my_missing_field" which is not present)
    """
    # Input data
    df = dataframe_basic.copy()

    # Set config keys
    normalizer_name = "normalizer"
    tokenizer_name = "tokenizer"
    missing_field_name = "my_missing_field"

    # Pipeline configuration
    conf_pipeline_basic = {
        "steps": [
            {
                "name": normalizer_name,
                "class_name": "Normalizer",
                "module": "melusine.processors",
                "parameters": {
                    "input_columns": ["text"],
                    "output_columns": ["text"],
                },
            },
            {
                "name": tokenizer_name,
                "class_name": "RegexTokenizer",
                "module": "melusine.processors",
                "parameters": {
                    "input_columns": [missing_field_name],
                    "output_columns": ["tokens"],
                },
            },
        ]
    }

    # Create pipeline from a json config file (using config key "my_pipeline")
    pipe = MelusinePipeline.from_config(config_dict=conf_pipeline_basic, verbose=True)

    # Fit the pipeline and transform the data
    with pytest.raises(ValueError, match=rf"(?s){tokenizer_name}.*{missing_field_name}"):
        pipe.transform(df)
