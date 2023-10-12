"""
Integration test of a VacationReplyDetector.
"""
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from melusine.pipeline import MelusinePipeline


@pytest.mark.parametrize(
    "df, expected_result",
    [
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": [""],
                    "body": [
                        "Bonjour, \nActuellement en congé je prendrai connaissance"
                        + " de votre message ultérieurement.\nCordialement,"
                    ],
                }
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": [""],
                    "body": [
                        "Bonjour,\nje vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h.\nBien cordialement,\nJohn Smith."
                    ],
                }
            ),
            False,
        ),
    ],
)
def test_pipeline_from_config(df, expected_result):
    """
    Instanciate from a config and test the pipeline.
    """
    # Pipeline config key
    pipeline_key = "vacation_reply_pipeline"

    # Create pipeline from config
    pipeline = MelusinePipeline.from_config(config_key=pipeline_key)

    # Apply pipeline on data
    df_transformed = pipeline.transform(df)
    result = df_transformed["vacation_reply_result"][0]

    # Check
    assert result == expected_result
