"""
Integration test of a Reply Detector.
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
                    "header": ["Re: Suivi de dossier"],
                    "body": ["Bonjour,\nle traitement de ma demande est deplorable.\nje suis tres en colere.\n"],
                }
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["re: Envoi d'un document de la Société Imaginaire"],
                    "body": ["Bonjour,\nLe traitement de ma demande est déplorable.\nJe suis très en colère.\n"],
                }
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["te: Virement"],
                    "body": [
                        "Bonjour,\nJe vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h.\nBien cordialement,\nJohn Smith."
                    ],
                }
            ),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": [""],
                    "body": [
                        "Bonjour,\nJe vous confirme l'annulation du rdv du 01/01/2022 "
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
    pipeline_key = "reply_pipeline"

    # Create pipeline from config
    pipeline = MelusinePipeline.from_config(config_key=pipeline_key)

    # Apply pipeline on data
    df_transformed = pipeline.transform(df)
    result = df_transformed["reply_result"][0]

    # Check
    assert result == expected_result
