"""
Integration test of a transfer Detector.
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
                    "header": ["tr :Suivi de dossier"],
                    "body": [
                        "",
                    ],
                }
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["fwd: Envoi d'un document de la Société Imaginaire"],
                    "body": [
                        "Bonjour,\n\n\n\n\n\nUn taux d’humidité de 30% a été relevé le 19/04/2022.\n\n\n\nNous reprendrons contact avec l’assurée"
                        + " en Aout 2022.\n\n\n\n\n\n\nBien cordialement,\n\n\n\n\n\nNuméro Auxiliaire : 116113 T / 116133 J\n\n\n\n\n\n\n\n\nABOU"
                        + " ELELA Donia\n\n\nSté LVP\n-\n\nL\na\nV\nalorisation du\nP\natrimoine\n\n\n2, rue de la Paix\n\n\n94300 VINCENNES"
                        + "\n\n\n\n\n\n\n\nTél :    0143740992\n\n\nPort :  0767396737\n\n\nhttp://lvpfrance.fr\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                        + "\n\n\nDe :\nAccueil - Lucile RODRIGUES <accueil@lvpfrance.fr>\n\n\n\nEnvoyé :\njeudi 13 janvier 2022 15:26\n\n\nÀ "
                        + ":\nCommercial <etudes7@lvpfrance.fr>\n\n\nObjet :\nTR: Evt : M211110545P survenu le 15/10/2021 - Intervention entreprise"
                        + " partenaire\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDe :\n\n\ngestionsinistre@maif.fr\n[\nmailto:gestionsinistre@maif.fr\n]\n\n\n\n"
                        + "Envoyé :\njeudi 13 janvier 2022 15:13\n\n\nÀ :\nAccueil - Lucile RODRIGUES\n\n\nObjet :\nEvt : M211110545P survenu le 15/10/2021"
                        + " - Intervention entreprise partenaire\n\n\n\n\n\nMerci de bien vouloir prendre connaissance du document ci-joint.\n\n\n\nSentiments"
                        + "mutualistes.\n\nLa MAIF",
                    ],
                }
            ),
            True,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["Virement"],
                    "body": [
                        "Bonjour,\n\n\n\n\n\nUn taux d’humidité de 30% a été relevé le 19/04/2022.\n\n\n\nNous reprendrons contact avec l’assurée"
                        + " en Aout 2022.\n\n\n\n\n\n\nBien cordialement,\n\n\n\n\n\nNuméro Auxiliaire : 116113 T / 116133 J\n\n\n\n\n\n\n\n\nABOU"
                        + " ELELA Donia\n\n\nSté LVP\n-\n\nL\na\nV\nalorisation du\nP\natrimoine\n\n\n2, rue de la Paix\n\n\n94300 VINCENNES"
                        + "\n\n\n\n\n\n\n\nTél :    0143740992\n\n\nPort :  0767396737\n\n\nhttp://lvpfrance.fr\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                        + "\n\n\nDe :\nAccueil - Lucile RODRIGUES <accueil@lvpfrance.fr>\n\n\n\nEnvoyé :\njeudi 13 janvier 2022 15:26\n\n\nÀ "
                        + ":\nCommercial <etudes7@lvpfrance.fr>\n\n\nObjet :\nTR: Evt : M211110545P survenu le 15/10/2021 - Intervention entreprise"
                        + " partenaire\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDe :\n\n\ngestionsinistre@maif.fr\n[\nmailto:gestionsinistre@maif.fr\n]\n\n\n\n"
                        + "Envoyé :\njeudi 13 janvier 2022 15:13\n\n\nÀ :\nAccueil - Lucile RODRIGUES\n\n\nObjet :\nEvt : M211110545P survenu le 15/10/2021"
                        + " - Intervention entreprise partenaire\n\n\n\n\n\nMerci de bien vouloir prendre connaissance du document ci-joint.\n\n\n\nSentiments"
                        + "mutualistes.\n\nLa MAIF",
                    ],
                }
            ),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "from": ["test@gmail.com"],
                    "header": ["tr: virement"],
                    "body": [
                        "Bonjour,\n\n\n\n\n\nUn taux d’humidité de 30% a été relevé le 19/04/2022.\n\n\n\nNous reprendrons contact avec l’assurée"
                        + " en Aout 2022.\n\n\n\n\n\n\nBien cordialement,\n\n\n\n\n\nNuméro Auxiliaire : 116113 T / 116133 J\n\n\n\n\n\n\n\n\nABOU"
                        + " ELELA Donia\n\n\nSté LVP\n-\n\nL\na\nV\nalorisation du\nP\natrimoine\n\n\n2, rue de la Paix\n\n\n94300 VINCENNES"
                        + "\n\n\n\n\n\n\n\nTél :    0143740992\n\n\nPort :  0767396737\n\n\nhttp://lvpfrance.fr\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
                        + "\n\n\nDe :\nAccueil - Lucile RODRIGUES <accueil@lvpfrance.fr>\n\n\n\nEnvoyé :\njeudi 13 janvier 2022 15:26\n\n\nÀ "
                        + ":\nCommercial <etudes7@lvpfrance.fr>\n\n\nObjet :\nTR: Evt : M211110545P survenu le 15/10/2021 - Intervention entreprise"
                        + " partenaire\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDe :\n\n\ngestionsinistre@maif.fr\n[\nmailto:gestionsinistre@maif.fr\n]\n\n\n\n"
                        + "Envoyé :\njeudi 13 janvier 2022 15:13\n\n\nÀ :\nAccueil - Lucile RODRIGUES\n\n\nObjet :\nEvt : M211110545P survenu le 15/10/2021"
                        + " - Intervention entreprise partenaire\n\n\n\n\n\nMerci de bien vouloir prendre connaissance du document ci-joint.\n\n\n\nSentiments"
                        + "mutualistes.\n\nLa MAIF",
                    ],
                }
            ),
            True,
        ),
    ],
)
def test_pipeline_from_config(df, expected_result):
    """
    Instanciate from a config and test the pipeline.
    """
    # Pipeline config key
    pipeline_key = "transfer_pipeline"

    # Create pipeline from config
    pipeline = MelusinePipeline.from_config(config_key=pipeline_key)

    # Apply pipeline on data
    df_transformed = pipeline.transform(df)
    result = df_transformed["transfer_result"][0]

    # Check
    assert result == expected_result
