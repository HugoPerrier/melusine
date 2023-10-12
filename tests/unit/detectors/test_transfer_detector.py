"""
Unit tests of the TransferDetector.
"""

from tempfile import TemporaryDirectory

import pytest
from pandas import DataFrame

from melusine.detectors import TransferDetector
from melusine.message import Message


def test_instanciation():
    """Instanciation unit test."""

    detector = TransferDetector(name="transfer", header_column="det_clean_header", messages_column="messages")
    assert isinstance(detector, TransferDetector)


@pytest.mark.parametrize(
    "row, good_result",
    [
        (
            {
                "reply_text": "tr: Devis habitation",
                "messages": [
                    Message(
                        meta="",
                        text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h. Bien cordialement, John Smith.",
                        tags=[
                            ("HELLO", "Bonjour,"),
                            (
                                "BODY",
                                "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                            ),
                            ("GREETINGS", "Cordialement, John Smith."),
                        ],
                    )
                ],
            },
            True,
        ),
        (
            {
                "reply_text": "re: Envoi d'un document de la Société Imaginaire",
                "messages": [
                    Message(
                        meta="this is meta",
                        text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h. Bien cordialement, John Smith.",
                        tags=[
                            ("HELLO", "Bonjour,"),
                            (
                                "BODY",
                                "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                            ),
                            ("GREETINGS", "Cordialement, John Smith."),
                        ],
                    )
                ],
            },
            True,
        ),
        (
            {
                "reply_text": "re: Virement",
                "messages": [
                    Message(
                        meta="",
                        text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h. Bien cordialement, John Smith.",
                        tags=[
                            ("HELLO", "Bonjour,"),
                            (
                                "BODY",
                                "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                            ),
                            ("GREETINGS", "Cordialement, John Smith."),
                        ],
                    )
                ],
            },
            False,
        ),
        (
            {
                "reply_text": "",
                "messages": [
                    Message(
                        meta="",
                        text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                        + "à 16h. Bien cordialement, John Smith.",
                        tags=[
                            ("HELLO", "Bonjour,"),
                            (
                                "BODY",
                                "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                            ),
                            ("GREETINGS", "Cordialement, John Smith."),
                        ],
                    )
                ],
            },
            False,
        ),
    ],
)
def test_deterministic_detect(row, good_result):
    """Method unit test."""

    # Instanciate manually a detector
    detector = TransferDetector(
        name="transfer",
        header_column="det_clean_header",
        messages_column="messages",
    )
    row = detector.detect(row)
    res = row[detector.result_column]
    assert res == good_result


@pytest.mark.parametrize(
    "df_emails, expected_result",
    [
        (
            DataFrame(
                {
                    "det_clean_header": "tr: Rdv",
                    "messages": [
                        [
                            Message(
                                meta="",
                                text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                                + "à 16h. Bien cordialement, John Smith.",
                                tags=[
                                    ("HELLO", "Bonjour,"),
                                    (
                                        "BODY",
                                        "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                                    ),
                                    ("GREETINGS", "Cordialement, John Smith."),
                                ],
                            )
                        ]
                    ],
                }
            ),
            True,
        ),
    ],
)
def test_transform(df_emails, expected_result):
    """Unit test of the transform() method."""

    # Copy for later load/save test
    df_copy = df_emails.copy()

    # Instanciate manually a detector
    detector = TransferDetector(
        name="transfer",
        header_column="det_clean_header",
        messages_column="messages",
    )

    # Get result column name
    res_col = detector.result_column

    # Apply the detector on data
    df_emails = detector.transform(df_emails)

    # Verify result
    result = df_emails[res_col][0]
    assert result == expected_result


@pytest.mark.parametrize(
    "df_emails, expected_result, expected_debug_info",
    [
        (
            DataFrame(
                {
                    "det_clean_header": ["Tr: Suivi de dossier"],
                    "messages": [
                        [
                            Message(
                                meta="",
                                text="Bonjour, je vous confirme l'annulation du rdv du 01/01/2022 "
                                + "à 16h. Bien cordialement, John Smith.",
                                tags=[
                                    ("HELLO", "Bonjour,"),
                                    (
                                        "BODY",
                                        "je vous confirme l'annulation du rdv du 01/01/2022 " + "à 16h. ",
                                    ),
                                    ("GREETINGS", "Cordialement, John Smith."),
                                ],
                            )
                        ]
                    ],
                }
            ),
            True,
            {
                "reply_text": "tr: suivi de dossier",
                "messages[0].meta": "",
                "TransferRegex": {
                    "match_result": True,
                    "negative_match_data": {},
                    "neutral_match_data": {},
                    "positive_match_data": {"DEFAULT": [{"match_text": "tr:", "start": 0, "stop": 3}]},
                },
            },
        ),
    ],
)
def test_transform_debug_mode(df_emails, expected_result, expected_debug_info):
    """Unit test of the debug mode."""

    # Copy for later load/save test
    df_copy = df_emails.copy()

    # Instanciate manually a detector
    detector = TransferDetector(
        name="transfer",
        header_column="det_clean_header",
        messages_column="messages",
    )

    # Get column names
    res_col = detector.result_column
    debug_dict_col = detector.debug_dict_col

    # Transform data
    df_emails.debug = True
    df_emails = detector.transform(df_emails)

    # Collect results
    result = df_emails[res_col].iloc[0]
    debug_result = df_emails[debug_dict_col].iloc[0]

    # Test result
    assert result == expected_result
    assert debug_result == expected_debug_info
