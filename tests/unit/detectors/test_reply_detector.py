"""
Unit tests of the ReplyDetector.
"""

from tempfile import TemporaryDirectory

import pytest
from pandas import DataFrame

from melusine.detectors import ReplyDetector


def test_instantiation():
    """Instanciation unit test."""

    # Instantiate manually a detector
    detector = ReplyDetector(
        name="reply",
        header_column="clean_header",
    )
    assert isinstance(detector, ReplyDetector)


@pytest.mark.parametrize(
    "row, good_result",
    [
        (
            {"reply_text": "Devis habitation"},
            False,
        ),
        (
            {"reply_text": "tr: Devis habitation"},
            False,
        ),
        (
            {"reply_text": "re: Envoi d'un document de la Société Imaginaire"},
            True,
        ),
        (
            {"reply_text": "re : Virement"},
            True,
        ),
        (
            {"reply_text": ""},
            False,
        ),
    ],
)
def test_deterministic_detect(row, good_result):
    """Method unit test."""

    # Instanciate manually a detector
    detector = ReplyDetector(
        name="reply",
        header_column="clean_header",
    )
    # Test method
    row = detector.detect(row)
    res = row[detector.result_column]
    assert res == good_result


@pytest.mark.parametrize(
    "df_emails, expected_result",
    [
        (
            DataFrame(
                {
                    "clean_header": ["Re: Suivi de dossier"],
                }
            ),
            True,
        ),
        (
            DataFrame(
                {
                    "clean_header": ["Suivi de dossier"],
                }
            ),
            False,
        ),
        (
            DataFrame(
                {
                    "clean_header": ["Tr: Suivi de dossier"],
                }
            ),
            False,
        ),
        (
            DataFrame(
                {
                    "clean_header": [""],
                }
            ),
            False,
        ),
    ],
)
def test_transform(df_emails, expected_result):
    """Unit test of the transform() method."""

    # Copy for later load/save test
    df_copy = df_emails.copy()

    # Instantiate manually a detector
    detector = ReplyDetector(
        name="reply",
        header_column="clean_header",
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
                    "clean_header": ["Re: Suivi de dossier"],
                }
            ),
            True,
            {
                "ReplyRegex": {
                    "match_result": True,
                    "negative_match_data": {},
                    "neutral_match_data": {},
                    "positive_match_data": {"DEFAULT": [{"match_text": "re:", "start": 0, "stop": 3}]},
                },
                "reply_text": "re: suivi de dossier",
            },
        ),
    ],
)
def test_transform_debug_mode(df_emails, expected_result, expected_debug_info):
    """Unit test of the debug mode."""

    # Copy for later load/save test
    df_copy = df_emails.copy()

    # Instanciate manually a detector
    detector = ReplyDetector(
        name="reply",
        header_column="clean_header",
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
