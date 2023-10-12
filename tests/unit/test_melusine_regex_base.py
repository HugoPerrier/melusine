from typing import Dict, List, Optional, Union

from melusine.base import MelusineRegex


class VirusRegex(MelusineRegex):
    """
    Detect computer viruses but not software bugs.
    """

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        return r"virus"

    @property
    def neutral(self) -> Optional[Union[str, Dict[str, str]]]:
        return dict(
            NEUTRAL_MEDICAL_VIRUS="corona virus",
            NEUTRAL_INSECT="ladybug",
        )

    @property
    def negative(self) -> Optional[Union[str, Dict[str, str]]]:
        return dict(
            NEGATIVE_BUG="bug",
        )

    @property
    def match_list(self) -> List[str]:
        return [
            "This email contains a virus",
            "There is a virus in the ladybug software",
            "The corona virus is not a computer virus",
        ]

    @property
    def no_match_list(self) -> List[str]:
        return [
            "This process just had a bug",
            "This is a bug not a virus",
            "There are ladybugs on the windows",
        ]


def test_method_test():
    regex = VirusRegex()
    regex.test()
    assert True


def test_match_method():
    regex = VirusRegex()
    match_data = regex.match("The computer virus in the ladybug software caused a bug in the corona virus dashboard")

    assert match_data[MelusineRegex.MATCH_RESULT] is False
    assert match_data[MelusineRegex.POSITIVE_MATCH_FIELD] == {
        "DEFAULT": [{"match_text": "virus", "start": 13, "stop": 18}]
    }
    assert match_data[MelusineRegex.NEUTRAL_MATCH_FIELD] == {
        "NEUTRAL_INSECT": [{"match_text": "ladybug", "start": 26, "stop": 33}],
        "NEUTRAL_MEDICAL_VIRUS": [{"match_text": "corona virus", "start": 63, "stop": 75}],
    }
    assert match_data[MelusineRegex.NEGATIVE_MATCH_FIELD] == {
        "NEGATIVE_BUG": [{"match_text": "bug", "start": 52, "stop": 55}]
    }


def test_direct_match_method():
    regex = VirusRegex()

    bool_match_result = regex.direct_match("The computer virus")

    assert bool_match_result is True

    bool_match_result = regex.direct_match(
        "The computer virus in the ladybug software caused a bug in the corona virus dashboard"
    )

    assert bool_match_result is False


def test_describe_method(capfd):
    regex = VirusRegex()
    regex.describe("The computer virus in the ladybug software caused a bug in the corona virus dashboard")
    out, err = capfd.readouterr()
    assert "NEGATIVE_BUG" in out
    assert "start" not in out

    regex.describe(
        "The computer virus in the ladybug software caused a bug in the corona virus dashboard",
        position=True,
    )
    out, err = capfd.readouterr()
    assert "NEGATIVE_BUG" in out
    assert "start" in out


def test_no_negative():
    class NoNegativeNoNeutralRegex(MelusineRegex):
        positive = r"hey"

        @property
        def match_list(self) -> List[str]:
            return []

        @property
        def no_match_list(self) -> List[str]:
            return []

    result = NoNegativeNoNeutralRegex().match("hey you")

    assert result[MelusineRegex.MATCH_RESULT] is True
