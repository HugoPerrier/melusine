import re
import pandas as pd

from melusine.base import MelusineDetector


# --8<-- [start:detector]
class MyVirusDetector(MelusineDetector):
    OUTPUT_RESULT_COLUMN = "virus_result"
    TMP_DETECTION_INPUT_COLUMN = "detection_input"
    TMP_POSITIVE_REGEX_MATCH = "positive_regex_match"
    TMP_NEGATIVE_REGEX_MATCH = "negative_regex_match"

    # --8<-- [start:detector_init]
    def __init__(self, body_column: str, header_column: str):
        self.body_column = body_column
        self.header_column = header_column

        super().__init__(
            input_columns=[self.body_column, self.header_column],
            output_columns=[self.OUTPUT_RESULT_COLUMN],
            name="virus"
        )
        # --8<-- [end:detector_init]

    # --8<-- [start:pre_detect]
    def pre_detect(self, row, debug_mode=False):
        effective_text = row[self.header_column] + "\n" + row[self.body_column]
        row[self.TMP_DETECTION_INPUT_COLUMN] = effective_text

        return row
        # --8<-- [end:pre_detect]

    # --8<-- [start:detect]
    def detect(self, row, debug_mode=False):
        text = row[self.TMP_DETECTION_INPUT_COLUMN]
        positive_regex = r"virus"
        negative_regex = r"corona[ _]virus"

        positive_match = re.search(positive_regex, text)
        negative_match = re.search(negative_regex, text)

        row[self.TMP_POSITIVE_REGEX_MATCH] = bool(positive_match)
        row[self.TMP_NEGATIVE_REGEX_MATCH] = bool(negative_match)

        return row
        # --8<-- [end:detect]

    # --8<-- [start:post_detect]
    def post_detect(self, row, debug_mode=False):
        if row[self.TMP_POSITIVE_REGEX_MATCH] and not row[self.TMP_NEGATIVE_REGEX_MATCH]:
            row[self.OUTPUT_RESULT_COLUMN] = True
        else:
            row[self.OUTPUT_RESULT_COLUMN] = False

        return row

    # --8<-- [end:post_detect]
# --8<-- [end:detector]


def run():
    # --8<-- [start:run]
    df = pd.DataFrame([
        {"body": "This is a dangerous virus", "header": "test"},
        {"body": "test", "header": "test"},
        {"body": "test", "header": "viruses are dangerous"},
        {"body": "corona virus is annoying", "header": "test"},
    ])

    detector = MyVirusDetector(body_column="body", header_column="header")

    df = detector.transform(df)
    # --8<-- [end:run]

    # --8<-- [start:run_debug]
    df = pd.DataFrame([
        {"body": "This is a dangerous virus", "header": "test"},
        {"body": "test", "header": "test"},
        {"body": "test", "header": "viruses are dangerous"},
        {"body": "corona virus is annoying", "header": "test"},
    ])
    df.debug = True

    df = detector.transform(df)
    # --8<-- [end:run_debug]

    return df
