import re
import pandas as pd


class TextPreprocessor:
    """
    Lightweight text preprocessing module for NLP models.
    Designed for sentiment, emotion, and risk classification tasks.
    """

    def __init__(self):
        pass

    # -----------------------------
    # BASIC CLEANING
    # -----------------------------
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning:
        - lowercase
        - remove URLs
        - remove special characters
        - remove extra spaces
        """

        text = str(text).lower()

        # remove URLs
        text = re.sub(r"http\S+", "", text)

        # remove non-alphabet characters
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # -----------------------------
    # BATCH PROCESSING
    # -----------------------------
    def transform_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Applies preprocessing to a dataframe column
        and returns updated dataframe with clean text.
        """

        df = df.copy()

        df["clean_text"] = df[text_column].apply(self.clean_text)

        return df