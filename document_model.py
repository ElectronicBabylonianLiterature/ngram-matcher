from util import overlap_coefficient
import pandas as pd


UNKNOWN_SIGN = "X"
LINE_SEP = "#"


class DocumentNotFoundError(Exception):
    pass


class DocumentModel:
    def intersection(self, other, *n_values):
        A = self.get_ngrams(*n_values)
        B = other.get_ngrams(*n_values)

        return A & B

    def similarity(self, other, *n_values):
        A = self.get_ngrams(*n_values)
        B = other.get_ngrams(*n_values)

        return overlap_coefficient(A, B)


def extract_ngrams(signs: pd.Series, n_values=(1, 2, 3)):
    subframes = [
        pd.concat([signs.shift(-i) for i in range(n)], axis=1)
        .dropna()
        .agg(tuple, axis=1)
        for n in n_values
    ]
    ngrams = pd.concat(subframes)

    return ngrams.drop_duplicates()


def preprocess(raw_signs: str) -> pd.Series:
    signs = pd.Series(raw_signs)

    signs = signs.str.split("\n").explode().reset_index(drop=True)
    signs = signs.str.replace(
        rf"{UNKNOWN_SIGN}[\s{UNKNOWN_SIGN}]*", f"{UNKNOWN_SIGN} ", regex=True
    ).str.strip()
    signs = signs[~signs.str.fullmatch(rf"[{UNKNOWN_SIGN}\s]*")]
    signs = signs.add(f" {LINE_SEP}")

    return signs.str.split().explode()


def linewise_ngrams(signs: pd.Series, n_values=(1, 2, 3)) -> pd.Series:
    return (
        signs.groupby(level=0)
        .apply(extract_ngrams, n_values=n_values)
        .reset_index(drop=True)
    )


def postprocess(signs: pd.Series):
    signs = signs[~signs.map(set).map(lambda ngram: ngram <= {"X", "#"})]
    return set(signs)
