from abc import ABC
import datetime
from functools import singledispatchmethod
import json
from typing import Dict, Sequence, Set
import pandas as pd

from ebl_ngrams.metrics import no_weight, weight_by_len

UNKNOWN_SIGN = "X"
LINE_SEP = "#"
DEFAULT_N_VALUES = (1, 2, 3)
API_URL = "https://www.ebl.lmu.de/api/"


def extract_ngrams(signs: pd.Series, n_values: Sequence[int]):
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
    signs = signs.str.strip()
    signs.iloc[:-1] = signs.iloc[:-1].add(f" {LINE_SEP}")
    signs = signs[~signs.str.fullmatch(rf"[{UNKNOWN_SIGN}{LINE_SEP}\s]*")]
    signs = signs.str.split().explode()

    return signs[signs.ne(UNKNOWN_SIGN)]


def linewise_ngrams(signs: pd.Series, n_values: Sequence[int]) -> pd.Series:
    return extract_ngrams(signs, n_values).reset_index(drop=True)


def postprocess(signs: pd.Series):
    signs = signs[~signs.map(set).map(lambda ngram: ngram <= {"X"})]
    return set(signs)


def validate_n_values(n_values: Sequence[int]):
    if any(n <= 0 for n in n_values):
        raise ValueError("All n values must be greater than zero.")
    if not any(n_values):
        raise ValueError("Must pass at least one non-zero n value.")


class BaseDocument(ABC):
    def __init__(self, id_: str, signs: str, n_values=DEFAULT_N_VALUES):
        self.id_ = self.url = id_
        self.signs = signs
        self.n_values = n_values
        self.retrieved_on = datetime.datetime.now()

    @classmethod
    def load_json(cls, path: str, n_values=DEFAULT_N_VALUES) -> "BaseDocument":
        with open(path) as jf:
            data = json.load(jf)
        return cls(data, n_values)

    def intersection(self, other, *n_values) -> set:
        A = self.get_ngrams(*n_values)
        B = other.get_ngrams(*n_values)

    @singledispatchmethod
    def intersection(self, other):
        raise NotImplementedError(
            f"Cannot intersect {type(self).__name__} with {type(other).__name__}"
        )

    @singledispatchmethod
    def match(self, other):
        raise NotImplementedError(
            f"Cannot match {type(self).__name__} with {type(other).__name__}"
        )

    def get_ngrams(self, *n_values) -> set:
        return (
            {ngram for ngram in self.ngrams if len(ngram) in n_values}
            if n_values
            else self.ngrams
        )

    def __str__(self):
        return "<{} {} {}>".format(
            type(self).__name__,
            self.url,
            self.retrieved_on.strftime("%Y-%m-%d"),
        )

    def __repr__(self):
        return str(self)

    @property
    def _vocab(self) -> Set[str]:
        return {sign for ngram in self.ngrams for sign in ngram}


@BaseDocument.intersection.register
def _(self: BaseDocument, other: BaseDocument, *n_values) -> set:
    A = self.get_ngrams(*n_values)
    B = other.get_ngrams(*n_values)

    return A & B


@BaseDocument.match.register
def match(
    self: BaseDocument, other: BaseDocument, *n_values, length_weighting=False
) -> float:
    intersection = self.intersection(other, *n_values)
    weighted_sum = weight_by_len if length_weighting else no_weight

    intersection_size = weighted_sum(intersection)
    self_size = weighted_sum(self.get_ngrams(*n_values))
    other_size = weighted_sum(other.get_ngrams(*n_values))

    return (
        intersection_size / min(self_size, other_size)
        if self_size and other_size
        else 0.0
    )
