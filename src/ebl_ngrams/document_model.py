from abc import ABC, abstractmethod
import datetime
from functools import singledispatchmethod
from itertools import tee
import json
import re
from typing import Sequence, Set, Tuple

from ebl_ngrams.metrics import no_weight, weight_by_len

UNKNOWN_SIGN = "X"
LINE_SEP = "#"
DEFAULT_N_VALUES = (1, 2, 3)
API_URL = "https://www.ebl.lmu.de/api/"

NGramSet = Set[Tuple[str]]


def validate_n_values(n_values: Sequence[int]):
    if any(n <= 0 for n in n_values):
        raise ValueError("All n values must be greater than zero.")
    if not any(n_values):
        raise ValueError("Must pass at least one non-zero n value.")


def ngrams(signs: Sequence[str], n) -> Set[Tuple[str]]:
    iterables = tee(signs, n)

    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return set(zip(*iterables))


def ngrams_multi_n(signs: Sequence[str], *n_values) -> NGramSet:
    validate_n_values(n_values)
    return set.union(*(ngrams(signs, n_) for n_ in n_values))


def preprocess(signs: str) -> Sequence[str]:
    lines = [line.strip() for line in signs.split("\n")]
    lines = [
        line
        for line in lines
        if not re.fullmatch(rf"[{UNKNOWN_SIGN}{LINE_SEP}\s]*", line)
    ]
    return f" {LINE_SEP} ".join(lines).split()


def postprocess(ngrams: NGramSet) -> NGramSet:
    return {ngram for ngram in ngrams if "X" not in ngram}


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

    @abstractmethod
    def set_ngrams(self, *n_values) -> "BaseDocument": ...

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
