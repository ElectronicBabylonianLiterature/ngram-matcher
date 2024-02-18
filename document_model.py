import datetime
import json
import os
import pickle
from typing import Dict, Literal, Sequence

from pymongo import MongoClient
from util import overlap_coefficient
import pandas as pd


UNKNOWN_SIGN = "X"
LINE_SEP = "#"
DEFAULT_N_VALUES = (1, 2, 3)


class DocumentNotFoundError(Exception):
    pass


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


def fetch(
    query: dict,
    projection: dict,
    collection: Literal["chapters", "fragments"],
    db="ebldev",
    uri=None,
):
    client = MongoClient(uri or os.environ["MONGODB_URI"])
    database = client.get_database(db)

    if data := database.get_collection(collection).find_one(
        query,
        projection=projection,
    ):
        return data
    else:
        raise DocumentNotFoundError(f"No document found for {query!r}")


class DocumentModel:

    def __init__(self, id_: str, signs: str, n_values=DEFAULT_N_VALUES):
        self.id_ = self.url = id_
        self.signs = signs
        self.n_values = n_values
        self.retrieved_on = datetime.datetime.now()
        self.is_compressed = False

    @classmethod
    def load_json(cls, path: str, n_values=DEFAULT_N_VALUES) -> "DocumentModel":
        with open(path) as jf:
            data = json.load(jf)
        return cls(data, n_values)

    def intersection(self, other, *n_values):
        A = self.get_ngrams(*n_values)
        B = other.get_ngrams(*n_values)

        return A & B

    def overlap_coefficient(self, other, *n_values):
        A = self.get_ngrams(*n_values)
        B = other.get_ngrams(*n_values)

        return overlap_coefficient(A, B)

    def get_ngrams(self, *n_values):
        return (
            {ngram for ngram in self.ngrams if len(ngram) in n_values}
            if n_values
            else self.ngrams
        )

    def __str__(self):
        return "<{} {} {}>".format(
            self.__class__.__name__,
            self.url,
            self.retrieved_on.strftime("%Y-%m-%d"),
        )

    def __repr__(self):
        return str(self)

    @property
    def _vocab(self):
        return {sign for ngram in self.ngrams for sign in ngram}

    def _compress(self, encoder: Dict[str, int]):
        if not self.is_compressed:

            def encode_ngram(ngram):
                return tuple(encoder[sign] for sign in ngram)

            self.ngrams = set(map(encode_ngram, self.ngrams))
            self.is_compressed = True

    def _decompress(self, decoder: Dict[int, str]):
        if self.is_compressed:

            def decode_ngram(ngram):
                return tuple(decoder[id_] for id_ in ngram)

            self.ngrams = set(map(decode_ngram, self.ngrams))
            self.is_compressed = False

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def open(cls, path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            raise TypeError(
                f"{cls.__name__} cannot load {model.__class__.__name__} data"
            )

        return model
