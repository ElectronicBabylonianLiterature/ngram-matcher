from abc import ABC, abstractmethod
from operator import attrgetter, contains
from functools import singledispatchmethod
import datetime
from typing import Callable, Optional, Sequence, Set
import pandas as pd
import numpy as np

import requests
from tqdm import tqdm

from ebl_ngrams.document_model import (
    API_URL,
    DEFAULT_N_VALUES,
    BaseDocument,
    NGramSet,
    validate_n_values,
)
from ebl_ngrams.metrics import no_weight, weight_by_len
from copy import deepcopy


class IntegerEncoder:
    def __init__(self, items: Optional[Sequence] = None):
        self._encode = {}
        self._decode = {}

        if items:
            for key, item in enumerate(set(items)):
                self._encode[item] = key
                self._decode[key] = item

    def add_item(self, item):
        if item not in self.items:
            key = max(self._decode) + 1
            self._encode[item] = key
            self._decode[key] = item

    @property
    def items(self):
        return set(self._encode)

    def update(self, items):
        for item in set(items) - self.items:
            self.add_item(item)

    def decode(self, key):
        return self._decode[key]

    def encode(self, item):
        return self._encode[item]

    def encode_many(self, items):
        return set(map(self.encode, items))
class BaseCorpus(ABC):
    _collection: str
    documents: pd.Series

    def __init__(self, data, n_values: Sequence[int], show_progress=False, name=""):
        self.n_values = validate_n_values(n_values)
        self.retrieved_on = datetime.datetime.now()
        self.name = name
        self.data = data
        self._tqdm_config = {
            "total": len(data) if show_progress else 0,
            "desc": f"Building {self._collection} model",
            "disable": not show_progress,
        }
        self._idf_table = None
        self._ngrams = None
        self._tf_idf_n_values = n_values

        self.documents = self._load(data)

    @abstractmethod
    def _create_model(self, entry): ...

    @property
    def ngrams_by_document(self) -> pd.Series:
        return self.documents.map(attrgetter("ngrams"))

    def get_ngrams_by_document(self, *n_values) -> pd.Series:
        n_values = n_values or self.n_values
        return self.documents.map(
            lambda document: document.get_ngrams(*(n_values or self.n_values))
        )

    def _to_series(self, data: list) -> pd.Series:
        return pd.Series(data, index=[item.id_ for item in data], name=self._collection)

    @classmethod
    def load(
        cls,
        n_values: Sequence[int] = DEFAULT_N_VALUES,
        show_progress=True,
        name="",
        transform: Callable[[Sequence[dict]], Sequence[dict]] = None,
    ):
        response = requests.get(f"{API_URL}{cls._api_url}")
        response.raise_for_status()

        return cls(
            response.json() if transform is None else transform(response.json()),
            n_values,
            show_progress,
            name,
        )

    def _load(self, data: dict) -> pd.Series:
        return self._to_series(
            [
                self._create_model(entry, self.n_values)
                for entry in tqdm(data, **self._tqdm_config)
            ]
        )

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def __str__(self):
        return "<{} {}>".format(
            self.name or type(self).__name__,
            self.retrieved_on.strftime("%Y-%m-%d"),
        )

    def __repr__(self):
        return str(self)

    @property
    def ngrams(self) -> set:
        if self._ngrams is None:
            self._ngrams = {
                ngram for document in self.documents for ngram in document.get_ngrams()
            }
        return self._ngrams

    def get_ngrams(self, *n_values) -> set:
        n_values = n_values or self.n_values
        return (
            {ngram for ngram in self.ngrams if len(ngram) in n_values}
            if n_values
            else self.ngrams
        )

    def rebuild_ngrams(self, *n_values) -> "BaseCorpus":
        validate_n_values(n_values)
        corpus = deepcopy(self)
        corpus._reset_ngrams()
        corpus.documents = corpus.documents.map(lambda doc: doc.set_ngrams(*n_values))

        return corpus

    @singledispatchmethod
    def intersection(self, other):
        raise NotImplementedError(
            f"Cannot intersect {type(self).__name__} with {type(other).__name__}"
        )

    @intersection.register(BaseDocument)
    def _(self, other: BaseDocument, *n_values) -> pd.Series:
        n_values = n_values or self.n_values
        return pd.Series(
            np.vectorize(set.intersection)(
                other.get_ngrams(*n_values), self.get_ngrams_by_document(*n_values)
            ),
            index=self.documents.index,
            name=other.id_,
        )

    @singledispatchmethod
    def match(self, other):
        raise NotImplementedError(
            f"Cannot match {type(self).__name__} with {type(other).__name__}"
        )

    @match.register(BaseDocument)
    def _(self, other: BaseDocument, *n_values, length_weighting=False) -> pd.Series:
        n_values = n_values or self.n_values
        intersection = self.intersection(other, *n_values)
        weighted_sum = weight_by_len if length_weighting else no_weight

        intersection_sizes = weighted_sum(intersection)
        self_sizes = weighted_sum(self.get_ngrams_by_document(*n_values))
        other_size = weighted_sum(other.get_ngrams(*n_values))

        result = intersection_sizes / np.minimum(self_sizes, other_size)
        result = result.rename(other.id_)

        return result.sort_values(ascending=False)

    def _init_idf(self, *n_values: Sequence[int]) -> None:
        n_values = n_values or self.n_values
        skip_init = (
            set(n_values) == set(self._tf_idf_n_values) and self._idf_table is not None
        )

        if skip_init:
            return

        self._tf_idf_n_values = n_values

        unique_ngrams = pd.Series(list(self.get_ngrams(*n_values)))

        df = pd.DataFrame(
            np.vectorize(contains)(
                self.get_ngrams_by_document(*n_values), unique_ngrams.values[:, None]
            ),
            index=unique_ngrams,
        )
        N = len(self.documents) + 1
        docs_with_ngram = df.sum(axis=1) + 1

        idf = np.log(N / docs_with_ngram) + 1
        self._idf_table = idf.to_dict()

    def _weight_tf_idf(self, ngrams: set):
        return sum(self._idf_table.get(term, 0) for term in ngrams)

    def _weight_tf_idf_length(self, ngrams: set):
        return sum(self._idf_table.get(term, 0) * len(term) ** 2 for term in ngrams)

    def _reset_ngrams(self):
        self._ngrams = None
        self._idf_table = None

    @singledispatchmethod
    def match_tf_idf(self, other):
        raise NotImplementedError(
            f"Cannot match {type(self).__name__} with {type(other).__name__}"
        )

    @match_tf_idf.register(BaseDocument)
    def _(self, other: BaseDocument, *n_values, length_weighting=False) -> pd.Series:
        self._init_idf(*n_values)
        weight_func = (
            self._weight_tf_idf_length if length_weighting else self._weight_tf_idf
        )
        return (
            self.intersection(other, *n_values)
            .map(weight_func)
            .sort_values(ascending=False)
        )

    def filter(self, condition: Callable[[BaseDocument], bool]) -> "BaseCorpus":
        corpus = deepcopy(self)
        corpus._reset_ngrams()
        corpus.documents = corpus.documents.loc[corpus.documents.map(condition)]

        return corpus


@BaseCorpus.intersection.register
def _(self: BaseCorpus, other: BaseCorpus, *n_values):
    return pd.DataFrame(
        np.vectorize(set.intersection)(
            self.get_ngrams_by_document(*n_values).values[:, None],
            other.ngrams_by_document,
        ),
        index=self.documents.index,
        columns=other.documents.index,
    )


@BaseCorpus.match.register
def _(
    self: BaseCorpus, other: BaseCorpus, *n_values, length_weighting=False
) -> pd.DataFrame:
    n_values = n_values or self.n_values
    weighted_sum = weight_by_len if length_weighting else no_weight

    intersection_sizes = weighted_sum(self.intersection(other, *n_values))
    self_sizes = weighted_sum(self.get_ngrams_by_document(*n_values))
    other_sizes = weighted_sum(other.get_ngrams_by_document(*n_values))

    return intersection_sizes / np.minimum(
        self_sizes.values[:, None],
        other_sizes,
    )


@BaseCorpus.match_tf_idf.register
def _(self, other: BaseCorpus, *n_values, length_weighting=False) -> pd.DataFrame:
    self._init_idf(*n_values)
    weight_func = (
        self._weight_tf_idf_length if length_weighting else self._weight_tf_idf
    )
    return self.intersection(other).map(weight_func)
