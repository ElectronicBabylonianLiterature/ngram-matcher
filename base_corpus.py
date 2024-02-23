from operator import attrgetter, contains
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial, singledispatchmethod
import pickle
import datetime
from typing import Literal
import pandas as pd
import numpy as np

from pymongo import MongoClient
from tqdm import tqdm
from chapter_model import ChapterModel

from document_model import DEFAULT_N_VALUES, DocumentModel
from fragment_model import FragmentModel
from metrics import no_weight, weight_by_len


def fetch_all(
    query: dict,
    projection: dict,
    collection: Literal["chapters", "fragments"],
    db="ebldev",
    uri=None,
    **kwargs,
):
    client = MongoClient(uri or os.environ["MONGODB_URI"])
    database = client.get_database(db)

    return database.get_collection(collection).find(
        query, projection=projection, **kwargs
    )


class BaseCorpus:
    _collection = ""

    def __init__(self, data, n_values, show_progress=False, threading=False, name=""):
        self.n_values = n_values
        self.is_compressed = False
        self.retrieved_on = datetime.datetime.now()
        self.name = name
        self._tqdm_config = {
            "total": len(data) if show_progress else 0,
            "desc": "Building model",
            "disable": not show_progress,
        }
        self.idf_table = None
        self._ngrams = None

    def _compress(self):
        if not self.is_compressed:
            encoder = {word: index for index, word in enumerate(self._vocab)}
            for document in self:
                document._compress(encoder)
            self.is_compressed = True

    def _decompress(self):
        if self.is_compressed:
            decoder = dict(enumerate(self._vocab))
            for document in self:
                document._decompress(decoder)
            self.is_compressed = False

    def save(self, path: str):
        self._compress()
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._decompress()

    @property
    def ngrams_by_document(self):
        return self.documents.map(attrgetter("ngrams"))

    @classmethod
    def open(cls, path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            raise TypeError(f"{cls.__name__} cannot load {type(model).__name__} data")

        model._decompress()

        return model

    def to_series(self, data: list) -> pd.Series:
        return pd.Series(data, index=[item.id_ for item in data], name=self._collection)

    @classmethod
    def load(
        cls,
        n_values=DEFAULT_N_VALUES,
        db="ebldev",
        uri=None,
        show_progress=True,
        threading=True,
        name="",
        query={},
        **kwargs,
    ):
        data = fetch_all(
            {**cls._query, **query},
            projection=cls._projection,
            collection=cls._collection,
            db=db,
            uri=uri,
            **kwargs,
        )
        return cls(
            list(data) if show_progress else data,
            n_values,
            show_progress,
            name,
            threading,
        )

    def _load(self, data: dict):
        return self.to_series(
            [
                self._create_model(entry, self.n_values)
                for entry in tqdm(data, **self._tqdm_config)
            ]
        )

    def _load_threading(self, data: dict):
        with ProcessPoolExecutor() as executor:
            result = list(
                tqdm(
                    executor.map(
                        partial(self._create_model, n_values=self.n_values), data
                    ),
                    **self._tqdm_config,
                )
            )

        return self.to_series(result)

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

    def __and__(self, other):
        return self.intersection(other)

    @property
    def ngrams(self):
        if self._ngrams is None:
            self._ngrams = {
                ngram for document in self.documents for ngram in document.get_ngrams()
            }
        return self._ngrams

    @singledispatchmethod
    def intersection(self, other):
        raise NotImplementedError(
            f"Cannot intersect {type(self).__name__} " f"with {type(other).__name__}"
        )

    @intersection.register(FragmentModel)
    @intersection.register(ChapterModel)
    def _(self, other: DocumentModel) -> pd.Series:
        return pd.Series(
            np.vectorize(set.intersection)(other.ngrams, self.ngrams_by_document),
            index=self.documents.index,
            name=other.id_,
        )

    @singledispatchmethod
    def match(self, other):
        raise NotImplementedError(
            f"Cannot match {type(self).__name__} " f"with {type(other).__name__}"
        )

    @match.register(FragmentModel)
    @match.register(ChapterModel)
    def _(self, other: DocumentModel, length_weighting=False) -> pd.Series:
        intersection = self.intersection(other)
        weighted_sum = weight_by_len if length_weighting else no_weight

        intersection_sizes = weighted_sum(intersection)
        self_sizes = weighted_sum(self.ngrams_by_document)
        other_size = weighted_sum(other.ngrams)

        result = intersection_sizes / np.minimum(self_sizes, other_size)
        result = result.rename(other.id_)

        return result.sort_values(ascending=False)

    def _init_idf(self) -> None:
        if self.idf_table is not None:
            return

        unique_ngrams = pd.Series(list(self.ngrams))

        df = pd.DataFrame(
            np.vectorize(contains)(
                self.ngrams_by_document, unique_ngrams.values[:, None]
            ),
            index=unique_ngrams,
        )
        N = len(self.documents) + 1
        docs_with_ngram = df.sum(axis=1) + 1

        idf = np.log(N / docs_with_ngram) + 1
        self.idf_table = idf.to_dict()

    def _weight_tf_idf(self, ngrams: set):
        return sum(self.idf_table.get(term, 0) for term in ngrams)

    def _weight_tf_idf_length(self, ngrams: set):
        return sum(self.idf_table.get(term, 0) * len(term) ** 2 for term in ngrams)

    @singledispatchmethod
    def match_tf_idf(self, other):
        raise NotImplementedError(
            f"Cannot match {type(self).__name__} " f"with {type(other).__name__}"
        )

    @match_tf_idf.register(FragmentModel)
    @match_tf_idf.register(ChapterModel)
    def _(self, other: DocumentModel, length_weighting=False) -> pd.Series:
        self._init_idf()
        weight_func = (
            self._weight_tf_idf_length if length_weighting else self._weight_tf_idf
        )
        return self.intersection(other).map(weight_func).sort_values(ascending=False)


@BaseCorpus.intersection.register
def _(self, other: BaseCorpus):
    return pd.DataFrame(
        np.vectorize(set.intersection)(
            self.ngrams_by_document.values[:, None], other.ngrams_by_document
        ),
        index=self.documents.index,
        columns=other.documents.index,
    )


@BaseCorpus.match.register
def _(self, other: BaseCorpus, length_weighting=False) -> pd.DataFrame:
    weighted_sum = weight_by_len if length_weighting else no_weight

    intersection_sizes = weighted_sum(self.intersection(other))
    self_sizes = weighted_sum(self.ngrams_by_document)
    other_sizes = weighted_sum(other.ngrams_by_document)

    return intersection_sizes / np.minimum(
        self_sizes.values[:, None],
        other_sizes,
    )


@BaseCorpus.match_tf_idf.register
def _(self, other: BaseCorpus, length_weighting=False) -> pd.DataFrame:
    self._init_idf()
    weight_func = (
        self._weight_tf_idf_length if length_weighting else self._weight_tf_idf
    )
    return self.intersection(other).map(weight_func)
