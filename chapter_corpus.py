from operator import contains
from typing import Sequence
from base_corpus import BaseCorpus, fetch_all
from chapter_model import ChapterModel
from document_model import DEFAULT_N_VALUES
import pandas as pd
import numpy as np


class ChapterCorpus(BaseCorpus):
    _collection = "chapters"

    def __init__(
        self,
        data: Sequence[dict],
        n_values=DEFAULT_N_VALUES,
        show_progress=False,
        threading=True,
        name="",
    ):
        super().__init__(data, n_values, show_progress, threading, name)

        load = self._load_threading if threading else self._load
        self.documents = self.chapters = load(data)
        self._vocab = {
            sign
            for document in self.documents
            for ngram in document.ngrams
            for sign in ngram
        }
        self._ngrams = None

    @classmethod
    def load(
        cls,
        n_values=DEFAULT_N_VALUES,
        db="ebldev",
        uri=None,
        show_progress=True,
        threading=True,
        name="",
        **kwargs,
    ) -> "ChapterCorpus":
        query = {"signs": {"$regex": "."}, "textId.category": {"$ne": 99}}
        data = fetch_all(
            query,
            projection={
                "signs": 1,
                "manuscripts": 1,
                "textId": 1,
                "stage": 1,
                "name": 1,
            },
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

    def _create_model(self, entry, n_values):
        return ChapterModel(entry, n_values=n_values)

    @property
    def ngrams(self):
        if self._ngrams is None:
            self._ngrams = {
                ngram for chapter in self.chapters for ngram in chapter.get_ngrams()
            }
        return self._ngrams

    @property
    def ngrams_by_chapter(self):
        return self.ngrams_by_document

    def _init_idf(self):
        unique_ngrams = pd.Series(list(self.ngrams))

        df = pd.DataFrame(
            np.vectorize(contains)(
                self.ngrams_by_chapter, unique_ngrams.values[:, None]
            ),
            index=unique_ngrams,
        )
        N = len(self.chapters) + 1
        docs_with_ngram = df.sum(axis=1) + 1

        idf = np.log(N / docs_with_ngram) + 1
        self.idf_table = idf.to_dict()
