from typing import Sequence
from base_corpus import BaseCorpus, fetch_all
from chapter_model import ChapterModel
from document_model import DEFAULT_N_VALUES


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
        query = {"signs": {"$regex": "."}}
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
        if show_progress:
            return cls(
                list(data),
                n_values,
                show_progress,
                name=name,
                threading=threading,
            )

        return cls(data, n_values, name=name, threading=threading)

    def _create_model(self, entry, n_values):
        return ChapterModel(entry, n_values=n_values)

    def _ngrams(self):
        return {ngram for chapter in self.chapters for ngram in chapter.get_ngrams()}
