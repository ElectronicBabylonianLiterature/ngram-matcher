from typing import Sequence, TypedDict

from base_corpus import BaseCorpus
from chapter_model import ChapterModel
from document_model import DEFAULT_N_VALUES


class ChapterRecord(TypedDict):
    signs: list
    manuscripts: list
    textId: dict
    stage: int
    name: str


class ChapterCorpus(BaseCorpus):
    _collection = "chapters"
    _api_url = "corpus/all-signs"

    def __init__(
        self,
        data: Sequence[ChapterRecord],
        n_values=DEFAULT_N_VALUES,
        show_progress=False,
        threading=True,
        name="",
    ):
        super().__init__(data, n_values, show_progress, name)

        load = self._load_threading if threading else self._load
        self.documents = self.chapters = load(data)
        self._vocab = {
            sign
            for document in self.documents
            for ngram in document.ngrams
            for sign in ngram
        }

    def _create_model(self, entry, n_values):
        return ChapterModel(entry, n_values=n_values)
