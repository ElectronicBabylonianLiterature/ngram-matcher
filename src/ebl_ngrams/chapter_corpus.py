from typing import Sequence

from ebl_ngrams.base_corpus import BaseCorpus
from ebl_ngrams.chapter_model import ChapterModel, ChapterRecord
from ebl_ngrams.document_model import DEFAULT_N_VALUES


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
        self.documents = load(data)
        self._vocab = {
            sign
            for document in self.documents
            for ngram in document.ngrams
            for sign in ngram
        }

    @property
    def chapters(self):
        return self.documents

    def _create_model(self, entry, n_values):
        return ChapterModel(entry, n_values=n_values)
