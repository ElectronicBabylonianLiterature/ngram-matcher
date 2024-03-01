from typing import Sequence, TypedDict

from ebl_ngrams.document_model import DEFAULT_N_VALUES
from ebl_ngrams.base_corpus import BaseCorpus
from ebl_ngrams.fragment_model import FragmentModel


class FragmentRecord(TypedDict):
    _id: str
    signs: str


class FragmentCorpus(BaseCorpus):
    _collection = "fragments"
    _api_url = "fragments/all-signs"

    def __init__(
        self,
        data: Sequence[FragmentRecord],
        n_values=DEFAULT_N_VALUES,
        show_progress=False,
        threading=True,
        name="",
    ):

        super().__init__(data, n_values, show_progress, name)

        load = self._load_threading if threading else self._load
        self.documents = load(data)

        self._vocab = {
            sign for fragment in self for ngram in fragment.ngrams for sign in ngram
        }

    @property
    def fragments(self):
        return self.documents

    def _create_model(self, entry, n_values):
        return FragmentModel(entry["_id"], entry["signs"], n_values=n_values)
