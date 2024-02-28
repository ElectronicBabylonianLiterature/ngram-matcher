from typing import Sequence, TypedDict
from base_corpus import BaseCorpus
from document_model import DEFAULT_N_VALUES
from fragment_model import FragmentModel


class FragmentRecord(TypedDict):
    _id: str
    signs: str


class FragmentCorpus(BaseCorpus):
    _collection = "fragments"
    _query = {"signs": {"$regex": "."}}
    _projection = {"signs": 1}

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
        self.documents = self.fragments = load(data)

        self._vocab = {
            sign for fragment in self for ngram in fragment.ngrams for sign in ngram
        }

    def _create_model(self, entry, n_values):
        return FragmentModel(entry["_id"], entry["signs"], n_values=n_values)
