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
    _api_url_ocr = "fragments/all-ocred-signs"

    def __init__(
        self,
        data: Sequence[FragmentRecord],
        n_values=DEFAULT_N_VALUES,
        show_progress=False,
        name="",
        use_ocr=False,
    ):
        self.use_ocr = use_ocr
        super().__init__(data, n_values, show_progress, name, use_ocr)
        self._vocab = {
            sign for fragment in self for ngram in fragment.ngrams for sign in ngram
        }

    @property
    def fragments(self):
        return self.documents

    def _create_model(self, entry, n_values):
        signs_field = "ocredSigns" if self.use_ocr else "signs"
        return FragmentModel(entry["_id"], entry[signs_field], n_values=n_values)
