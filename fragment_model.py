import requests
from document_model import (
    DEFAULT_N_VALUES,
    DocumentModel,
    preprocess,
    postprocess,
    linewise_ngrams,
)

FRAGMENTS_API = "https://www.ebl.lmu.de/api/fragments/"


def fetch_fragment(id_: str):
    response = requests.get(f"{FRAGMENTS_API}{id_}")
    response.raise_for_status()

    data = {"signs": response.json()["signs"], "_id": id_}

    return data


class FragmentModel(DocumentModel):
    _collection = "fragments"

    def __init__(self, id_: str, signs: str, n_values=DEFAULT_N_VALUES):
        super().__init__(id_, signs, n_values)

        self._extract_ngrams()

    @classmethod
    def load(cls, id_: str, n_values=DEFAULT_N_VALUES) -> "FragmentModel":
        data = fetch_fragment(id_)
        return cls(data["_id"], data["signs"], n_values)

    def _extract_ngrams(self):
        self.ngrams = (
            preprocess(self.signs)
            .pipe(linewise_ngrams, n_values=self.n_values)
            .pipe(postprocess)
        )
        return self

    def __len__(self):
        return len(self.ngrams)
