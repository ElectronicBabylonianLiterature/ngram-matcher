import requests
from ebl_ngrams.document_model import (
    API_URL,
    DEFAULT_N_VALUES,
    BaseDocument,
    preprocess,
    postprocess,
    linewise_ngrams,
    validate_n_values,
)


def fetch_fragment(id_: str):
    response = requests.get(f"{API_URL}fragments/{id_}")
    response.raise_for_status()

    data = {"signs": response.json()["signs"]}

    return data


class FragmentModel(BaseDocument):
    _collection = "fragments"

    def __init__(self, id_: str, signs: str, n_values=DEFAULT_N_VALUES):
        super().__init__(id_, signs, n_values)

        self.set_ngrams(*n_values)

    @classmethod
    def load(cls, id_: str, n_values=DEFAULT_N_VALUES) -> "FragmentModel":
        data = fetch_fragment(id_)
        return cls(id_, data["signs"], n_values)

    def set_ngrams(self, *n_values) -> "FragmentModel":
        validate_n_values(n_values)
        self.n_values = n_values
        self.ngrams = (
            preprocess(self.signs)
            .pipe(linewise_ngrams, n_values=self.n_values)
            .pipe(postprocess)
        )
        return self

    def __len__(self):
        return len(self.ngrams)
