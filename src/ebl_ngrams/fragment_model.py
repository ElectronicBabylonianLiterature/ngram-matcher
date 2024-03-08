import requests
from ebl_ngrams.document_model import (
    API_URL,
    DEFAULT_N_VALUES,
    BaseDocument,
    ngrams_multi_n,
    preprocess,
    postprocess,
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

        self.set_ngrams()

    @classmethod
    def load(cls, id_: str, n_values=DEFAULT_N_VALUES) -> "FragmentModel":
        id_ = id_.split("/")[-1]
        data = fetch_fragment(id_)
        return cls(id_, data["signs"], n_values)

    def set_ngrams(self, *n_values) -> "FragmentModel":
        self.n_values = validate_n_values(n_values) if n_values else self.n_values
        self.ngrams = postprocess(
            ngrams_multi_n(preprocess(self.signs), *self.n_values)
        )
        return self

    def __len__(self):
        return len(self.ngrams)
