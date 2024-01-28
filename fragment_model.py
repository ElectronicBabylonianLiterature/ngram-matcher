import json
import datetime
from document_model import (
    DEFAULT_N_VALUES,
    DocumentModel,
    fetch,
    preprocess,
    postprocess,
    linewise_ngrams,
)


class FragmentModel(DocumentModel):
    _collection = "fragments"

    def __init__(self, data: dict, n_values=DEFAULT_N_VALUES):
        self.signs = data["signs"]
        self.id_ = self.url = data["_id"]

        self.n_values = n_values
        self.retrieved_on = datetime.datetime.now()
        self._set_ngrams()

    @classmethod
    def load_json(
        cls, path: str, n_values=DEFAULT_N_VALUES
    ) -> "FragmentModel":
        with open(path) as jf:
            data = json.load(jf)
        return cls(data, n_values)

    @classmethod
    def load(
        cls, id_: str, n_values=DEFAULT_N_VALUES, db="ebldev", uri=None
    ) -> "FragmentModel":
        data = fetch(
            {"_id": id_},
            projection={"signs": 1},
            collection=cls._collection,
            db=db,
            uri=uri,
        )

        return cls(data, n_values)

    def _set_ngrams(self):
        self.ngrams = (
            preprocess(self.signs)
            .pipe(linewise_ngrams, n_values=self.n_values)
            .pipe(postprocess)
        )
