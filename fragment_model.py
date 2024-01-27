import json
import datetime
import pandas as pd
from pymongo import MongoClient
import os
from document_model import (
    DocumentModel,
    DocumentNotFoundError,
    preprocess,
    postprocess,
    linewise_ngrams,
)


def extract_ngrams(signs: pd.Series, n_values=(1, 2, 3)):
    subframes = [
        pd.concat([signs.shift(-i) for i in range(n)], axis=1)
        .dropna()
        .agg(tuple, axis=1)
        for n in n_values
    ]
    ngrams = pd.concat(subframes)

    return ngrams.drop_duplicates()


class FragmentModel(DocumentModel):
    def __init__(self, data: dict, n_values=(1, 2, 3)):
        self.signs = data["signs"]
        self.id_ = data["_id"]

        self.n_values = n_values
        self.retrieved_on = datetime.datetime.now()
        self._set_ngrams()

    @classmethod
    def load_json(cls, path: str, n_values=(1, 2, 3)) -> "FragmentModel":
        with open(path) as jf:
            data = json.load(jf)
        return cls(data, n_values)

    @classmethod
    def load(
        cls, id_: str, n_values=(1, 2, 3), db="ebldev", uri=None
    ) -> "FragmentModel":
        client = MongoClient(uri or os.environ["MONGODB_URI"])
        database = client.get_database(db)
        collection = database.get_collection("fragments")

        if data := collection.find_one(
            {"_id": id_},
            projection={"signs": 1},
        ):
            return cls(data, n_values)
        else:
            raise DocumentNotFoundError(f"No document found for id {id_!r}")

    def _set_ngrams(self):
        self.ngrams = (
            preprocess(self.signs)
            .pipe(linewise_ngrams, n_values=self.n_values)
            .pipe(postprocess)
        )

    def get_ngrams(self, *n_values):
        return (
            {ngram for ngram in self.ngrams if len(ngram) in n_values}
            if n_values
            else self.ngrams
        )

    def __str__(self):
        return "<FragmentNgramModel {} {}>".format(
            self.id_, self.retrieved_on.strftime("%Y-%m-%d")
        )

    def __repr__(self):
        return str(self)
