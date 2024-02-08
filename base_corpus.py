import os
import pickle
import datetime
from typing import Literal

from pymongo import MongoClient


def fetch_all(
    query: dict,
    projection: dict,
    collection: Literal["chapters", "fragments"],
    db="ebldev",
    uri=None,
    **kwargs,
):
    client = MongoClient(uri or os.environ["MONGODB_URI"])
    database = client.get_database(db)

    return database.get_collection(collection).find(
        query, projection=projection, **kwargs
    )


def get_total(
    query: dict,
    collection: Literal["chapters", "fragments"],
    db="ebldev",
    uri=None,
):
    client = MongoClient(uri or os.environ["MONGODB_URI"])
    database = client.get_database(db)

    return database.get_collection(collection).count_documents(query)


class BaseCorpus:

    def __init__(self, n_values):
        self.n_values = n_values
        self.is_compressed = False
        self.retrieved_on = datetime.datetime.now()

    def _compress(self):
        if not self.is_compressed:
            encoder = {word: index for index, word in enumerate(self._vocab)}
            for document in self:
                document._compress(encoder)
            self.is_compressed = True

    def _decompress(self):
        if self.is_compressed:
            decoder = dict(enumerate(self._vocab))
            for document in self:
                document._decompress(decoder)
            self.is_compressed = False

    def save(self, path: str):
        self._compress()
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._decompress()

    @classmethod
    def open(cls, path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            raise TypeError(
                f"{cls.__name__} cannot load {model.__class__.__name__} data"
            )

        model._decompress()

        return model

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)
