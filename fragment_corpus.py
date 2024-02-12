from typing import Sequence
from base_corpus import BaseCorpus, fetch_all
from document_model import DEFAULT_N_VALUES
from fragment_model import FragmentModel


class FragmentCorpus(BaseCorpus):
    _collection = "fragments"

    def __init__(
        self,
        data: Sequence[dict],
        n_values=DEFAULT_N_VALUES,
        show_progress=False,
        threading=True,
        name="",
    ):
        super().__init__(data, n_values, show_progress, threading, name)

        load = self._load_threading if threading else self._load
        self.documents = self.fragments = load(data)

        self._vocab = {
            sign for fragment in self for ngram in fragment.ngrams for sign in ngram
        }

    @classmethod
    def load(
        cls,
        n_values=DEFAULT_N_VALUES,
        db="ebldev",
        uri=None,
        show_progress=True,
        threading=True,
        name="",
        **kwargs,
    ) -> "FragmentCorpus":
        query = {"signs": {"$regex": "."}}
        data = fetch_all(
            query,
            projection={"signs": 1},
            collection=cls._collection,
            db=db,
            uri=uri,
            **kwargs,
        )
        if show_progress:
            return cls(
                list(data),
                n_values,
                show_progress,
                name=name,
                threading=threading,
            )

        return cls(data, n_values, name=name, threading=threading)

    def match(self, other, *n_values):
        return [
            (
                fragment.id_,
                fragment.similarity(other, *n_values),
            )
            for fragment in self.fragments
        ]

    def _create_model(self, entry, n_values):
        return FragmentModel(entry["_id"], entry["signs"], n_values=n_values)
