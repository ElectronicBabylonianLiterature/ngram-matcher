from typing import Iterable
from base_corpus import BaseCorpus, fetch_all, get_total
from document_model import DEFAULT_N_VALUES
from fragment_model import FragmentModel
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def create_fragment_model(entry, n_values):
    return FragmentModel(entry["_id"], entry["signs"], n_values=n_values)


class FragmentCorpus(BaseCorpus):
    _collection = "fragments"

    def __init__(
        self,
        data: Iterable[dict],
        n_values=DEFAULT_N_VALUES,
        show_progress=False,
        total=None,
    ):
        super().__init__(n_values)

        if show_progress and total is None:
            raise ValueError("If 'show_progress' is True, 'total' must be provided")

        with ProcessPoolExecutor() as executor:
            self.fragments = list(
                tqdm(
                    executor.map(
                        partial(create_fragment_model, n_values=n_values), data
                    ),
                    total=total,
                    desc="Building model",
                    disable=not show_progress,
                )
            )
        self._vocab = {
            sign for fragment in self for ngram in fragment for sign in ngram
        }
        self.documents = self.fragments

    @classmethod
    def load(
        cls,
        n_values=DEFAULT_N_VALUES,
        db="ebldev",
        uri=None,
        show_progress=True,
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
            total = get_total(query, cls._collection, db, uri)
            return cls(
                data, n_values, show_progress, min(total, kwargs.get("limit", total))
            )

        return cls(data, n_values)

    def compare(self, other, *n_values):
        return [
            (
                fragment.id_,
                fragment.similarity(other, *n_values),
            )
            for fragment in self.fragments
        ]
