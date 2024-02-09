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
        threading=True,
        name="",
    ):
        super().__init__(n_values, name=name)

        if show_progress and total is None:
            raise ValueError("If 'show_progress' is True, 'total' must be provided")

        tqdm_config = {
            "total": total,
            "desc": "Building model",
            "disable": not show_progress,
        }

        self.documents = self.fragments = (
            self._load_threading if threading else self._load
        )(data, tqdm_config)

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
            total = get_total(query, cls._collection, db, uri)
            return cls(
                data,
                n_values,
                show_progress,
                min(total, kwargs.get("limit", total)),
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

    def _load(self, data: dict, tqdm_config: dict):
        for entry in tqdm(data, **tqdm_config):
            yield create_fragment_model(entry, self.n_values)

    def _load_threading(self, data: dict, tqdm_config: dict):
        with ProcessPoolExecutor() as executor:
            return list(
                tqdm(
                    executor.map(
                        partial(create_fragment_model, n_values=self.n_values), data
                    ),
                    **tqdm_config,
                )
            )
