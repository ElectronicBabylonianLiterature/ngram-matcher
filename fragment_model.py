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

    def __init__(self, id_: str, signs: str, n_values=DEFAULT_N_VALUES):
        super().__init__(id_, signs, n_values)

        self._extract_ngrams()

    @classmethod
    def load(
        cls,
        id_: str,
        n_values=DEFAULT_N_VALUES,
        db="ebldev",
        uri=None,
    ) -> "FragmentModel":
        data = fetch(
            {"_id": id_},
            projection={"signs": 1},
            collection=cls._collection,
            db=db,
            uri=uri,
        )
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
