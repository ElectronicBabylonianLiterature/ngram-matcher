import pandas as pd
from document_model import (
    DEFAULT_N_VALUES,
    DocumentModel,
    fetch,
    preprocess,
    postprocess,
    linewise_ngrams,
)
from util import overlap_coefficient
from ebl_enums import Provenance, Stage, ManuscriptType, Period

PROVENANCES = {p.long_name: p.abbreviation for p in Provenance}
STAGES = {s.value: s.abbreviation for s in Stage}
REVERSED_STAGES = {v: k for k, v in STAGES.items()}
MANUSCRIPT_TYPES = {m.long_name: m.abbreviation for m in ManuscriptType}
PERIODS = {p.long_name: p.abbreviation for p in Period}


def get_line_labels(row):
    line_types = pd.Series(["TextLine", "ColophonLine", "UnplacedLine"])

    total = len(row.signs)
    colophon = row.manuscript["colophon"]["numberOfLines"]
    unplaced = row.manuscript["unplacedLines"]["numberOfLines"]

    return line_types.repeat(
        [total - colophon - unplaced, colophon, unplaced]
    ).to_list()


def to_siglum(manuscript: dict) -> str:
    parts = {
        "provenance": PROVENANCES,
        "period": PERIODS,
        "type": MANUSCRIPT_TYPES,
        "siglumDisambiguator": {},
    }

    return "".join(
        mapper.get(manuscript[key], manuscript[key]) for key, mapper in parts.items()
    )


def set_sigla(frame):
    frame = frame.dropna(subset="signs")

    return frame.assign(siglum=frame.manuscript.map(to_siglum)).set_index("siglum")


def drop_colophon_lines(frame):
    frame["signs"] = frame.signs.str.split("\n")
    frame["line_type"] = frame.apply(get_line_labels, axis=1)
    frame = frame.drop("manuscript", axis=1)
    frame = frame.explode(["signs", "line_type"])

    return frame.loc[frame.line_type != "ColophonLine", "signs"]


def to_url(data: dict):
    text_id = data["textId"]

    return "/".join(
        map(
            str,
            [
                "",
                text_id["genre"],
                int(text_id["category"]),
                int(text_id["index"]),
                STAGES[data["stage"]],
                data["name"],
            ],
        )
    )


def url_to_query(url: str) -> dict:
    *_, genre, category, index, stage, name = url.split("/")

    return {
        "textId.genre": genre,
        "textId.category": int(category),
        "textId.index": int(index),
        "stage": REVERSED_STAGES[stage],
        "name": name,
    }


class ChapterModel(DocumentModel):
    _collection = "chapters"

    def __init__(self, data: dict, n_values=DEFAULT_N_VALUES):
        super().__init__(data["_id"], data["signs"], n_values)

        self._manuscripts = data["manuscripts"]
        self.url = to_url(data)

        self._set_ngrams()

    @classmethod
    def load(
        cls, url: str, n_values=DEFAULT_N_VALUES, db="ebldev", uri=None
    ) -> "ChapterModel":
        data = fetch(
            url_to_query(url),
            projection={
                "signs": 1,
                "manuscripts": 1,
                "textId": 1,
                "stage": 1,
                "name": 1,
            },
            collection=cls._collection,
            db=db,
            uri=uri,
        )

        return cls(data, n_values)

    def _set_ngrams(self):
        frame = (
            pd.DataFrame({"manuscript": self._manuscripts, "signs": self.signs})
            .pipe(set_sigla)
            .pipe(drop_colophon_lines)
        )

        self.ngrams_by_manuscript = (
            frame.groupby(level=0)
            .agg("\n".join)
            .map(
                lambda signs: postprocess(
                    linewise_ngrams(preprocess(signs), n_values=self.n_values)
                )
            )
            .to_dict()
        )

        self.ngrams = (
            set.union(*ngrams.values())
            if (ngrams := self.ngrams_by_manuscript)
            else set()
        )

    def get_manuscript_ngrams(self, siglum: str, *n_values):
        return (
            {
                ngram
                for ngram in self.ngrams_by_manuscript[siglum]
                if len(ngram) in n_values
            }
            if n_values
            else self.ngrams_by_manuscript[siglum]
        )

    def intersections_per_manuscript(self, other, *n_values):
        result = {}

        for siglum in self.ngrams_by_manuscript:
            A = self.get_manuscript_ngrams(siglum, *n_values)
            B = other.get_ngrams(*n_values)

            result[siglum] = A & B

        return result

    def similarities_per_manuscript(self, other, *n_values):
        result = {}

        for siglum in self.ngrams_by_manuscript:
            A = self.get_manuscript_ngrams(siglum, *n_values)
            B = other.get_ngrams(*n_values)

            result[siglum] = overlap_coefficient(A, B)

        return result
