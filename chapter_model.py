from helper import PROVENANCES, PERIODS, TYPES, STAGES
import os
import json
import datetime
import pandas as pd
from pymongo import MongoClient
from document_model import (
    DocumentModel,
    DocumentNotFoundError,
    preprocess,
    postprocess,
    linewise_ngrams,
)
from util import overlap_coefficient

REVERSED_STAGES = {v: k for k, v in STAGES.items()}


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
        "type": TYPES,
        "siglumDisambiguator": {},
    }

    return "".join(
        mapper.get(manuscript[key], manuscript[key])
        for key, mapper in parts.items()
    )


def set_sigla(frame):
    frame = frame.dropna(subset="signs")

    return frame.assign(siglum=frame.manuscript.map(to_siglum)).set_index(
        "siglum"
    )


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
                text_id["category"],
                text_id["index"],
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
    def __init__(self, data: dict, n_values=(1, 2, 3)):
        self.signs = data["signs"]
        self._manuscripts = data["manuscripts"]
        self.id_ = data["_id"]
        self.url = to_url(data)

        self.n_values = n_values
        self.retrieved_on = datetime.datetime.now()
        self._set_ngrams()

    @classmethod
    def load_json(cls, path: str, n_values=(1, 2, 3)) -> "ChapterModel":
        with open(path) as jf:
            data = json.load(jf)
        return cls(data, n_values)

    @classmethod
    def load(
        cls, url: str, n_values=(1, 2, 3), db="ebldev", uri=None
    ) -> "ChapterModel":
        client = MongoClient(uri or os.environ["MONGODB_URI"])
        database = client.get_database(db)

        collection = database.get_collection("chapters")

        if data := collection.find_one(
            url_to_query(url),
            projection={
                "signs": 1,
                "manuscripts": 1,
                "textId": 1,
                "stage": 1,
                "name": 1,
            },
        ):
            return cls(data, n_values)
        else:
            raise DocumentNotFoundError(f"No document found for url {url!r}")

    def _set_ngrams(self):
        frame = (
            pd.DataFrame(
                {"manuscript": self._manuscripts, "signs": self.signs}
            )
            .pipe(set_sigla)
            .pipe(drop_colophon_lines)
        )

        self.ngrams_by_manuscript = (
            frame.groupby(level=0)
            .agg("\n".join)
            .map(
                lambda signs: postprocess(
                    linewise_ngrams(preprocess(signs), n_values=(1, 2, 3))
                )
            )
            .to_dict()
        )
        self.ngrams = set.union(*self.ngrams_by_manuscript.values())

    def get_ngrams(self, *n_values):
        return (
            {ngram for ngram in self.ngrams if len(ngram) in n_values}
            if n_values
            else self.ngrams
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

    def __str__(self):
        return "<ChapterNgramModel {} {}>".format(
            self.url, self.retrieved_on.strftime("%Y-%m-%d")
        )

    def __repr__(self):
        return str(self)

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
