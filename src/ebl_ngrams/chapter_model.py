from typing import TypedDict
import pandas as pd
import requests
from ebl_ngrams.document_model import (
    API_URL,
    DEFAULT_N_VALUES,
    BaseDocument,
    ngrams_multi_n,
    postprocess,
    preprocess,
    validate_n_values,
)
from ebl_ngrams.enums.provenance import Provenance
from ebl_ngrams.enums.stage import Stage
from ebl_ngrams.enums.manuscript_type import ManuscriptType
from ebl_ngrams.enums.period import Period


PROVENANCES = {p.long_name: p.abbreviation for p in Provenance}
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


class ChapterRecord(TypedDict):
    signs: list
    manuscripts: list
    textId: dict
    stage: int
    name: str


class TextId:
    genre: str
    category: int
    index: int

    def __init__(self, data: dict):
        self.genre = data["genre"]
        self.category = int(data["category"])
        self.index = int(data["index"])

    def __str__(self):
        return f"{self.genre}/{self.category}/{self.index}"

    def __repr__(self):
        return repr(str(self))


class ChapterModel(BaseDocument):
    _collection = "chapters"

    def __init__(self, data: ChapterRecord, n_values=DEFAULT_N_VALUES):
        self.text_id = TextId(data["textId"])
        self.stage = Stage.from_name(data["stage"])
        self.name = data["name"].strip()

        super().__init__(self._create_id(data), data["signs"], n_values)

        self._manuscripts = data["manuscripts"]
        self.set_ngrams(*n_values)

    @classmethod
    def load(
        cls, url: str, n_values=DEFAULT_N_VALUES, db="ebldev", uri=None
    ) -> "ChapterModel":

        response = requests.get(cls._create_api_url(url))
        response.raise_for_status()

        return cls(response.json(), n_values)

    @staticmethod
    def _create_api_url(url: str) -> str:
        return "{}{}/{}/{}/chapters/{}/{}/signs".format(
            f"{API_URL}/texts/", *url.split("/")[-5:]
        )

    def _create_id(self, data: dict) -> str:
        return "/".join(
            map(
                str,
                [
                    "",
                    self.genre,
                    self.category,
                    self.index,
                    self.stage.abbreviation,
                    self.name,
                ],
            )
        )

    def set_ngrams(self, *n_values) -> "ChapterModel":
        self.n_values = validate_n_values(n_values) if n_values else self.n_values
        df = (
            pd.DataFrame({"manuscript": self._manuscripts, "signs": self.signs})
            .pipe(set_sigla)
            .pipe(drop_colophon_lines)
        )

        self.ngrams_by_manuscript = (
            df.groupby(level=0)
            .agg("\n".join)
            .map(
                lambda signs: postprocess(
                    ngrams_multi_n(preprocess(signs), *self.n_values)
                )
            )
            .to_dict()
        )

        self.ngrams = (
            set.union(*ngrams.values())
            if (ngrams := self.ngrams_by_manuscript)
            else set()
        )
        return self

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

    @property
    def genre(self) -> str:
        return self.text_id.genre

    @property
    def category(self) -> int:
        return self.text_id.category

    @property
    def index(self) -> int:
        return self.text_id.index
