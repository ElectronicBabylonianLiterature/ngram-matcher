import pandas as pd
import requests
from ebl_ngrams.document_model import (
    API_URL,
    DEFAULT_N_VALUES,
    BaseDocument,
    preprocess,
    postprocess,
    linewise_ngrams,
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
                Stage.from_name(data["stage"]).abbreviation,
                data["name"].strip(),
            ],
        )
    )


def fetch(url: str):
    response = requests.get(url)
    response.raise_for_status()

    return response.json()


class TextId:
    def __init__(self, data: dict):
        self.genre = data["genre"]
        self.category = int(data["category"])
        self.index = int(data["index"])


class ChapterModel(BaseDocument):
    _collection = "chapters"

    def __init__(self, data: dict, n_values=DEFAULT_N_VALUES):
        super().__init__(to_url(data), data["signs"], n_values)

        self.text_id = TextId(data["textId"])
        self.stage = Stage.from_name(data["stage"])
        self.name = data["name"]
        self._manuscripts = data["manuscripts"]
        self._extract_ngrams()

    @classmethod
    def load(
        cls, url: str, n_values=DEFAULT_N_VALUES, db="ebldev", uri=None
    ) -> "ChapterModel":
        *_, genre, category, index, stage, name = url.split("/")
        fetch_url = "{}{}/{}/{}/chapters/{}/{}/signs".format(
            f"{API_URL}/texts/",
            genre,
            category,
            index,
            stage,
            name,
        )
        data = fetch(fetch_url)

        return cls(data, n_values)

    def _extract_ngrams(self):
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
