from typing import Sequence
import pytest
from ebl_ngrams.chapter_corpus import ChapterCorpus, ChapterRecord
from ebl_ngrams.document_model import DEFAULT_N_VALUES

from ebl_ngrams.fragment_corpus import FragmentCorpus
from tests.test_support import N_VALUES, create_multiline_ngrams


@pytest.fixture
def mock_fragments_data():
    return [
        {"_id": "Mock.1", "signs": "A B C D X\nE F X X\nG X H I"},
        {"_id": "Mock.2", "signs": "G H X\nJ X X\nK L M"},
        {"_id": "Mock.2", "signs": "N O\nP Q"},
    ]


def mock_manuscript(**kwargs):
    data = {
        "provenance": "Nineveh",
        "period": "Old Babylonian",
        "type": "Library",
        "siglumDisambiguator": "",
        "colophon": {"numberOfLines": 0},
        "unplacedLines": {"numberOfLines": 0},
    }
    data.update(kwargs)

    return data


@pytest.fixture
def mock_chapter_data() -> Sequence[ChapterRecord]:
    return [
        {
            "signs": ["A B C D X\nE F X X\nG X H I", "G H X\nJ X X\nK L M", ""],
            "manuscripts": [mock_manuscript(), mock_manuscript(), mock_manuscript()],
            "textId": {"genre": "L", "category": 1, "index": 2},
            "stage": "Old Babylonian",
            "name": "-",
        },
        {
            "signs": ["N O\nP Q"],
            "manuscripts": [mock_manuscript()],
            "textId": {"genre": "L", "category": 1, "index": 2},
            "stage": "Old Babylonian",
            "name": "-",
        },
        {
            "signs": ["A B\nR S T\nU", None],
            "manuscripts": [mock_manuscript(), mock_manuscript()],
            "textId": {"genre": "L", "category": 1, "index": 2},
            "stage": "Old Babylonian",
            "name": "-",
        },
    ]


@pytest.fixture
def mock_fragment_corpus(mock_fragments_data):
    return FragmentCorpus(
        mock_fragments_data,
        DEFAULT_N_VALUES,
        show_progress=False,
        name="MockFragmentarium",
    )


@pytest.fixture
def mock_chapter_corpus(mock_chapter_data):
    return ChapterCorpus(
        mock_chapter_data,
        DEFAULT_N_VALUES,
        show_progress=False,
        name="MockFragmentarium",
    )


@pytest.mark.parametrize("n_values", N_VALUES)
def test_get_ngrams(mock_fragment_corpus, n_values, mock_fragments_data):
    expected = set.union(
        *(
            create_multiline_ngrams(entry["signs"], *n_values)
            for entry in mock_fragments_data
        )
    )
    assert mock_fragment_corpus.get_ngrams(*n_values) == expected