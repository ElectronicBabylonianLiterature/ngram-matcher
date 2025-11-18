import pytest
import numpy as np
from ebl_ngrams import DEFAULT_N_VALUES, ChapterCorpus, FragmentCorpus, ChapterModel
from ebl_ngrams.document_model import API_URL

from test_support import N_VALUES, create_multiline_ngrams


@pytest.fixture
def mock_fragments_data():
    return [
        {"_id": "Mock.1", "signs": "A B C D X\nE F X X\nG X H I"},
        {"_id": "Mock.2", "signs": "G H X\nJ X X\nK L M"},
        {"_id": "Mock.2", "signs": "N O\nP Q"},
    ]


@pytest.fixture
def mock_fragments_data_ocr():
    return [
        {"_id": "Mock.1", "ocredSigns": "A B C D X\nE F X X\nG X H I"},
        {"_id": "Mock.2", "ocredSigns": "G H X\nJ X X\nK L M"},
        {"_id": "Mock.2", "ocredSigns": "N O\nP Q"},
    ]


def mock_manuscript_factory(**kwargs):
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


MOCK_CHAPTER_DATA = [
    {
        "signs": ["A B C D X\nE F X X\nG X H I", "G H X\nJ X X\nK L M", ""],
        "manuscripts": [mock_manuscript_factory() for _ in range(3)],
        "textId": {"genre": "L", "category": 1, "index": 2},
        "stage": "Old Babylonian",
        "name": "-",
    },
    {
        "signs": ["N O\nP Q"],
        "manuscripts": [mock_manuscript_factory()],
        "textId": {"genre": "L", "category": 1, "index": 2},
        "stage": "Old Babylonian",
        "name": "-",
    },
    {
        "signs": ["A B\nR S T\nU", None],
        "manuscripts": [mock_manuscript_factory() for _ in range(2)],
        "textId": {"genre": "L", "category": 1, "index": 2},
        "stage": "Old Babylonian",
        "name": "-",
    },
]

MOCK_CHAPTER_CORPUS = ChapterCorpus(
    MOCK_CHAPTER_DATA,
    DEFAULT_N_VALUES,
    show_progress=False,
)


@pytest.fixture
def mock_chapter() -> ChapterModel:
    data = MOCK_CHAPTER_DATA[0]
    partial_signs = ["\n".join(signs.split("\n")[:2]) for signs in data["signs"]]
    return ChapterModel({**data, "signs": partial_signs}, DEFAULT_N_VALUES)


@pytest.fixture
def mock_fragment_corpus(mock_fragments_data):
    return FragmentCorpus(
        mock_fragments_data,
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


@pytest.mark.parametrize("n_values", N_VALUES)
def test_intersect_document(mock_chapter, n_values):
    intersections = (
        MOCK_CHAPTER_CORPUS.get_ngrams_by_document(*n_values)
        .map(mock_chapter.get_ngrams(*n_values).intersection)
        .to_list()
    )
    expected = [{x for x in s if len(x) in n_values} for s in intersections]

    assert (
        MOCK_CHAPTER_CORPUS.intersection(mock_chapter, *n_values).to_list() == expected
    )


@pytest.mark.parametrize("n_values", N_VALUES)
@pytest.mark.parametrize("other_chapter", [*MOCK_CHAPTER_CORPUS.documents])
def test_match_document(other_chapter, n_values):
    intersections = MOCK_CHAPTER_CORPUS.intersection(other_chapter, *n_values)
    mock_corpus_sizes = MOCK_CHAPTER_CORPUS.get_ngrams_by_document(*n_values).str.len()
    expected = intersections.str.len() / np.minimum(
        mock_corpus_sizes,
        len(other_chapter.get_ngrams(*n_values)),
    )
    assert MOCK_CHAPTER_CORPUS.match(
        other_chapter, *n_values
    ).to_list() == pytest.approx(sorted(expected.to_list(), reverse=True))


def test_load_with_ocr(requests_mock, mock_fragments_data_ocr):
    requests_mock.get(
        f"{API_URL}fragments/all-ocred-signs",
        json=mock_fragments_data_ocr,
    )

    corpus = FragmentCorpus.load(use_ocr=True, show_progress=False)

    assert len(corpus) == len(mock_fragments_data_ocr)
    assert requests_mock.called
    assert requests_mock.last_request.url == f"{API_URL}fragments/all-ocred-signs"


def test_load_without_ocr(requests_mock, mock_fragments_data):
    requests_mock.get(
        f"{API_URL}fragments/all-signs",
        json=mock_fragments_data,
    )

    corpus = FragmentCorpus.load(use_ocr=False, show_progress=False)

    assert len(corpus) == len(mock_fragments_data)
    assert requests_mock.called
    assert requests_mock.last_request.url == f"{API_URL}fragments/all-signs"


def test_load_default_no_ocr(requests_mock, mock_fragments_data):
    requests_mock.get(
        f"{API_URL}fragments/all-signs",
        json=mock_fragments_data,
    )

    corpus = FragmentCorpus.load(show_progress=False)

    assert requests_mock.last_request.url == f"{API_URL}fragments/all-signs"
