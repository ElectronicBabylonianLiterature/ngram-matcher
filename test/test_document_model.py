from itertools import tee
import numpy as np
import re
from typing import Sequence, Set, Tuple

import pytest
from ebl_ngrams.document_model import DEFAULT_N_VALUES
from ebl_ngrams.document_model import API_URL
from ebl_ngrams.fragment_model import FragmentModel

N_VALUES = [
    [1],
    [1, 2],
    [1, 3],
    [1, 2, 3],
    [2, 3],
    [3],
]


def sign_factory(size, seed=None) -> str:
    if seed is not None:
        np.random.seed(seed)

    def mock_sign():
        if np.random.random() > 0.8:
            return "X"
        return f"ABZ{np.random.randint(1, 200)}"

    return " ".join(mock_sign() for _ in range(size))


def _create_ngrams(signs: Sequence[str], *n) -> Set[Tuple[str]]:
    if len(n) == 1:
        iterables = tee(signs, n[0])

        for i, sub_iterable in enumerate(iterables):
            for _ in range(i):
                next(sub_iterable, None)
        return set(zip(*iterables))
    else:
        return set.union(*(_create_ngrams(signs, n_) for n_ in n))


def create_ngrams(signs: str, *n) -> Set[Tuple[str]]:
    return _create_ngrams([s for s in signs.split() if s != "X"], *n)


def create_multiline_ngrams(signs: str, *n) -> Set[Tuple[str]]:
    return create_ngrams(re.sub(r"\n+", " #\n", signs), *n)


@pytest.fixture
def mock_fragment() -> FragmentModel:
    return FragmentModel("Mock.Fragment", sign_factory(10, seed=42), DEFAULT_N_VALUES)


@pytest.fixture
def mock_fragment_long() -> FragmentModel:
    return FragmentModel(
        "Mock.Fragment",
        "\n".join(sign_factory(10) for _ in range(10)) + "\n\n",
        DEFAULT_N_VALUES,
    )


def test_create_ngrams():
    signs = "A B C X\nX D E F G"
    expected = {
        ("A", "B"),
        ("B", "C"),
        ("C", "#"),
        ("#", "D"),
        ("D", "E"),
        ("E", "F"),
        ("F", "G"),
    }
    assert create_multiline_ngrams(signs, 2) == expected


def test_load(requests_mock, mock_fragment):
    requests_mock.get(
        f"{API_URL}fragments/{mock_fragment.id_}",
        json={"signs": mock_fragment.signs, "_id": mock_fragment.id_},
    )
    fetched_fragment = FragmentModel.load("Mock.Fragment")

    assert fetched_fragment.signs == mock_fragment.signs
    assert fetched_fragment.ngrams == mock_fragment.ngrams


@pytest.mark.parametrize(
    "n_values",
    N_VALUES,
)
def test_get_ngrams(mock_fragment, n_values):
    assert mock_fragment.get_ngrams(*n_values) == create_ngrams(
        mock_fragment.signs, *n_values
    )


@pytest.mark.parametrize(
    "n_values",
    N_VALUES,
)
def test_set_multiline_ngrams(mock_fragment_long, n_values):
    mock_fragment_long.set_ngrams(*n_values)

    assert mock_fragment_long.ngrams == create_multiline_ngrams(
        mock_fragment_long.signs, *n_values
    )


@pytest.mark.parametrize(
    "n_values",
    N_VALUES,
)
def test_intersection(n_values):
    left = FragmentModel("Lef.T", "A X B\nC D E F G", n_values)
    right = FragmentModel("Righ.T", "A B E\nF G X", n_values)
    expected = create_multiline_ngrams(left.signs, *n_values) & create_multiline_ngrams(
        right.signs, *n_values
    )
    assert left.intersection(right) == expected
    assert left.intersection(right) == right.intersection(left)


@pytest.mark.parametrize(
    "n_values",
    N_VALUES,
)
def test_match(n_values):
    left = FragmentModel("Lef.T", "A B C", n_values)
    right = FragmentModel("Righ.T", "A B D E", n_values)

    shared = len(left.intersection(right))
    max_length = len(create_ngrams(left.signs, *n_values))

    assert left.match(right) == right.match(left)
    assert left.match(right) == pytest.approx(shared / max_length)
