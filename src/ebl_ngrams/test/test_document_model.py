import pytest
from ebl_ngrams.document_model import DEFAULT_N_VALUES
from ebl_ngrams.document_model import API_URL
from ebl_ngrams.fragment_model import FragmentModel
from ebl_ngrams.test.test_support import (
    N_VALUES,
    create_multiline_ngrams,
    create_ngrams,
    sign_factory,
)


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


@pytest.mark.parametrize("n_values", N_VALUES)
def test_get_ngrams(mock_fragment, n_values):
    assert mock_fragment.get_ngrams(*n_values) == create_ngrams(
        mock_fragment.signs, *n_values
    )


@pytest.mark.parametrize("n_values", N_VALUES)
def test_set_multiline_ngrams(mock_fragment_long, n_values):
    mock_fragment_long.set_ngrams(*n_values)

    assert mock_fragment_long.ngrams == create_multiline_ngrams(
        mock_fragment_long.signs, *n_values
    )


@pytest.mark.parametrize("n_values", N_VALUES)
def test_intersection(n_values):
    left = FragmentModel("Lef.T", "A X B\nC D E F G", n_values)
    right = FragmentModel("Righ.T", "A B E\nF G X", n_values)
    expected = create_multiline_ngrams(left.signs, *n_values) & create_multiline_ngrams(
        right.signs, *n_values
    )
    assert left.intersection(right) == expected
    assert left.intersection(right) == right.intersection(left)


@pytest.mark.parametrize("n_values", N_VALUES)
def test_match(n_values):
    left = FragmentModel("Lef.T", "A B C", n_values)
    right = FragmentModel("Righ.T", "A B D E", n_values)

    shared = len(left.intersection(right))
    max_length = len(create_ngrams(left.signs, *n_values))

    assert left.match(right) == right.match(left)
    assert left.match(right) == pytest.approx(shared / max_length)
