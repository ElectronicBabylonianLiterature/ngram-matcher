from itertools import tee
import numpy as np
import re
from typing import Sequence, Set, Tuple


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
