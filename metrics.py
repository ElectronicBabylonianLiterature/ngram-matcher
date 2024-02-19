from functools import singledispatch
import pandas as pd


@singledispatch
def weight_by_len(ngrams):
    raise NotImplementedError(
        f"Can only weight Series, DataFrame or set, got {type(ngrams)} instead"
    )


@weight_by_len.register
def _(ngrams: set) -> int:
    return sum(len(ngram) ** 2 for ngram in ngrams)


@weight_by_len.register(pd.Series)
@weight_by_len.register(pd.DataFrame)
def _(ngrams) -> int:
    return ngrams.map(weight_by_len)


@singledispatch
def no_weight(ngrams):
    raise NotImplementedError(
        f"Can only weight Series, DataFrame or set, got {type(ngrams)} instead"
    )


@no_weight.register
def _(ngrams: set) -> int:
    return len(ngrams)


@no_weight.register(pd.Series)
@no_weight.register(pd.DataFrame)
def _(ngrams) -> int:
    return ngrams.map(len)
