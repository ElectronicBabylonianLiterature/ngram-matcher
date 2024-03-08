# N-Gram-Based Cuneiform Fragment Matcher

A Python package to load cuneiform sign data from the database of the [electronic Babylonian Library](https://www.ebl.lmu.de/) (eBL) platform
and identify and match pieces that could be

- part of the same tablet,
- belong to a known collection of documents of the same content, or
- are similar for other reasons, e.g., written by the same author

The package extracts character-level (i.e., cueniform sign level) n-grams from the eBL sign data. The n-grams consist of overlapping
subsequences of different lengths that create a type of fingerprint of a document that can be statistically compared. The approach
is unsusceptible with respect to slight variations across documents and at the same time capable of modeling the data
effectively for the task of finding matching pieces in a semi-automatic setting by reducing the number of texts an expert needs to
review from tens of thousands to a few hundred.

## Installation

It is recommended to use a virtual environment. Set it up with a tool of your choice, e.g., the built-in [venv](https://docs.python.org/3/library/venv.html). To install the package run

```python
pip install git+https://github.com/ElectronicBabylonianLiterature/ngram-matcher
```

## Usage

The matcher lends itself for experimentation in Jupyter Notebooks to run a test, inspect the result, and repeat.

To import the classes you need do

```python
from ebl_ngrams import FragmentCorpus, FragmentModel, ChapterCorpus, ChapterModel
```

### Loading Models

By default, n-grams with $n \in [1, 2, 3]$ are extracted. To modify, set the `n_values` parameter,
e.g., `n_values=[4, 5]` when loading models.

```python
# load fragments
fragmentarium = FragmentCorpus.load()
test_fragment = FragmentModel.load("Test.Fragment")

# load chapters
chapter_corpus = ChapterCorpus.load()
test_chapter = ChapterModel.load("/L/1/4/SB/I")
```

When loading fragments, pass either the url or just the **id** (aka museum number; displayed in the
fragment view on eBL or the last part of the url of a fragment).

When loading chapters, pass either the full url, e.g.,
`https://www.ebl.lmu.de/corpus/L/1/4/SB/I`, or just everything after `/corpus` (cf. code snippet).

### Matching

To match things, call the `match` method or one of its variants (see below). All of the `match` and
`intersection` methods take n values as arguments to limit the computation to a subset of the
available n-grams. E.g., to match only 2- and 3-grams call `chapter_corpus.match(test_fragment, 2, 3)`.

```python
>>> result = chapter_corpus.match(test_fragment)
>>> result
/D/2/6/SB/92          0.333333
/L/3/3/SB/-           0.294393
/L/1/2/SB/VII         0.292056
/L/1/4/SB/XI          0.273364
/L/1/2/SB/VI          0.271028
                        ...   
/L/1/9/OB/Susa        0.081776
/L/1/4/OB/Schøyen₁    0.078873
/L/1/4/OB/Nippur      0.070093
/L/1/1/OB/III         0.000000
/L/1/1/OB/II          0.000000
Name: Test.Fragment, Length: 143, dtype: float64
```

The `result` is a pandas series in this case (but can be a scalar or a dataframe/matrix, depending on
what is matched - see below). You can proceed working with the object as is or convert it according
to your needs. E.g., call `result.to_dict()` to obtain a dictionary or `result.to_csv("path/to/my_result.csv")`
to save the result as CSV. Please consult the
[pandas documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html)
to learn more.

When matching collections against single documents, you can get more detailed info on the matches by
setting `include_overlaps=True`.

```python
detailed_result = chapter_corpus.match(test_fragment, include_overlaps=True)
```

This gives you not only the scores but also the number and sets of overlapping n-grams in a single dataframe:

```python
                       score  overlap_size                                            overlap
/D/2/6/SB/92        0.333333             1                                        {(ABZ570,)}
/L/3/3/SB/-         0.294393           126  {(ABZ480, #), (ABZ545, ABZ1), (ABZ115,), (ABZ5...
/L/1/2/SB/VII       0.292056           125  {(ABZ480, #), (ABZ545, ABZ1), (ABZ115,), (ABZ5...
/L/1/4/SB/XI        0.273364           117  {(ABZ480, #), (ABZ545, ABZ1), (ABZ579, ABZ70, ...
/L/1/2/SB/VI        0.271028           116  {(ABZ480, #), (ABZ545, ABZ1), (ABZ115,), (ABZ5...
...                      ...           ...                                                ...
/L/1/9/OB/Susa      0.081776            35  {(ABZ73,), (ABZ231, ABZ384), (ABZ457,), (ABZ58...
/L/1/4/OB/Schøyen₁  0.078873            28  {(ABZ73,), (#, ABZ1), (ABZ99,), (ABZ457,), (AB...
/L/1/4/OB/Nippur    0.070093            30  {(ABZ73,), (ABZ131,), (ABZ586, ABZ131), (ABZ99...
/L/1/1/OB/II        0.000000             0                                                 {}
/L/1/1/OB/III       0.000000             0                                                 {}

[143 rows x 3 columns]
```

Like for the plain scores, it can be processed using pandas or exported to CSV or other formats
with the [pandas api](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html),
e.g., `detailed_result.to_csv("path/to/my_result.csv")`.

You can also get only the intersections manually:

```python
chapter_corpus.intersection(test_fragment)
```

You can compare all possible combinations of corpora (fragmentarium or chapter corpus) and
individual documents (fragments or chapters). The output datatype depends on what you compare.

```python
>>> fragmentarium.match(test_fragment)
K.21453           1.0
K.20224           1.0
1876,1117.2651    1.0
K.7675            1.0
K.21593           1.0
                 ... 
K.17937           0.0
K.23031           0.0
K.20698           0.0
1880,1112.2192    0.0
BM.98841          0.0
Name: Test.Fragment, Length: 26755, dtype: float64

>>> test_chapter.match(test_fragment)
0.2523364485981308

>>> fragmentarium.match(test_chapter)
K.14897    1.0
K.23044    1.0
K.16519    1.0
K.21735    1.0
K.23083    1.0
          ... 
K.22596    0.0
K.22607    0.0
K.18715    0.0
K.22624    0.0
K.23142    0.0
Name: /L/1/4/SB/I, Length: 26755, dtype: float64

>>> chapter_corpus.match(test_fragment)
/D/2/6/SB/92          0.333333
/L/3/3/SB/-           0.294393
/L/1/2/SB/VII         0.292056
/L/1/4/SB/XI          0.273364
/L/1/2/SB/VI          0.271028
                        ...   
/L/1/9/OB/Susa        0.081776
/L/1/4/OB/Schøyen₁    0.078873
/L/1/4/OB/Nippur      0.070093
/L/1/1/OB/III         0.000000
/L/1/1/OB/II          0.000000
Name: Test.Fragment, Length: 143, dtype: float64
```

You can also match entire collections, but be aware that this can be computationally expensive
especially with the TF-IDF matching. Usually you will want to compare only a subset of the entire
data. For more details on filtering see below. E.g.:

```python
>>> subcorpus = fragmentarium.filter(lambda fragment: fragment.id_.startswith("IM"))
>>> subcorpus.match(chapter_corpus)
                 IM.183628  IM.45906  ...  IM.85507  IM.67692
/L/3/12/SB/-      0.311628  0.226415  ...  0.343590  0.155579
/L/0/0/NA/-       0.260465  0.124214  ...  0.251282  0.229846
/L/0/0/NB/-       0.130233  0.135593  ...  0.158974  0.372881
/L/1/2/SB/I       0.306977  0.300314  ...  0.338462  0.177803
/L/1/2/SB/II      0.293023  0.250000  ...  0.338462  0.150788
...                    ...       ...  ...       ...       ...
/Mag/1/1/SB/III   0.334884  0.267296  ...  0.328205  0.187454
/Mag/1/1/SB/IV    0.279070  0.224843  ...  0.292308  0.184366
/Mag/1/1/SB/I     0.302326  0.246855  ...  0.312821  0.162769
/Mag/1/1/SB/II    0.344186  0.272013  ...  0.364103  0.173905
/Mag/1/1/SB/V     0.334884  0.235849  ...  0.317949  0.160677

[143 rows x 837 columns]
```

### Match Strategies

There are a number of matching strategies available. The basic matching is rather naive
but also very fast. TF-IDF is better in theory but very slow or even unfeasible in some settings.

#### 1. Overlap coefficient

- `A.match(B)`
- Computes the overlap coefficient between the sets of n-grams between $A$ and $B$:
  $$\frac{|A \cap B|}{\min(|A|, |B|)}$$

#### 2. Overlap coefficient with length weighting

- `A.match(B, length_weighting=True)`
- Computes the overlap coefficient between the sets of n-grams between $A$ and $B$, but instead
  of the cardinalities of the sets, computes the sum of the squared length of each n-gram.

The rationale behind weighting by length is to give a greater importance to shared n-grams the
longer they are.

#### 3. TF-IDF-based overlap

- `A.match_tf_idf(B)`
- Computes the overlap weighted by TF-IDF
- Rare n-grams receive a higher weight than common ones

#### 4. TF-IDF-based overlap with length weighting

- `A.match_tf_idf(B, length_weighting=True)`
- A combination of TF-IDF and length weighting

### Filtering Options

There are some experimental filtering options to limit the computations to a subset of all
documents in a collection. When loading the fragmentarium or the chapters, `transform` takes
a function that takes the raw json data to be manually preprocessed.

For the **fragmentarium**, the JSON consists of a list of fragment records:

```json
{
   "_id": "Test.Fragment",
   "signs": "ABZ1 ABZ2\nABZ3 X X X"
}
```

E.g., to fetch only the fragments with a museum number starting with `BM`:

```python
bm_fragments = FragmentCorpus.load(
   transform=lambda data: [f for f in data if f["_id"].startswith("BM,")]
   )
```

For the **chapter corpus**, the  (simpified) format is (inspect the `.data` attribute of a
`ChapterCorpus` instance to see all fields):

```json
{
    "signs": ["1st manuscript signs", "2nd manuscript signs"],
    "manuscripts": [
      {"": "<1st manuscript metadata>"},
      {"": "<2nd manuscript metadata>"},
      ],
    "textId": {"category": 3, "index": 12, "genre": "L"},
    "stage": "Standard Babylonian",
    "name": "-",
}
```

E.g., to include only chapters from the magic genre do

```python
magic = ChapterCorpus.load(
   transform=lambda data: c for c in data if c["textId"]["genre"] == "Mag"
   )
```

Alternatively, load the complete data and filter it on the fly using `.filter`.
This one takes a single document instance and includes the ones for which the function
returns `True`. It creates a copy of the original collection. Accordingly, you can

```python
# load the full data
chapters = ChapterCorpus.load()
fragmentarium = FragmentCorpus.load()

# define subcorpora
bm_fragments = fragmentarium.filter(
   lambda fragment: fragment.id_.startswith("BM.")
   )
magic_chapters = chapters.filter(
   lambda chapter: chapter.text_id.genre == "Mag"
   )
```

Note that when matching with TF-IDF, the distribution of signs *depends on the reference corpus*.
So if you use TF-IDF weighting, you should load the full data and `.filter` later.

### Saving Models to Disk

There is no built-in way to serialize objects but you can use pickle. Since the database is
updated constantly, please make sure to *keep a local copy of your model* if you need the ability
to reproduce results. All models have a `retrieved_on` attribute with the full timestamp of
creation.

```python
import pickle

fragments = FragmentCorpus.load()

# save to disk
with open("path/to/my/fragments.pkl", "wb") as f:
   pickle.dump(fragments, f)

# load from disk
with open("path/to/my/fragments.pkl", "rb") as f:
   fragments = pickle.load(f)
```
