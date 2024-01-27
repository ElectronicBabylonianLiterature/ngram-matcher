# Cuneiform Fragment Matcher

A set of utility classes for comparing documents from the eBL database
(fragments and chapters) in order to find potential matches.
Data can be read from locally stored JSON files or fetched directly
from the db (requires read permissions and a MongoDB uri to be set as environment variable `MONGODB_URI`)

## Usage

```python
from chapter_model import ChapterModel
from fragment_model import FragmentModel


local_chapter = ChapterModel.load_json("path/to/my_chapter.json")

# load from db
chapter = ChapterModel.load("/L/3/12/SB/-")
fragment = FragmentModel.load("Test.Fragment")

>>> chapter.similarity(fragment)
0.24347826086956523

>>> chapter.intersection(fragment)
{('ABZ1',),
 ('ABZ1', 'ABZ1'),
 ('ABZ10',),
 ...}

>>> chapter.intersection(fragment, 2, 3)  # only use 2-, 3-grams
{('ABZ1', 'ABZ1'),
 ('ABZ411', 'ABZ411'),
 ('ABZ461', 'ABZ1'),
 ...}
```
