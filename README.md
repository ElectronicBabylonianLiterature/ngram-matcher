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

## Setup

In order to access the db you need a MongoDB connection string, either stored as
environment variable `MONGODB_URI` or passed to the constructors as `uri="..."` keyword
argument.

It is recommended to install the required packages listed in requirements.txt in a virtual
environment. Here is how to do it with [Pipenv](https://pipenv.pypa.io/) which conveniently
reads environment variables automatically when provided with a .env file.

1. Create a new folder, e.g., `mkdir cuneiform_matching && cd cuneiform_matching`
2. Download this repo:

   ```sh
   git clone https://github.com/ElectronicBabylonianLiterature/ngram-matcher.git
   ```

3. Create a Python 3 environment by running `pipenv --three` and run `pipenv install -r ngram-matcher/requirements.txt` to install the requirements.
4. Create a new file named `.env` and insert the following variables, filling in your information:

   ```sh
   MONGODB_URI="mongodb://..."
   PYTHONPATH="/Path/to/ebl-api:/Path/to/ngram-matcher:$PYTHONPATH"
   ```

5. Run `pipenv shell` to activate the environment. You can now import and use the n-gram model classes (e.g., `from fragment_model import FragmentModel`) in Python scripts, IPython, etc. from within the shell. Alternatively, you can launch your favorite code editor, e.g., `pipenv run code` for VSCode and proceed as usual.
