# Coursework: Ensemble Selection

Files for the COMP3222 Ensemble Selection coursework. The brief explains what to submit.

| File | Purpose |
| --- | --- |
| [`requirements.txt`](requirements.txt) | The Python package versions your code must run with |
| [`part3_setup.py`](part3_setup.py) | Datasets, classifier pool and settings for Part 3 |
| [`data/ucr`](data/ucr) | The eight UCR datasets used in Part 3 |

## Setting up

Create a project folder for your submission, then copy `requirements.txt`, `part3_setup.py` and the `data` folder into it. From that folder, create a virtual environment with Python 3.11, 3.12 or 3.13 and install the packages.

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Your folder should end up laid out like this, with `part3_setup.py` and `data/` next to `solution/`:

```
data/
    ucr/
        Computers/
        ...
part3_setup.py
requirements.txt
solution/
tests/
```

## Part 3 data

`part3_setup.load_dataset(name)` reads the datasets from `data/ucr` next to `part3_setup.py`. If you would rather not copy the data, set the environment variable `COMP3222_UCR_DIR` to the `coursework/data/ucr` folder of your clone of this repository instead.

If a dataset is missing from that folder, `load_dataset` downloads it from the UCR archive and saves it there, which needs an internet connection. Use the copies in this repository where you can: they are the data your submission will be marked with.

Do not change `part3_setup.py` or the data, and do not submit either of them.
