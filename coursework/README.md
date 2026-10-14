# Coursework: Ensemble Selection

Files for the COMP3222 Ensemble Selection coursework. The brief explains what to submit.

| File | Purpose |
| --- | --- |
| [`requirements.txt`](requirements.txt) | The Python package versions your code must run with |
| [`part3_setup.py`](part3_setup.py) | Datasets, classifier pool and settings for Part 3 |

## Setting up

Create a project folder for your submission, then copy `requirements.txt` and `part3_setup.py` into it. From that folder, create a virtual environment with Python 3.11, 3.12 or 3.13 and install the packages.

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

Your folder should end up laid out like this, with `part3_setup.py` next to `solution/`:

```
part3_setup.py
requirements.txt
solution/
tests/
```

## Part 3 data

`part3_setup.load_dataset(name)` downloads each UCR dataset the first time it is loaded and saves it in `data/ucr` next to `part3_setup.py`, so the first run needs an internet connection. To keep the data somewhere else, set the environment variable `COMP3222_UCR_DIR` to that directory.

Do not change `part3_setup.py`, and do not submit it or the data.
