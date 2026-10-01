# COMP3222 Machine Learning Technologies — 2026 labs

This repository contains lab sheets, starter code, notebooks, and datasets for COMP3222.

| Start here | Material |
| --- | --- |
| [Lab 1 guide](code/lab1/README.md) | [Lab Sheet 1](Lab%20Sheet%201.pdf), [starter example](code/lab1/lab_1.py), [notebook](notebooks/lab1.ipynb), and solutions in `code/lab1` |
| [Lab 2 guide](code/lab2/README.md) | [Lab Sheet 2](Lab%20Sheet%202.pdf), starter code in `code/lab2`, and data in `data/lab2` |
| [Jupyter introduction](notebooks/labW1_Intro_to_Jupyter.ipynb) | Optional NumPy and notebook practice |

Later lab code is in `code/lab3` and `code/lab4`, with its data in the corresponding `data` directories. The lab sheet PDFs are at the repository root.

## Getting started

Run commands from the repository root unless a lab guide says otherwise. Create and activate a virtual environment, then install the core packages:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scikit-learn numba matplotlib jupyter
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn numba matplotlib jupyter
```

Use the guides above for each lab's tasks and run commands.
