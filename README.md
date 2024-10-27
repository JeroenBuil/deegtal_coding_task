# dEEGtal_coding_task

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This repo stores the code and findings for the modeling task assigned to me for the job interview of AI Engineer for dEEGtal/Wyss Center.

## How to run:
1. Create python 3.12 environment (this code has been tested with 3.12.7)
    - ```powershell python -m venv .venv
2. Activate environment
    - ```powershell venv\Scripts\activate
3. Install requirements.txt
    - ```powershell pip install -r requirements.txt
4. Run notebook in /notebooks in order!
    - 1_Exploratory_Data_Analysis.ipynb
    - 2_1_Build_CNN_Model.ipynb
    - 3_XAI.ipynb
    - 4_(Experimental)_Build_XGBoost_Model.ipynb


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump. => Contains the dEEGtal coding task data
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         deegtal_coding_task and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── deegtal_coding_task   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes deegtal_coding_task a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dimensionality_reduction.py  <- Package for easy dimensionality tooling. Written by Jeroen Buil
    │
    └── model_architecture.py   <- Code to initialise a simple CNN network
```

--------

