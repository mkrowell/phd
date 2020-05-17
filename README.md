PhD
==============================

Dissertation

Project Organization
------------

    ├── LICENSE
    ├── README.md           <- The top-level README for developers using this project
    ├── data
    │   ├── raw             <- The original, immutable data dump
    │   ├── cleaned         <- Intermediate data that has been cleaned
    │   └── processed       <- The final, canonical data sets for analysis
    │
    ├── docs                <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures         <- Generated graphics and figures to be used in reporting
    │   └── logs            <- Generated logs
    │
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py            <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                 <- Source code for use in this project
    │   ├── __init__.py     <- Contains common functions
    │   ├── clean.py        <- Class to turn raw data files into processed data files
    │   ├── download.py     <- Classes to download data files
    │   ├── dataframe.py    <- Classes that perform dataframe operations to clean and process data
    │   ├── database.py     <- Classes to construct PostgreSQL tables in a database
    │
    ├── scripts             <- scripts for use in this project
    │   ├── download_data.py<- Script to download raw data
    │   ├── clean_data.py   <- Script to clean and process data
    │   ├── build_tables.py <- Script to construct database and tables
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
