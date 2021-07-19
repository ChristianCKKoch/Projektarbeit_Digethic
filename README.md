Projektarbeit_Digethic
==============================

Mein Abschluss-Projekt im Rahmen der Zertifizierung zum Data Scientist bei Dighetic

- cookiecutter-Struktur angelegt
- Datensatz angepasst und erweitert:
    - Views erstellt für die Zusammenführung von Match und Team Daten
    - View erstellt mit der Selektion der Daten für die Aufgabe
    - Daten erweitert mit Transfermarkdaten zur Verwendung des Gesamtmarktwertes und der Spielermarktwerte eines Teams
- Bibliothek mit erstellten Classifier-Modellen sowie Preprocessing aus erstem Projekt angelegt
- train_model.py erstellt für Ablauf des Trainierens

- Neuronales Netz verwenden
    - Ensemble learning

- eingangsdaten-Analyse
- Auswertung des Modells ROC confusion matrix

Todo:
- Optimierungen:
    - Prediction ermöglichen!
    - Random search verwenden für Modell(Hyperparameter)-Optimierung
    - batch size? dropout verwenden? early stopping programmatisch?
- Daten erweitern:
    - Wetterdaten hinzufügen
    - Team-Attribute ausweiten

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
