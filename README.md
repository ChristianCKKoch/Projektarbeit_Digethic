Projektarbeit_Digethic
==============================

Mein Abschluss-Projekt im Rahmen der Zertifizierung zum Data Scientist bei Dighetic
Autor: Christian Koch
Im Juli 2021

Das PDF zur Projektarbeit ist im Ordner \docs zu finden "Projektarbeit-Data-Scientist-ChristianKoch-1.0.pdf"

Kurze Anleitung:
1) Projekt aus Github laden und benötigte Installation durchführen (pip install -r requirements.txt), empfohlen in der lokalen virtuellen Environment (python -m venv .venv)
2) Vorbereiten der Daten ausführen (python .\src\data\prepare.py)
--> Schritt 2 ist nur nötig, wenn die Daten erneut aus der Datenbank gelesen werden sollen! Github erlaubt nur Dateien < 100MB. Darum ist die Datenbank selber nicht im Umfang von Github verfügbar.
3) Modelle trainieren (python .\src\models\train_model.py)
4) Prädiktion durchführen (python .\src\models\predict_result.py), evtl. vorher noch Vorhersage-File /data/raw/Bundesliga_Vorhersage.xlsx anpassen

Zusammenfasung der erbrachten Programmierleistung:
- cookiecutter-Struktur angelegt
- Datensatz angepasst und erweitert:
    - Views erstellt für die Zusammenführung von Match und Team Daten
    - View erstellt mit der Selektion der Daten für die Aufgabe
    - Daten erweitert mit Transfermarkdaten zur Verwendung des Gesamtmarktwertes und der Spielermarktwerte eines Teams
- prepare.py erstellt mit Prozeduren zum One-hot-encoden und Vorbereiten sowie Analysieren der Eingangsdaten
- visualize.py erstellt
- Bibliothek model_library.py mit erstellten Classifier-Modellen sowie Preprocessing aus erstem Projekt angelegt
- train_model.py erstellt für Ablauf des Trainierens
- Bibliothek erweitert mit Voting Classifier (Ensemble Learning) und Neuronalem Netz
- Early Stopping in Neuronales Netz eingebaut; hierzu early_stopping.py erstellt basierend auf https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
- Auswertung des Modells mittels ROC-AUC und confusion matrix
- Prädiktion ermöglicht (predict_result.py), noch sehr simpel, mittels xl-sheet und einfacher Ausgabe (print)

Todo's:
- Optimierungen:
    - Prädiktionsanwendung ausweiten und benutzerfreundlicher sowie flexibler gestalten
    - Random-search verwenden für Modell(Hyperparameter)-Optimierung
    - Programmierung noch abstrakter und schlanker gestalten
- Daten erweitern:
    - Wetterdaten hinzufügen
    - Team-Attribute ausweiten
    - Aufstellungen und Spielerstärken integrieren

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
