
from operator import itemgetter, attrgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd

import sys
sys.path.append("src/models/")
sys.path.append("src/data/")
from model_library import Classifier
from preprocessor import Preprocessor

#Daten laden
daten = pd.read_pickle("data/processed/Prepared_Data.pickle")
X = daten.loc[:,daten.columns != "result"]
y = daten["result"]

X_train, X_test, y_train, y_test, scaler = Preprocessor(X, y).get_data()

#Models spezifieren
models = {"rf":RandomForestClassifier, "dt":DecisionTreeClassifier, "knn":KNeighborsClassifier, "svm":SVC, "mlp":MLPClassifier}

#Classifier verwenden
clf = Classifier(X_train, X_test, y_train, y_test)
resultat = clf.train_models(models)

#Bestes Ergebnis bestimmen und als Modell speichern
print("Bestes Model ist: {} mit einer Akkuranz von {}%".format(sorted(resultat, key=itemgetter(1), reverse=True)[0][0],sorted(resultat, key=itemgetter(1), reverse=True)[0][1]*100))
bestes_model = sorted(resultat, key=itemgetter(1), reverse=True)[0][2]
print("Alle Ergebnisse: {}".format(resultat))