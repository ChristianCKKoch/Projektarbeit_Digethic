from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle as pi

import sys
sys.path.append("src/models/")
sys.path.append("src/data/")
sys.path.append("src/visualization/")
from model_library import Classifier
from preprocessor import Preprocessor
from visualize import plot_confusion_matrix

#Daten laden aus im prepare-Schritt erstellten pickle-Datei
daten = pd.read_pickle("data/processed/Prepared_Data.pickle")

#Aufsplitten in Feature-Datensatz (X) und Zielvariablen-Datensatz (y)
X = daten.loc[:,daten.columns != "result"]
y = daten["result"]

#Weiteres Aufsplitten in normalisierte Train- und Testdatensätze sowie Rückgabe des verwendeten Skalierers
X_train, X_test, y_train, y_test, scaler = Preprocessor(X, y).get_data()

#Models spezifieren, die verwendete werden sollen
models = {"rf":RandomForestClassifier, "knn":KNeighborsClassifier, "rf":RandomForestClassifier, "svm":SVC, "dt":DecisionTreeClassifier, "mlp":MLPClassifier}

#Classifier initialisieren mit den Trainings- und Testdaten
clf = Classifier(X_train, X_test, y_train, y_test)

#Klassische Methoden trainieren und Ergebnis-Array empfangen
resultat = clf.train_models(models)
#Ebenfalls ensemble learning durchführen und das Resultat dem Ergebnis-Array hinzufügen
resultat_mitEnsemble = clf.ensemble_model()

#Ebenfalls Neuronales Netz trainieren und durchführen und das Resultat dem Ergebnis-Array hinzufügen
#Das Neuronale Netz wird mit maximal 1000 Epochen trainiert.
#Die patience" des Early Stopping wird auf 2 sowie der Akkuranz-Threshold auf 50 gesetzt.
#Hierbei werden auch die arrays zur Fehler bzw. Akkuranz-Ausgabe empfangen sowie das mittels y_test berechneten Prädiktion
resultat_mitNN, epoch_errors, epoch_train_accuracy, epoch_test_accuracy, y_pred_nn = clf.neuronal_network(1000, 2, 50)

#Zwischenvariable für die übergabe des Ergebnis-Arrays
ausgabe = resultat_mitNN

#Bestes Ergebnis bestimmen und als Modell speichern
print("Bestes Model ist: {} mit einer Akkuranz von {}%".format(sorted(ausgabe, key=itemgetter(1), reverse=True)[0][0],sorted(ausgabe, key=itemgetter(1), reverse=True)[0][1]*100))
bestes_model = sorted(ausgabe, key=itemgetter(1), reverse=True)[0][2]
print("Alle Ergebnisse: ") #{}".format(ausgabe))
df_ergebnisse = pd.DataFrame(ausgabe)
df_ergebnisse.to_excel("data/interim/Ergebnis_Akkuranzen.xlsx", header=['Name des Modells','Akkuranz','Modellobjekt'])

#Wenn bestes Modell NN, dann predict probability vorbereiten
if sorted(ausgabe, key=itemgetter(1), reverse=True)[0][0] != "NN_Model":
    y_pred_proba = bestes_model.predict_proba(X_test)
else:
    with torch.no_grad():
        print()
        print('Ergebnis-Tensor des neuronalen Netzwerks auf Basis der Testdaten')
        print(bestes_model(torch.from_numpy(X_test).float()))
        m = torch.nn.Softmax(dim=1)
        tor = bestes_model(torch.from_numpy(X_test).float())
        y_pred_proba = m(tor)

        plt.plot(epoch_errors, color="green", label='CrossEntropyLoss')
        plt.xlabel("Epochs")
        plt.ylabel("CrossEntropyLoss")
        plt.title("Epoch errors")
        plt.legend()
        plt.show()

        plt.plot(epoch_train_accuracy, color="red", label='Train accuracy')
        plt.plot(epoch_test_accuracy, color="blue", label='Test accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy in %")
        plt.title("Epoch accuracy")
        plt.legend()
        plt.show()

#ROC-AUC score ermitteln
roc_auc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print('ROC-AUC Score')
print('ROC-AUC: %.3f' % roc_auc)

#Ermittelten Testwerte zurückgeben für Berechnung der Confusion Matrix
y_pred = bestes_model.predict(X_test) if sorted(ausgabe, key=itemgetter(1), reverse=True)[0][0] != "NN_Model" else y_pred_nn

#Confusion Matrix berechnen
cnf_matrix = metrics.confusion_matrix(y_test, y_pred,labels=[0, 1, 2])
#Plot nicht-normalisierte confusion matrix
plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'],
                      title='Confusion matrix, without normalization')
plt.show()

# Model in der Datei speichern
clf_file = "models/classifier_object.pickle"
f = open(clf_file, 'wb')
pi.dump(bestes_model, f)
f.close()

#Scaler auch in Datei speichern
sca_file = "models/scaler_object.pickle"
f = open(sca_file, 'wb')
pi.dump(scaler, f)
f.close()