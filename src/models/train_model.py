
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

import sys
sys.path.append("src/models/")
sys.path.append("src/data/")
sys.path.append("src/visualization/")
from model_library import Classifier
from preprocessor import Preprocessor
from visualize import plot_confusion_matrix

#Daten laden
daten = pd.read_pickle("data/processed/Prepared_Data.pickle")
X = daten.loc[:,daten.columns != "result"]
y = daten["result"]

X_train, X_test, y_train, y_test, scaler = Preprocessor(X, y).get_data()

#Models spezifieren
models = {"knn":KNeighborsClassifier, "rf":RandomForestClassifier, "svm":SVC, "dt":DecisionTreeClassifier, "mlp":MLPClassifier}

#Classifier verwenden
clf = Classifier(X_train, X_test, y_train, y_test)
resultat = clf.train_models(models)
resultat_mitEnsemble = clf.ensemble_model()

resultat_mitNN, epoch_errors, epoch_train_accuracy, epoch_test_accuracy, y_pred_nn = clf.neuronal_network(86)

ausgabe = resultat_mitNN

#Bestes Ergebnis bestimmen und als Modell speichern
print("Bestes Model ist: {} mit einer Akkuranz von {}%".format(sorted(ausgabe, key=itemgetter(1), reverse=True)[0][0],sorted(ausgabe, key=itemgetter(1), reverse=True)[0][1]*100))
bestes_model = sorted(ausgabe, key=itemgetter(1), reverse=True)[0][2]
print("Alle Ergebnisse: {}".format(ausgabe))

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
    plt.show()
    plt.plot(epoch_train_accuracy, color="red", label='Train accuracy')
    plt.plot(epoch_test_accuracy, color="blue", label='Test accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy in % / Error rate")
    plt.title("Epoch accuracy / Epoch error rate")
    plt.show()


#print(y_pred_proba)
print()
print('Testdaten')
print(y_test)
print()
auc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print('AUC: %.3f' % auc)

y_pred = bestes_model.predict(X_test) if sorted(ausgabe, key=itemgetter(1), reverse=True)[0][0] != "NN_Model" else y_pred_nn

cnf_matrix = metrics.confusion_matrix(y_test, y_pred,labels=[0, 1, 2])
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'],
                      title='Confusion matrix, without normalization')
plt.show()
