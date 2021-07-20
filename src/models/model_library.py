from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import pickle as pi
from operator import itemgetter

import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append("src/models/")
from early_stopping import EarlyStopping

class Classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        #Array für alle Ergebnisse
        self.ergebnis = []
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def train_models(self, models):
        for model in models:
            #-----------------------
            #Knn-Classifier
            #-----------------------
            if model == 'knn':
                #Optimalen Knn-Classifier bestimmen
                error = []
                for i in range(1, 250):
                    knn = KNeighborsClassifier(n_neighbors=i)
                    knn.fit(self.X_train, self.y_train)
                    pred_i = knn.predict(self.X_test)
                    error.append([i, np.mean(pred_i != self.y_test)])

                #Debug-Print
                print()
                print("Debug KNN-Classifier")
                print("knn n: {}".format(sorted(error, key=itemgetter(1), reverse=False)[0][0]))
                print("knn error: {}".format(sorted(error, key=itemgetter(1), reverse=False)[0][1]))
                print()

                #Optimale Anzahl der n_neighbors übergeben
                optimal_n = sorted(error, key=itemgetter(1), reverse=False)[0][0]

                #Knn-Classifier trainieren
                knnclf = KNeighborsClassifier(n_neighbors=optimal_n)
                knnclf.fit(self.X_train, self.y_train)

                #Knn-Classifier Akkuranz bestimmen
                score = knnclf.score(self.X_test,self.y_test)
                self.ergebnis.append([knnclf.__class__.__name__, score, knnclf])
            #-----------------------
                
            #-----------------------
            #Decision Tree
            #-----------------------
            elif model == 'dt':
                #class_weight gebrauchen für DT und RF

                #Optimalen Decision Tree bestimmen
                #Zu testende Decision Tree Parameter
                dt = DecisionTreeClassifier()
                tree_para = {'criterion':['gini','entropy'],'max_depth':[i for i in range(1,20)]
                    , 'min_samples_split':[i for i in range (2,10)]}

                #GridSearchCV 
                grd_clf = GridSearchCV(dt, tree_para, cv=5)
                grd_clf.fit(self.X_train, self.y_train)

                #Besten gefundenen Decision Tree übergeben
                dt_clf = grd_clf.best_estimator_

                #Debug-Print
                print()
                print("Debug DecisionTreeClassifier")
                print("dt best parameters: {}".format(grd_clf.best_params_))
                print()

                score = dt_clf.score(self.X_test, self.y_test)
                self.ergebnis.append([dt_clf.__class__.__name__, score, dt_clf])
            #-----------------------

            #-----------------------
            #Random Forest
            #-----------------------
            elif model == 'rf':
                #rf = RandomForestClassifier(max_depth=8, criterion="entropy", min_samples_split=9)
                rf = RandomForestClassifier(n_estimators=100)
                rf.fit(self.X_train, self.y_train)
                score = rf.score(self.X_test, self.y_test)
                self.ergebnis.append([rf.__class__.__name__, score, rf])
            #-----------------------

            #-----------------------
            #Support Vector Machine
            #-----------------------
            elif model == 'svm':
                svm = SVC(kernel = 'poly', probability=True)
                svm.fit(self.X_train, self.y_train)
                score = svm.score(self.X_test, self.y_test)
                self.ergebnis.append([svm.__class__.__name__, score, svm])

            #-----------------------
            #MLP
            #-----------------------
            elif model == 'mlp':
                mlp = MLPClassifier(hidden_layer_sizes=[100,100], max_iter=5000, solver='sgd'
                , learning_rate='adaptive', learning_rate_init=0.01, n_iter_no_change=200, early_stopping=True)
                mlp.fit(self.X_train, self.y_train)
                score = mlp.score(self.X_test, self.y_test)
                self.ergebnis.append([mlp.__class__.__name__, score, mlp])

                #Debug-Print
                print()
                print("Debug MLPClassifier")
                print("iterations: {}; layers: {}; loss: {}".format(mlp.n_iter_, mlp.n_layers_, mlp.loss_))
                print()
                #epochs = np.linspace(1,mlp.n_iter_, mlp.n_iter_)

        return self.ergebnis
    
    def ensemble_model(self):
        
        #Alle inkludierten Modelle werden in eine Liste geladen, die dann als Parameter
        #dem Voting Classifier übergeben wird.
        models = list()
        for model in self.ergebnis:
            models.append([model[0], model[2]])
        
        voting_clf = VotingClassifier(estimators=models, voting='soft')
        voting_clf.fit(self.X_train, self.y_train)
        score = voting_clf.score(self.X_test, self.y_test)
        self.ergebnis.append([voting_clf.__class__.__name__, score, voting_clf])

        return self.ergebnis

    def neuronal_network(self, epochs, patience_early_stopping, threshold_for_early_stopping):
        #Funktion für das Ansprechen und Ausführen des Neuronalen Netzes mittels Pytorch

        #Standardausgabe für Pytorch, auf welcher processing unit gerechnet wird
        #In meinem Falle nur CPU möglich
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('This Computation is running on {}'.format(device))

        #Initialisierung des Neuronalen Netzwerks
        nn_model = NN_Model()
        #Als Fehlerfunktion wird CrossEntropyLoss verwendet
        loss_func = torch.nn.CrossEntropyLoss()
        #Als Optimizer der Adam-Optimizer mit einer learing rate von 0.001
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.0001)

        #Leere Arrays für das Speichern von Fehler-/Akkuranzdaten über die Epochen hinweg
        epoch_errors = []
        epoch_train_accuracy = []
        epoch_test_accuracy = []

        #Initialisierung des Early Stopping
        early_stopping = EarlyStopping(patience=patience_early_stopping)

        #Umsetzen der Trainings- und Testdaten in die benötigten Tensoren-Formate
        X_Train = torch.from_numpy(self.X_train).float()
        y_Train = torch.tensor(self.y_train, dtype=torch.long)
        X_Test = torch.from_numpy(self.X_test).float()
        y_Test = torch.from_numpy(np.array(self.y_test)).long()
        
        #Trainieren des Neuronalen Netzwerks; maximale Anzahl der Epochen als Funktionsparameter übergeben
        for epoch in range(epochs):
            
            #Vorbereiten der Ergebnisse des Neuronalen Netzwerkes
            #LogSoftmax explizit hier, da diese in der Fehlerfunktion (CrossEntropyLoss) automatisch
            #angewandt wird!
            log_sm = torch.nn.LogSoftmax(dim=1)
            train_nn_model = log_sm(nn_model(X_Train))
            test_nn_model = log_sm(nn_model(X_Test))

            #Erstellen von leerem Array für das Speichern der einzelnen vom Modell berechneten Ergebnisse
            #Zusätzlich noch ein Zähler zum Aufsummieren der korrekt vorhergesagten Ergebnisse, mit 0 initialisiert
            train_pred_ergebnis = []
            train_running_correct = 0

            test_pred_ergebnis = []
            test_running_correct = 0

            #Autograd ausschalten für das Berechnen der Ergebnisse zu Validierungszwecken
            with torch.no_grad():
            #Trainings-Akkuranz
            # Leeren array füllen mit Ergebnissen aus Ergebnis-Tensor
            # Hierbei werden die probalistischen Werte verglichen und das wahrscheinlichste Ergebnis übergeben
            # als 0 - Heimsieg, 1 - Unentschieden, 2 - Auswärtssieg
                for i in range(train_nn_model.shape[0]):
                    ergebnis = 0 if (train_nn_model[i][0] > train_nn_model[i][1] and train_nn_model[i][0] > train_nn_model[i][2]) else 1 if (train_nn_model[i][1] > train_nn_model[i][0] and train_nn_model[i][1] > train_nn_model[i][2]) else 2
                    train_pred_ergebnis.append(ergebnis)
            #Test-Akkuranz
            # Leeren array füllen mit Ergebnissen aus Ergebnis-Tensor
            # Hierbei werden die probalistischen Werte verglichen und das wahrscheinlichste Ergebnis übergeben
            # als 0 - Heimsieg, 1 - Unentschieden, 2 - Auswärtssieg
                for i in range(test_nn_model.shape[0]):
                    ergebnis = 0 if (test_nn_model[i][0] > test_nn_model[i][1] and test_nn_model[i][0] > test_nn_model[i][2]) else 1 if (test_nn_model[i][1] > test_nn_model[i][0] and test_nn_model[i][1] > test_nn_model[i][2]) else 2
                    test_pred_ergebnis.append(ergebnis)

            #Arrays in tensor umwandeln
            train_pred_tensor = torch.tensor(train_pred_ergebnis, dtype=torch.float)
            test_pred_tensor = torch.tensor(test_pred_ergebnis, dtype=torch.float)

            #Die korrekten Ergebnisse aus dem Traininsdatensatz werden aufsummiert und
            #daraus die Akkuranz dieser Epoche berechnet und dem Array epoch_train_accuracy für spätere Auswertung übergeben
            train_running_correct += (train_pred_tensor == y_Train).sum().item()
            train_accuracy = train_running_correct*100./y_Train.shape[0]
            epoch_train_accuracy.append(train_accuracy)

            #Die korrekten Ergebnisse aus dem Testdatensatz werden aufsummiert und
            #daraus die Akkuranz dieser Epoche berechnet und dem Array epoch_test_accuracy für spätere Auswertung übergeben
            test_running_correct += (test_pred_tensor == y_Test).sum().item()
            test_accuracy = test_running_correct*100./y_Test.shape[0]
            epoch_test_accuracy.append(test_accuracy)
            
            #---------------------------------------------------------------------------------------
            #Hier werden nun die entscheidenden Schritte zum Trainineren des NN Modells durchgeführt
            #---------------------------------------------------------------------------------------
            error = loss_func(nn_model(X_Train),y_Train)
            optimizer.zero_grad()
            error.backward()
            epoch_errors.append(error.item())
            optimizer.step()
            #---------------------------------------------------------------------------------------
            
            #Debug-Print Ausgabe der Epoche mit Akkuranzen
            print("Epoche: {}/{} mit Train-Akkuranz: {} und Test-Akkuranz: {}".format(epoch, epochs, train_accuracy, test_accuracy))

            #-----------------------------
            #Early Stopping
            #-----------------------------
            #Loss für Testdaten berechnen
            error_Test = loss_func(nn_model(X_Test),y_Test)

            #Aufruf der Early Stopping Funktion
            # Die Fehlerfunkion der Testdaten dient hier als zentrales Kriterium:
            # Sinkt diese mit der Rate "delta" eine bestimmte Anzahl Schritte "patience" 
            # hintereinander NICHT MEHR, wird gestoppt. 
            # Zusätzlich wird ein Threshold mit angegeben, sodass erst ab einer bestimmten erreichten
            # Akkuranz das Early Stopping aktiviert wird.
            early_stopping(error_Test, nn_model, train_accuracy > threshold_for_early_stopping)
            #Sollte ein Early Stop erreicht sein, wird das Durchlaufen der Epochen unterbrochen
            if early_stopping.early_stop:
                print("Early stopping")
                break
            #-----------------------------

        #Debug-Print finales Loss-Ergebnis
        #print('Loss nach {} Epochen: {}'.format(epoch+1,error.item()))

        #Übergabe der Ergebnisdaten and den zentralen Ergebnis-Array
        self.ergebnis.append([nn_model.__class__.__name__, test_accuracy/100, nn_model])

        #Rückgabewerte für weitere Verwendung (Ausgabe, Test) im Hauptprogramm
        return self.ergebnis, epoch_errors, epoch_train_accuracy, epoch_test_accuracy, test_pred_tensor


class NN_Model(torch.nn.Module):
    def __init__(self):
        super(NN_Model, self).__init__()
        self.fc1 = nn.Linear(75,120)
        self.fc2 = nn.Linear(120,180)
        self.fc3 = nn.Linear(180,100)
        self.fc4 = nn.Linear(100,40)
        self.output = nn.Linear(40,3)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        #Keine Softmax-Funktion benötigt bei output, da CrossEntropyLoss
        #als Fehlerfunktion dies automatisch tut
        #Bemerkung: softmax muss aber beim Validieren/Testen angewandt werden!
        x = self.output(x)

        return x