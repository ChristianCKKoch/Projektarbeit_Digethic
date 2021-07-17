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

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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
                for i in range(1, 40):
                    knn = KNeighborsClassifier(n_neighbors=i)
                    knn.fit(self.X_train, self.y_train)
                    pred_i = knn.predict(self.X_test)
                    error.append(np.mean(pred_i != self.y_test))

                #Knn-Classifier trainieren
                knnclf = KNeighborsClassifier(n_neighbors=7)
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
                tree_para = {'criterion':['gini','entropy'],'max_depth':[i for i in range(1,20)], 'min_samples_split':[i for i in range (2,20)]}

                #GridSearchCV 
                grd_clf = GridSearchCV(dt, tree_para, cv=5)
                grd_clf.fit(self.X_train, self.y_train)

                #Besten gefundenen Decision Tree übergeben
                dt_clf = grd_clf.best_estimator_

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
                svm = SVC(kernel = 'poly')
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
                print("iterations: {}; layers: {}; loss: {}".format(mlp.n_iter_, mlp.n_layers_, mlp.loss_))
                epochs = np.linspace(1,mlp.n_iter_, mlp.n_iter_)

        return self.ergebnis
    
    def ensemble_model(self):
        
        models = list()
        for model in self.ergebnis:
            models.append([model[0], model[2]])
        
        voting_clf = VotingClassifier(estimators=models, voting='hard')
        voting_clf.fit(self.X_train, self.y_train)
        score = voting_clf.score(self.X_test, self.y_test)
        self.ergebnis.append([voting_clf.__class__.__name__, score, voting_clf])

        return self.ergebnis

    def neuronal_network(self):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('This Computation is running on {}'.format(device))

        nn_model = NN_Model()
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
        epoch_errors = []

        #X = torch.tensor(self.X_train, dtype=torch.float).reshape(-1,1)
        X = torch.from_numpy(self.X_train).float()
        #y = torch.from_numpy(self.y_train).long()
        y = torch.tensor(self.y_train, dtype=torch.long)

        for epoch in range(1000):
            error = loss_func(nn_model(X),y)
            optimizer.zero_grad()
            error.backward()
            epoch_errors.append(error.item())
            optimizer.step()

        print('Loss nach {} Iterationen: {}'.format(epoch+1,error.item()))

        plt.plot(epoch_errors)
        plt.show()

class NN_Model(torch.nn.Module):
    def __init__(self):
        super(NN_Model, self).__init__()
        self.fc1 = nn.Linear(77,120)
        self.fc2 = nn.Linear(120,40)
        self.output = nn.Linear(40,3)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.output(x)

        return x