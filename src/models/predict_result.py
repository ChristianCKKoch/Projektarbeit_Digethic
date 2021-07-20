#Dies ist das Python-File zur Prädiktion von Fußball-Ergebnisse
import pandas as pd

#Model und Scaler aus Dateien lesen
clf = pd.read_pickle(r'models/classifier_object.pickle')
sca = pd.read_pickle(r'models/scaler_object.pickle')

#Zu bestimmende Werte aus Excel-Datei lesen
data = pd.read_excel('data/raw/Bundesliga_Vorhersage.xlsx', sheet_name="Ohne_Resultat")

#Normalisieren der Daten
X_output = sca.transform(data)

#Anwenden des Modells
y_pred = clf.predict(X_output)

#Ausgabe der vorhergesagten Ergebnisse
print(y_pred)