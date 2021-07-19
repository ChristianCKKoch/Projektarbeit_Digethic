import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from preprocessor import Preprocessor
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Verbindung zu sqlite Datenbank öffnen
connection = sqlite3.connect("data/raw/database.sqlite")

#SQL-Abfrage in pandas DataFrame
sql = "SELECT * FROM vw_Match_Team_Data"

daten = pd.read_sql_query(sql, connection)
connection.close()

#Korrelationen der Eingangsvariablen erkennen
corr_matrix = daten.corr()

plt.figure(figsize=(20, 20))
sns.set(font_scale=0.7)
heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, fmt='.3f', linewidths=.1)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=8);
plt.show()

print(abs(corr_matrix["result"]).sort_values(ascending=False))

attributes = ["result", "away_Gesamtmarktwert", "home_Gesamtmarktwert", "home_chanceCreationCrossing", "away_Kadergroesse", "home_Kadergroesse"]
scatter_matrix(daten[attributes])
#scatter_matrix(daten)
plt.show()

onehot_enc_list = list()
daten_join = daten

#Teamname-Spalten encoden (one-hot-encoding)
for spalte in ('home_team_name','away_team_name'):
    team_name = daten[spalte]
    #Anpassung der Form des Vektors mit den Werten der Teamnamen
    team_name_res = team_name.values.reshape(-1,1)
    #Initialisierung des One Hot Encoders
    team_name_onehot_enc = OneHotEncoder()
    #Fit des One Hot Encoders
    team_name_onehot = team_name_onehot_enc.fit_transform(team_name_res)
    
    #Die in alphabetischer Reihenfolge angeordneten Teamnamen werden in einen Array geladen
    sp_name_array = team_name_onehot_enc.categories_[0]
    #Dieser Array wird durchlaufen und die Teamnamen im Falle der Heimteams um die Endung "_H" bzw.
    #im Falle der Auswärtsteams durch die Endung "_A" erweitert
    for i in range(0, sp_name_array.size):
        sp_name_array[i] = sp_name_array[i] + '_H' if spalte == 'home_team_name' else sp_name_array[i] + '_A'

    #Zusammenführen (Join) der One-hot-encodeten Team-Namen mit dem ursprünglichen Datensatz
    daten_join = daten_join.join(pd.DataFrame(team_name_onehot.toarray(), columns=sp_name_array))

result_spalte = daten_join.pop("result")
daten_join['result'] = result_spalte
daten_prepared = daten_join.drop(["home_team_name", "away_team_name"], axis=1)

# print(onehot_enc_list[1][1].toarray())
# print("Array-Objekt 0")
# print(onehot_enc_list[1][2].categories_[0])

# sp_name_array = onehot_enc_list[1][2].categories_[0]
# for i in range(0, sp_name_array.size):
#     sp_name_array[i] = sp_name_array[i] + '_H'

# daten_join = daten.join(pd.DataFrame(onehot_enc_list[1][1].toarray(), columns=sp_name_array))

daten_prepared.to_excel("data/processed/Bundesliga_daten.ods")

X = daten_prepared.loc[:,daten_prepared.columns != "result"]
y = daten_prepared["result"]

X_train, X_test, y_train, y_test, scaler = Preprocessor(X, y).get_data()

X_train_exp = pd.DataFrame(X_train)
X_train_exp.to_excel("data/processed/Bundesliga_daten_X_normalized.xlsx")

daten_prepared.to_pickle("data/processed/Prepared_Data.pickle")