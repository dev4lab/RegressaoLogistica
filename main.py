import pandas as pd

df = pd.read_csv("dados/high_diamond_ranked_10min.csv", index_col=0)
pd.set_option('display.max_columns', None) 
df.head(3)

#Para fazer a regressão logística, precisamos que os dados sejam inteiros
#e não floats

df = df.astype(int)

df_blue = df[['blueWins','blueWardsPlaced','blueWardsDestroyed','blueFirstBlood','blueKills',
'blueDeaths','blueAssists','blueEliteMonsters','blueDragons','blueHeralds','blueTowersDestroyed',
'blueTotalGold','blueAvgLevel','blueTotalExperience','blueTotalMinionsKilled',
'blueTotalJungleMinionsKilled','blueGoldDiff','blueExperienceDiff','blueCSPerMin','blueGoldPerMin']]

"""
Para realizar a regressão precisamos estabelecer quem será X e quem será Y.

Y será a variável que assumirá os valores "vitória" ou "derrota"

Todo o resto das variáveis (colunas de df_blue) serão consideradas valores de X.

"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
np.set_printoptions(precision=6, suppress=True)

X = df_blue.iloc[:,df_blue.columns != "blueWins"]
Y = df_blue.iloc[:, 0]

lr = LogisticRegression()
lr.fit(X, Y) 
Y_pred = lr.predict(X)

column_labels = np.array(X.columns.values) 
column_labels # to remind us what each label is for each coefficient

coef = lr.coef_
print(coef)