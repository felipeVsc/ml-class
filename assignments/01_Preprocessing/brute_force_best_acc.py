# BRUTE FORCE BEST SCORE
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from itertools import combinations

data = pd.read_csv(r'S:\Cloud\Pessoal\machine-learning\assignments\01_Preprocessing\diabetes_dataset.csv')
cols = list(data.columns)
cols.remove('Outcome')

#fill
class0, class1 = data[data['Outcome'] == 0].copy(), data[data['Outcome'] == 1].copy()
for column in cols:
    c0m, c1m = class0[column].mean(), class1[column].mean()
    class0[column].fillna(value=c0m, inplace=True)
    class1[column].fillna(value=c1m, inplace=True)

data = pd.concat([class0, class1],axis=0)

#normalize
zscore_cols = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'SkinThickness']
for column in zscore_cols:
    #metrics
    mean = data[column].mean()
    std = data[column].std()
    #calculate z-score
    data[column] = (data[column] - mean) / std

#age bins
data['Age'] = data['Age'].apply(lambda x: 6 if x >= 60 else x // 10)

accs = {}
stds = {}
for pairing in range(1,len(cols)):
    for features in combinations(cols, pairing):
        selected = data[list(features)+["Outcome"]].copy()
        X = selected[list(features)]

        y = selected['Outcome']
        knn = KNeighborsClassifier(n_neighbors=3)
        scores = cross_val_score(knn, X, y, cv=5)

        title = '\',\''.join(features)
        accs[title] = scores.mean()
        stds[title] = scores.std()

order = sorted(accs.items(), reverse=True, key=lambda x:x[1])
for i in range(50):
    key = order[i][0]
    print(f'\'{key}\': {accs[key]:.2f} +/- {stds[key]:.2f}')
