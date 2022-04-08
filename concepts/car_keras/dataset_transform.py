import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


## Dataset transformation - Drop all NA values, 

dataset = pd.read_csv('Car.csv', index_col=0).dropna().reset_index(drop=True)

#### ENCODE SPECIFIC DATASET FEATURES

ohe = OneHotEncoder(sparse=False)
types = []
for i in range(4, 10):
    tmp = ohe.fit_transform(dataset.iloc[:, i].values.reshape(-1, 1))
    types.append(tmp)

fuels = []
for i in range(10, 16):
    tmp = ohe.fit_transform(dataset.iloc[:, i].values.reshape(-1, 1))
    fuels.append(tmp)

le = LabelEncoder()
choice = le.fit_transform(dataset.iloc[:, 0].values)


#### DROP ORIGINAL TRANSFORMED COLUMNS FROM DATASET AND ADD NEW ONES

dataset = dataset.drop(columns=['choice'])
for i in range(1, 7):
    dataset = dataset.drop(columns=['type'+str(i), 'fuel'+str(i)])


for i in range(6):
    for j in range(types[i].shape[1]):
        dataset['type' + str(i+1) + '_' + str(j+1)] = pd.Series(types[i][:, j])

for i in range(6):
    for j in range(fuels[i].shape[1]):
        dataset['fuel' + str(i+1) + '_' + str(j+1)] = pd.Series(fuels[i][:, j])

dataset['choice'] = pd.Series(choice)


dataset.to_csv("Car_transformed.csv", index=False)
