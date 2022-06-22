import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


## Dataset transformation - Drop all NA values, 

dataset = pd.read_csv('Train.csv').dropna().reset_index(drop=True)

#### ENCODE SPECIFIC DATASET FEATURES

# Gender, Married, Graduated - to binary 0/1
gender = []
married = []
for couple in dataset.iloc[:, 1:3].values:
    if couple[0] == 'Male':
        gender.append(0)
    else:
        gender.append(1)
    if couple[1] == 'No':
        married.append(0)
    else:
        married.append(1)
graduated = []
for g in dataset.iloc[:, 4].values:
    if g == 'No':
        graduated.append(0)
    else:
        graduated.append(1)


# Profession, Var_1 - One hot encode
ohe = OneHotEncoder(sparse=False)
profession = ohe.fit_transform(dataset.iloc[:, 5].values.reshape(-1, 1))
var_1 = ohe.fit_transform(dataset.iloc[:, 9].values.reshape(-1, 1))


# Spending score - (-1), 0, 1 based on low, avg, high
spending_score = []
for ss in dataset.iloc[:, 7].values:
    if ss == 'Low':
        spending_score.append(-1)
    elif ss == 'Average':
        spending_score.append(0)
    else:
        spending_score.append(1)


# Segmentation classes - label encode
le = LabelEncoder()
segmentation = le.fit_transform(dataset.iloc[:, 10].values)


#### DROP ORIGINAL TRANSFORMED COLUMNS FROM DATASET AND ADD NEW ONES

dataset = dataset.drop(columns=['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1', 'Segmentation'])

dataset['Gender'] = pd.Series(gender)
dataset['Ever_Married'] = pd.Series(married)
dataset['Graduated'] = pd.Series(graduated)
dataset['Spending_Score'] = pd.Series(spending_score)
for i in range(profession.shape[1]):
    dataset['Profession_' + str(i)] = pd.Series(profession[:, i])
for i in range(var_1.shape[1]):
    dataset['Var_' + str(i)] = pd.Series(var_1[:, i])
dataset['Segmentation'] = pd.Series(segmentation)

dataset.to_csv("Train_transformed.csv", index=False)
