import pandas as pd

#https://jonathan-sands.com/eda/tabular/multilabel/pandas/seaborn/matplotlib/2021/01/16/Drug-consumption-analysis.html#Dataset-preparation

demographic_columns = [
    'Age', 
    'Gender', 
    'Education', 
    'Country',
    'Ethnicity',
]

personality_columns = [
    'Neuroticism',
    'Extraversion',
    'Openness to experience',
    'Agreeableness',
    'Conscientiousness',
    'Impulsiveness',
    'Sensation seeking'
]

feature_columns = demographic_columns + personality_columns

drugs_columns = [
    'Alcohol consumption',
    'Amphetamines consumption',
    'Amyl nitrite consumption',
    'Benzodiazepine consumption',
    'Caffeine consumption',
    'Cannabis consumption',
    'Chocolate consumption',
    'Cocaine consumption',
    'Crack consumption',
    'Ecstasy consumption',
    'Heroin consumption',
    'Ketamine consumption',
    'Legal highs consumption',
    'Lysergic acid diethylamide consumption',
    'Methadone consumption',
    'Magic mushrooms consumption',
    'Nicotine consumption',
    'Fictitious drug Semeron consumption',
    'Volatile substance abuse consumption'
]

all_columns = feature_columns + drugs_columns

dataset = pd.read_csv("drug_consumption.data", names=["ID"] + all_columns)
dataset = dataset.drop(['ID'], axis=1)
dataset.head()

for i in drugs_columns:
    dataset[i] = dataset[i].map({'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6})

dataset.to_csv('drug_consumption.csv', index=False)