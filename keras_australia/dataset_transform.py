import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime

dataset = pd.read_csv('weatherAUS.csv')

''' MONTHS in case its needed
dates = dataset.iloc[:, 0].values#.join(dataset.iloc[:, 5:7])
months = []
for date in dates:
    if int(date[5:7]) <= 3:
        months.append(1)
    elif int(date[5:7]) <= 6:
        months.append(2)
    elif int(date[5:7]) <= 9:
        months.append(3)
    else:
        months.append(4)
data = {'Month':months}
df_months = pd.DataFrame(data)
'''

################ ENCODE SPECIFIC DATASET FEATURES
'''
# DATES to DAY of the year
days = []
for date in dataset.iloc[:, 0].values:
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    days.append(dt.timetuple().tm_yday)

# LOCATION - one hot encode
ohe = OneHotEncoder(sparse=False)
locations = ohe.fit_transform(dataset.iloc[:, 1].values.reshape(-1, 1))



# GROUP - (Month, Location) - MinTemp, MaxTemp, Rainfall
group = dataset.iloc[:, :5].groupby( [dataset.Date.str[5:7],'Location'] ).mean().reset_index()

# MIN TEMP - replace NAs with average for the selected month and city
minTemps = []
for triple in dataset.iloc[:, :3].values:
    # NaN check
    if triple[2] == triple[2]:
        minTemps.append(triple[2])
    else:
        minTemps.append( group.loc[(group['Date'] == triple[0][5:7]) & (group['Location'] == triple[1]), 'MinTemp'].iloc[0] )

# MAX TEMP - ditto
maxTemps = []
for triple in dataset.iloc[:, :2].join(dataset.iloc[:, 3]).values:
    # NaN check
    if triple[2] == triple[2]:
        maxTemps.append(triple[2])
    else:
        maxTemps.append( group.loc[(group['Date'] == triple[0][5:7]) & (group['Location'] == triple[1]), 'MaxTemp'].iloc[0] )

# RAINFALL - ditto
rainfalls = []
for triple in dataset.iloc[:, :2].join(dataset.iloc[:, 4]).values:
    # NaN check
    if triple[2] == triple[2]:
        rainfalls.append(triple[2])
    else:
        rainfalls.append( group.loc[(group['Date'] == triple[0][5:7]) & (group['Location'] == triple[1]), 'Rainfall'].iloc[0] )



# GROUP - (Year and Month) - Evaporation, Sunshine
group = dataset.iloc[:, :1].join(dataset.iloc[:, 5:7]).groupby( [dataset.Date.str[:7]] ).mean().reset_index()

# EVAPORATION - replace NAs with average for selected year and month globally
evaporations = []
for double in dataset.iloc[:, :1].join(dataset.iloc[:, 5]).values:
    # NaN check
    if double[1] == double[1]:
        evaporations.append(double[1])
    else:
        evaporations.append( group.loc[group['Date'] == double[0][:7], 'Evaporation'].iloc[0] )

# SUNSHINE - ditto
sunshines = []
for double in dataset.iloc[:, :1].join(dataset.iloc[:, 6]).values:
    # NaN check
    if double[1] == double[1]:
        sunshines.append(double[1])
    else:
        sunshines.append( group.loc[group['Date'] == double[0][:7], 'Sunshine'].iloc[0] )
'''


# WINDDIR9AM, WINDDIR3PM, WINDGUSTDIR - one hot encode
ohe = OneHotEncoder(sparse=False)
windDirs = ohe.fit_transform(dataset.iloc[:, 9:11].join(dataset.iloc[:, 7]))
print(windDirs.shape)
