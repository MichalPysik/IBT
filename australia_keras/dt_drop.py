import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime

## Dataset transformation - Version with dropping all rows containing NA values

dataset = pd.read_csv('weatherAUS.csv').dropna().reset_index(drop=True)



#### ENCODE SPECIFIC DATASET FEATURES

# DATES to DAY of the year
days = []
for date in dataset.iloc[:, 0].values:
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    days.append(dt.timetuple().tm_yday)

# LOCATION - one hot encode
ohe = OneHotEncoder(sparse=False)
locations = ohe.fit_transform(dataset.iloc[:, 1].values.reshape(-1, 1))

# WIND(GUST)DIR(9am/3pm) - one hot encode
windDirs = ohe.fit_transform(dataset.iloc[:, 9:11].join(dataset.iloc[:, 7]))

# RAINTODAY/TOMMOROW - convert yes/no to binary
rainTodays = []
rainTommorows = []
for r in dataset.iloc[:, 21:].values:
    rainTodays.append(int(r[0] == 'Yes'))
    rainTommorows.append(int(r[1] == 'Yes'))



#### DROP ORIGINAL TRANSFORMED COLUMNS FROM DATASET AND ADD NEW ONES

dataset = dataset.drop(columns=['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow'])

dataset['Day'] = pd.Series(days)

for i in range( locations.shape[1] ):
    dataset['Location' + str(i)] = pd.Series(locations[:, i])

for i in range( windDirs.shape[1] ):
    dataset['WindDir' + str(i)] = pd.Series(windDirs[:, i])

dataset['RainToday'] = pd.Series(rainTodays)
dataset['RainTomorrow'] = pd.Series(rainTommorows)



dataset.to_csv("weatherAUS_transformed_drop.csv")