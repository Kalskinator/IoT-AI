import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#New Variable
from dateutil.relativedelta import *
from datetime import *


df = pd.read_csv('lab2/Data_processing/student-data.csv')


def get_age(dob):
    now = datetime.now()
    age = relativedelta(now, dob).years
    return age 

df['age'] = pd.to_datetime(df['dob']).apply(get_age)

# print(df)

#Splitting
splitnames = df.copy()
split = splitnames['name'].str.split(' ', expand = True)
splitnames['first'] = split[0]
splitnames['last'] = split[1]
# print(splitnames)

#adding a column for abb and combining
splitnames['name_abb'] = splitnames['first'] + ', ' + splitnames['last']
# print(splitnames)

#Data Value transforms 
# df['target'].hist() #The function plots the distribution of the data in the "target" column

#Log transform
# transforms = df.copy()
# transforms['log'] = transforms['target'].transform(np.log10)
# transforms['log'].hist()

#Cube transform
transforms = df.copy()
transforms['cube'] = transforms['target'].transform(lambda x: np.power(x, 3))
transforms['cube'].hist()

# plt.show()

# # Standardizing only numeric columns
# numeric_cols = df.select_dtypes(include=[np.number]).columns
# standardized_df = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())

# #OR

# scaling = df.copy()
# mean_target = np.mean(scaling['target'])
# sd_target = np.std(scaling['target'])
# scaling['standardized_target'] = (scaling['target'] - mean_target) / (sd_target)
# print(scaling)

# #Normalizing
# normalized_df=(df-df.min())/(df.max()-df.min())

# #OR
# scaling = df.copy()
# min_target = np.min(scaling['target'])
# max_target = np.max(scaling['target'])
# scaling['norm_target'] = (scaling['target'] - min_target) / (max_target - max_target)
# print(scaling)

df.drop(["participant_id"],axis=1,inplace=True) #axis=1 referes to colummns

print(df)

# Assuming 'target' is the column to predict
X = df.drop(columns=['target'])
Y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(X_train)
print(X_test)
print(y_train)
print(y_test)