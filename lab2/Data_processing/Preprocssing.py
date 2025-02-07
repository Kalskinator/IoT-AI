import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('lab2/Data_processing/Data.csv')
# print(df.head())
# print(df.tail())
# print(df.info())
# print(pd.isna(df))
df.describe().transpose()


# #Sort by country name
# sorted = df.sort_values(by=['Country'])
# print(sorted)

# #Filter rows , show only Purchased rows
# purchased_df = df.query('Purchased=="Yes"')
# print(purchased_df)

# #Filter columns, show specific columns
# filltered_df = df.filter(['Country','Age','Salary'])
# print(filltered_df)

# #Remove duplicates
# print(df.duplicated())
# dups_removed = df.drop_duplicates()
# print(dups_removed)

# #Rename column, change the "Age" column name
# renamed = df.rename(columns={'Age':'Min_Age'})
# print(renamed)


# df.info()
# pd.isna(df)
# df.isna().sum()

# df['Age'] = df['Age'].fillna(df['Age'].mean())
# df['Salary'] = df['Salary'].fillna(df['Salary'].median())

df['Age'] = df['Age'].ffill()
df['Salary'] = df['Salary'].bfill()


# print(df)

cols = ['Age', 'Salary'] #selecting and saving the columns that need to be removed
df_2 = df.drop(cols, axis=1) #removing the selected columns. axis 1 refers to columns and axis 0 refers to rows

# print(df_2)


newdf = df.copy()
newdf = newdf.replace(83000, -83000)


# print(newdf.describe().transpose())

sal = newdf['Salary']
bad_sal = sal == -83000.0 #save the bad_data in salary if sal is equal to -83000
sal[bad_sal] = 83000.0 #Changing all -83000 values in salary with 83000
# The above inplace edits are reflected in the DataFrame. New lets see the dataframe
print(newdf['Salary'].min()) 
