import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import re

#Splittin dataset into columns.
df = pd.read_csv(r'C:\Users\nezih\Desktop\bankdata\bank-full.csv',sep = "?")
quotes_strip = list(df.columns)[0].replace('"','')
columns_split = quotes_strip.split(';')
df = df[df.iloc[:,0].name].str.split(pat = ';',expand = True)
df.columns =  columns_split
df.replace('"','',regex = True,inplace = True)

feature_list = list(df.columns.values)

def convert_categorical(df):
    categorical_features = []
    letter_pattern = re.compile(r'[A-z]')
    #If values types are all str or int,it is impossible to distinguish them with this method,so i prefer to do it with regex.
    for column in feature_list:

        if letter_pattern.match(str(df[column].values[0])):
            df[column] = pd.Categorical(df[column])
            categorical_features.append(df[column].name)
        
    return list(set(categorical_features)),df

categorical_features,df = convert_categorical(df)
numerical_features = [name for name in feature_list if name not in categorical_features]

numerical_df = df[numerical_features]
categorical_df = df[categorical_features]

#unless numerical features are converted into int,it won't group them by categorical ones.
for feature in numerical_features:
    df[feature] = df[feature].astype('int')


def groupby_method(groupby_features,method):
        series_groupby = df[groupby_features].groupby(df[groupby_features].iloc[:,-1].name)
        if method == 'sum':
            return series_groupby.sum()
        elif method == 'mean':
            return series_groupby.mean()