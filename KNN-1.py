import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Import dataset
df_train = pd.read_csv('./Survivors/train.csv')
df_test = pd.read_csv('./Survivors/test.csv')

# Feature selection
df_train_set = df_train.drop(['PassengerId',
                            'Name', 
                            'Ticket', 
                            'Cabin', 
                            'SibSp', 
                            'Parch', 
                            'Age'], 
                            axis=1)
                            
df_test_set = df_train.drop(['Name', 
                            'Ticket', 
                            'Cabin', 
                            'SibSp', 
                            'Parch', 
                            'Age'], 
                            axis=1)

# Fill in NaN with mean
mean = df_test_set['Fare'].mean()
df_test_set['Fare'] = df_test_set['Fare'].fillna(mean)

# Categorical featureas to numerical 
labelencoder = LabelEncoder()
df_train_set.iloc[:, 2] = labelencoder.fit_transform(df_train_set.iloc[:, 2].values)
df_train_set.iloc[:, 4] = labelencoder.fit_transform(df_train_set.iloc[:, 4].values)
df_test_set.iloc[:, 2] = labelencoder.fit_transform(df_test_set.iloc[:, 2].values)
df_test_set.iloc[:, 4] = labelencoder.fit_transform(df_test_set.iloc[:, 4].values)

# Split
X = df_train_set.iloc[:, 1:5].values
Y = df_train_set.iloc[:, 0].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=4)

# Calculate distance between two points
def minkowski_distance(a, b, p=1):
    
    dim = len(a)
    distance = 0

    for d in range(dim):
        distance += abs(a[d] - b[d])**p
        
    distance = distance**(1/p)
    
    return distance
print(minkowski_distance(a=df_train_set.iloc[:, 2], b=df_train_set.iloc[:, 1], p=1))
