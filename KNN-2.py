import pandas as pd
from sklearn.preprocessing import LabelEncoder

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


