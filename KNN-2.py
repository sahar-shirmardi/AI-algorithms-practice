import pandas as pd

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

