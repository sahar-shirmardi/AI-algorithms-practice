import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score

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

# manual KNN
def knn_predict(X_train, X_test, Y_train, Y_test, k, p):

    Y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        
        df_dists = pd.DataFrame(data=distances, columns=['dist'], index=Y_train.index)
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]
        counter = Counter(Y_train[df_nn.index])
        prediction = counter.most_common()[0][0]
        Y_hat_test.append(prediction)
        
    return Y_hat_test

# KNN model
k_range = range(1, 26)
scores = []

for k in k_range:
    Y_hat_test = knn_predict(X_train, X_test, Y_train, Y_test, k, p=1)
    scores.append(accuracy_score(Y_test, Y_hat_test))

for score in scores:
    print(score)