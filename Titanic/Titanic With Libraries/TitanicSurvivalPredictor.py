# Random Forest Classification Model to 
# Predict Survival on Titanic

# Importing Libraries

import numpy as np
import pandas as pd
import re
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Importing Datasets

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Data Analysis

#train_df.info()
#print(train_df.head())

# Sex
#sns.barplot(x = 'Sex', y = 'Survived', data = train_df)

# Embarked
#FacetGrid = sns.FacetGrid(train_df, row = 'Embarked', aspect = 1.5)
#FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order = None)
#FacetGrid.add_legend()

# Pclass
#sns.barplot(x = 'Pclass', y = 'Survived', data = train_df)

# SibSp + Parch
train_df['relatives'] = train_df['SibSp'] + train_df['Parch']
test_df['relatives'] = test_df['SibSp'] + test_df['Parch']
#sns.factorplot('relatives', 'Survived', data = train_df, aspect = 2.5)
#sns.barplot(x = 'relatives', y = 'Survived', data = train_df)

# Age
#women = train_df[train_df['Sex'] == 'female']
#men = train_df[train_df['Sex'] == 'male']
#survived = 'survived'
#not_survived = 'not survived'
#plt.subplot(1, 2, 1)
#ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins = 18, label = survived, kde =False)
#ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins = 36, label = not_survived, kde =False)
#ax.legend()
#plt.subplot(1, 2, 2)
#ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins = 18, label = survived, kde =False)
#ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins = 36, label = not_survived, kde =False)
#ax.legend()

# Data Preprocessing

Y = train_df['Survived']
train_df.drop('Survived', inplace = True, axis = 1)

genders = {"male": 0, "female": 1}
train_df['Sex'] = train_df['Sex'].map(genders)

train_df['Embarked'] = train_df['Embarked'].fillna('S')
ports = {"S": 0, "C": 1, "Q": 2}
train_df['Embarked'] = train_df['Embarked'].map(ports)

train_df["Fare"] = train_df["Fare"].astype(int)

ageMean = train_df["Age"].mean()
ageStd = test_df["Age"].std()
is_null = train_df["Age"].isnull().sum()
randomAge = np.random.randint(ageMean - ageStd, ageMean + ageStd, size = is_null)
ageCopy = train_df["Age"].copy()
ageCopy[np.isnan(ageCopy)] = randomAge
train_df["Age"] = ageCopy
train_df["Age"] = train_df["Age"].astype(int)
train_df.loc[ train_df['Age'] <= 11, 'Age'] = 0
train_df.loc[(train_df['Age'] > 11) & (train_df['Age'] <= 18), 'Age'] = 1
train_df.loc[(train_df['Age'] > 18) & (train_df['Age'] <= 22), 'Age'] = 2
train_df.loc[(train_df['Age'] > 22) & (train_df['Age'] <= 27), 'Age'] = 3
train_df.loc[(train_df['Age'] > 27) & (train_df['Age'] <= 33), 'Age'] = 4
train_df.loc[(train_df['Age'] > 33) & (train_df['Age'] <= 40), 'Age'] = 5
train_df.loc[(train_df['Age'] > 40) & (train_df['Age'] <= 66), 'Age'] = 6
train_df.loc[ train_df['Age'] > 66, 'Age'] = 6

train_df['Age_Class']= train_df['Age']* train_df['Pclass']

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
train_df['Cabin'] = train_df['Cabin'].fillna("U0")
train_df['Deck'] = train_df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
train_df['Deck'] = train_df['Deck'].map(deck)
train_df['Deck'] = train_df['Deck'].fillna(0)
train_df['Deck'] = train_df['Deck'].astype(int)

train_df.drop('PassengerId', inplace = True, axis = 1)
train_df.drop('Name', inplace = True, axis = 1)
train_df.drop('Ticket', inplace = True, axis = 1)
train_df.drop('Cabin', inplace = True, axis = 1)
train_df.drop('Parch', inplace = True, axis = 1)

train_df.info()

# Model

X_train, X_test, y_train, y_test = train_test_split(train_df, Y, test_size = 0.2)
classifier = RandomForestClassifier(criterion = "gini", n_estimators = 100, min_samples_leaf = 1, min_samples_split = 10, max_features='auto', random_state=1, n_jobs=-1)
classifier.fit(X_train, y_train)

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(classifier.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances.head(10))

prediction = classifier.predict(X_test)
print(prediction)
accuracy = metrics.accuracy_score(y_test, prediction)
print(accuracy)



# Predictions

df_predictions = pd.DataFrame()
df_predictions['PassengerId'] = test_df['PassengerId']

test_df['Sex'] = test_df['Sex'].map(genders)

test_df['Embarked'] = test_df['Embarked'].fillna('S')
test_df['Embarked'] = test_df['Embarked'].map(ports)

is_null = test_df["Age"].isnull().sum()
randomAge = np.random.randint(ageMean - ageStd, ageMean + ageStd, size = is_null)
ageCopy = test_df["Age"].copy()
ageCopy[np.isnan(ageCopy)] = randomAge
test_df["Age"] = ageCopy
test_df["Age"] = test_df["Age"].astype(int)
test_df.loc[ test_df['Age'] <= 11, 'Age'] = 0
test_df.loc[(test_df['Age'] > 11) & (test_df['Age'] <= 18), 'Age'] = 1
test_df.loc[(test_df['Age'] > 18) & (test_df['Age'] <= 22), 'Age'] = 2
test_df.loc[(test_df['Age'] > 22) & (test_df['Age'] <= 27), 'Age'] = 3
test_df.loc[(test_df['Age'] > 27) & (test_df['Age'] <= 33), 'Age'] = 4
test_df.loc[(test_df['Age'] > 33) & (test_df['Age'] <= 40), 'Age'] = 5
test_df.loc[(test_df['Age'] > 40) & (test_df['Age'] <= 66), 'Age'] = 6
test_df.loc[ test_df['Age'] > 66, 'Age'] = 6

test_df['Age_Class']= test_df['Age']* test_df['Pclass']

test_df['Fare'].replace(np.NaN, 0, inplace = True)
test_df["Fare"] = test_df["Fare"].astype(int)
                                         
test_df['Cabin'] = test_df['Cabin'].fillna("U0")
test_df['Deck'] = test_df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
test_df['Deck'] = test_df['Deck'].map(deck)
test_df['Deck'] = test_df['Deck'].fillna(0)
test_df['Deck'] = test_df['Deck'].astype(int)

test_df.drop('PassengerId', inplace = True, axis = 1)
test_df.drop('Name', inplace = True, axis = 1)
test_df.drop('Ticket', inplace = True, axis = 1)
test_df.drop('Cabin', inplace = True, axis = 1)
test_df.drop('Parch', inplace = True, axis = 1)

test_df.info()

predictions = classifier.predict(test_df)
df_predictions['Survived'] = predictions
df_predictions.to_csv('Predictions.csv', index = False)  


