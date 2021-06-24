# Logistic Regression Model to Predict 
# Survival on Titanic based on  
# Ticket Class, Gender and Age
# Without using any External 
# Machine Learning Library

# Importing Libraries

import pandas as pd
import numpy as np
import math


df_train = pd.read_csv('train.csv')
ageMean = df_train['Age'].mean()
df_train['Age'].replace(np.NaN, ageMean, inplace = True) # Replacing Null values with Mean of Ages
df_train.loc[ df_train['Age'] <= 11, 'Age'] = 0
df_train.loc[(df_train['Age'] > 11) & (df_train['Age'] <= 18), 'Age'] = 1
df_train.loc[(df_train['Age'] > 18) & (df_train['Age'] <= 22), 'Age'] = 2
df_train.loc[(df_train['Age'] > 22) & (df_train['Age'] <= 27), 'Age'] = 3
df_train.loc[(df_train['Age'] > 27) & (df_train['Age'] <= 33), 'Age'] = 4
df_train.loc[(df_train['Age'] > 33) & (df_train['Age'] <= 40), 'Age'] = 5
df_train.loc[(df_train['Age'] > 40) & (df_train['Age'] <= 66), 'Age'] = 6
df_train.loc[ df_train['Age'] > 66, 'Age'] = 6
df_train['Relatives'] = df_train['SibSp'] + df_train['Parch']
arr_train = df_train.to_numpy()
train_survived = np.array(arr_train[:,1], np.int8)
train_tclass = np.array(arr_train[:,2], np.int8)
train_gender = np.array(arr_train[:,4], np.unicode)
train_size = train_gender.size
for i in range(0, train_size):                 # 1 if Female & 0 if Male
    if train_gender[i] == "male":
        train_gender[i] = 0
    elif train_gender[i] == "female":
        train_gender[i] = 1
train_gender = np.array(train_gender, np.int8)
train_age = np.array(arr_train[:,5], np.float64)
train_X = np.ones(train_size, np.int8)
train_relatives = np.array(arr_train[:,12], np.int32)

# Model

def check(x):
    if(x >= 0.5):
        return 1
    else:
        return 0

def sigmoid(t):
    return (1/(1 + math.exp(-t)))


theta = [0, 0, 0, 0, 0]
alpha = 0.01
m = train_gender.size
for j in range(0,30000):
    feature = [0, 0, 0, 0, 0]
    for i in range(0, train_size):
        t = train_X[i]*theta[0] + train_gender[i]*theta[1] + train_tclass[i]*theta[2] + train_age[i]*theta[3] + train_relatives[i]*theta[4]
        t = sigmoid(t)
        t = (t - train_survived[i])
        feature[0] = feature[0] + t*train_X[i]
        feature[1] = feature[1] + t*train_gender[i]
        feature[2] = feature[2] + t*train_tclass[i]
        feature[3] = feature[3] + t*train_age[i]
        feature[4] = feature[4] + t*train_relatives[i]
    
    feature[0] = (feature[0] * alpha)/m
    theta[0] = theta[0] - feature[0]
    print(theta[0])
    feature[1] = (feature[1] * alpha)/m
    theta[1] = theta[1] - feature[1]
    print(theta[1])
    feature[2] = (feature[2] * alpha)/m
    theta[2] = theta[2] - feature[2]
    print(theta[2])
    feature[3] = (feature[3] * alpha)/m
    theta[3] = theta[3] - feature[3]
    print(theta[3])
    feature[4] = (feature[4] * alpha)/m
    theta[4] = theta[4] - feature[4]
    print(theta[4])

# predictions
  
df_test = pd.read_csv('test.csv')  
df_test['Age'].replace(np.NaN, ageMean, inplace = True) # Replacing Null values with Mean of Ages    
df_test.loc[ df_test['Age'] <= 11, 'Age'] = 0
df_test.loc[(df_test['Age'] > 11) & (df_test['Age'] <= 18), 'Age'] = 1
df_test.loc[(df_test['Age'] > 18) & (df_test['Age'] <= 22), 'Age'] = 2
df_test.loc[(df_test['Age'] > 22) & (df_test['Age'] <= 27), 'Age'] = 3
df_test.loc[(df_test['Age'] > 27) & (df_test['Age'] <= 33), 'Age'] = 4
df_test.loc[(df_test['Age'] > 33) & (df_test['Age'] <= 40), 'Age'] = 5
df_test.loc[(df_test['Age'] > 40) & (df_test['Age'] <= 66), 'Age'] = 6
df_test.loc[ df_test['Age'] > 66, 'Age'] = 6
df_test['Relatives'] = df_test['SibSp'] + df_test['Parch']
arr_test = df_test.to_numpy()    
test_tclass = np.array(arr_test[:,1], np.int8)
test_gender = np.array(arr_test[:,3], np.unicode)
test_index = np.array(arr_test[:,0], np.int32)
test_size = test_gender.size
for i in range(0, test_size):                 # 1 if Female & 0 if Male
    if test_gender[i] == "male":
        test_gender[i] = 0
    elif test_gender[i] == "female":
        test_gender[i] = 1
test_gender = np.array(test_gender, np.int8)
test_X = np.ones(test_size, np.int8)
test_age = np.array(arr_test[:,4], np.float64)   
test_relatives = np.array(arr_test[:,11], np.int32) 
    
predictions = np.zeros(test_size, np.int8)
for i in range(0, test_size):
    predictions[i] = check(sigmoid(test_X[i]*theta[0] + test_gender[i]*theta[1] + test_tclass[i]*theta[2] + test_age[i]*theta[3] + test_relatives[i]*theta[4]))

df_predictions = pd.DataFrame()
df_predictions['PassengerId'] = test_index
df_predictions['Survived'] = predictions
df_predictions.to_csv('Predictions.csv', index = False)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    