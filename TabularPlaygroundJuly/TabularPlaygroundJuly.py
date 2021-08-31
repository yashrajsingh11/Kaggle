# Importing Libraries

import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
    
# Importing Data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Data Analysis and Preprocessing

#plt.subplot(3, 3, 1)
#sns.distplot(train.target_carbon_monoxide)
#plt.subplot(3, 3, 2)
#sns.distplot(train.target_benzene)
#plt.subplot(3, 3, 3)
#sns.distplot(train.target_nitrogen_oxides)

#sns.heatmap(train.corr())

train['deg_C'] = (train['deg_C'] - train['deg_C'].mean()) / (train['deg_C'].max() - train['deg_C'].min())
train['relative_humidity'] = (train['relative_humidity'] - train['relative_humidity'].mean()) / (train['relative_humidity'].max() - train['relative_humidity'].min())
train['absolute_humidity'] = (train['absolute_humidity'] - train['absolute_humidity'].mean()) / (train['absolute_humidity'].max() - train['absolute_humidity'].min())

train['sensor_1'] = (train['sensor_1'] - train['sensor_1'].mean()) / (train['sensor_1'].max() - train['sensor_1'].min())
train['sensor_2'] = (train['sensor_2'] - train['sensor_2'].mean()) / (train['sensor_2'].max() - train['sensor_2'].min())
train['sensor_3'] = (train['sensor_3'] - train['sensor_3'].mean()) / (train['sensor_3'].max() - train['sensor_3'].min())
train['sensor_4'] = (train['sensor_4'] - train['sensor_4'].mean()) / (train['sensor_4'].max() - train['sensor_4'].min())
train['sensor_5'] = (train['sensor_5'] - train['sensor_5'].mean()) / (train['sensor_5'].max() - train['sensor_5'].min())

train['target_carbon_monoxide_normalized'] = (train['target_carbon_monoxide'] - train['target_carbon_monoxide'].mean()) / (train['target_carbon_monoxide'].max() - train['target_carbon_monoxide'].min())
train['target_benzene_normalized'] = (train['target_benzene'] - train['target_benzene'].mean()) / (train['target_benzene'].max() - train['target_benzene'].min())
no2_Max = train['target_nitrogen_oxides'].max()
no2_Min = train['target_nitrogen_oxides'].min()
no2_Mean = train['target_nitrogen_oxides'].mean()
train['target_nitrogen_oxides_normalized'] = (train['target_nitrogen_oxides'] - no2_Mean) / (no2_Max - no2_Min)

test['deg_C'] = (test['deg_C'] - train['deg_C'].mean()) / (train['deg_C'].max() - train['deg_C'].min())
test['relative_humidity'] = (test['relative_humidity'] - train['relative_humidity'].mean()) / (train['relative_humidity'].max() - train['relative_humidity'].min())
test['absolute_humidity'] = (test['absolute_humidity'] - train['absolute_humidity'].mean()) / (train['absolute_humidity'].max() - train['absolute_humidity'].min())

test['sensor_1'] = (test['sensor_1'] - train['sensor_1'].mean()) / (train['sensor_1'].max() - train['sensor_1'].min())
test['sensor_2'] = (test['sensor_2'] - train['sensor_2'].mean()) / (train['sensor_2'].max() - train['sensor_2'].min())
test['sensor_3'] = (test['sensor_3'] - train['sensor_3'].mean()) / (train['sensor_3'].max() - train['sensor_3'].min())
test['sensor_4'] = (test['sensor_4'] - train['sensor_4'].mean()) / (train['sensor_4'].max() - train['sensor_4'].min())
test['sensor_5'] = (test['sensor_5'] - train['sensor_5'].mean()) / (train['sensor_5'].max() - train['sensor_5'].min())

y1_tester = train['target_carbon_monoxide']
y2_tester = train['target_benzene']
y3_tester = train['target_nitrogen_oxides']
y1_trainer = train['target_carbon_monoxide_normalized']
y2_trainer = train['target_benzene_normalized']
y3_trainer = train['target_nitrogen_oxides_normalized']

train.drop('date_time', inplace = True, axis = 1)
train.drop('target_carbon_monoxide', inplace = True, axis = 1)
train.drop('target_benzene', inplace = True, axis = 1)
train.drop('target_nitrogen_oxides', inplace = True, axis = 1)
train.drop('target_carbon_monoxide_normalized', inplace = True, axis = 1)
train.drop('target_benzene_normalized', inplace = True, axis = 1)
train.drop('target_nitrogen_oxides_normalized', inplace = True, axis = 1)

# Model and predictions

df_predictions = pd.DataFrame()
df_predictions['date_time'] = test['date_time']
test.drop('date_time', inplace = True, axis = 1)

X_train, X_test, y01_train, y01_test, yn1_train, yn1_test, y02_train, y02_test, yn2_train, yn2_test, y03_train, y03_test, yn3_train, yn3_test = train_test_split(train, y1_tester, y1_trainer, y2_tester, y2_trainer, y3_tester, y3_trainer, test_size = 0.2)

#n_estimators = [100, 500, 700]
#max_features = ['auto', 'sqrt']
#max_depth = [10, 50, 70]
#max_depth.append(None)
#min_samples_split = [2, 5, 8]
#min_samples_leaf = [1, 2, 5]

#random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
               
#regressor = RandomForestRegressor(random_state = 42)
#rf_random = GridSearchCV(estimator = regressor, param_grid = random_grid, cv = 3, verbose = 2, n_jobs = -1)
#rf_random.fit(train, y3_trainer)

#print(rf_random.best_params_)
#best_random = rf_random.best_estimator_

#prediction = best_random.predict(train)
#error = metrics.mean_squared_error(y3_trainer, prediction, squared = False)
#print(error)

#prediction = (prediction * (no2_Max - no2_Min)) + no2_Mean
#error = metrics.mean_squared_error(y3_tester, prediction, squared = False)
#print(error)

regressornnwo = RandomForestRegressor(n_estimators = 100, random_state = 42)
regressornnwo.fit(X_train, yn3_train)

predictions = regressornnwo.predict(X_test)
predictions = (predictions * (no2_Max - no2_Min)) + no2_Mean
error = metrics.mean_squared_error(y03_test, predictions, squared = False)
print(error)

regressornno = RandomForestRegressor(n_estimators = 700, random_state = 42, max_features = 'sqrt', max_depth = 50, min_samples_leaf = 5, min_samples_split = 2) 
regressornno.fit(X_train, yn3_train)

predictions = regressornno.predict(X_test)
predictions = (predictions * (no2_Max - no2_Min)) + no2_Mean
error = metrics.mean_squared_error(y03_test, predictions, squared = False)
print(error)

##regressor01 = RandomForestRegressor(n_estimators = 100, random_state = 42, bootstrap = True, max_depth = 40, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 2, n_jobs = -1)
##regressor01.fit(X_train, y01_train)

##prediction01 = regressor01.predict(X_test)
##error = metrics.mean_squared_error(y01_test, prediction01, squared = False)
##print(error)

##regressor1 = RandomForestRegressor(n_estimators = 100, random_state = 42, bootstrap = True, max_depth = 40, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 2, n_jobs = -1)
##regressor1.fit(train, y1_test)

##prediction1 = regressor1.predict(train)
##error = metrics.mean_squared_error(y1_test, prediction1, squared = False)
##print(error)

##predictions1 = regressor1.predict(test)
##df_predictions['target_carbon_monoxide'] = predictions1

##regressor02 = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs = -1, min_samples_leaf = 1, min_samples_split = 2, max_depth = None, max_features = 'auto', bootstrap = True)
##regressor02.fit(X_train, y02_train)

##prediction02 = regressor02.predict(X_test)
##error = metrics.mean_squared_error(y02_test, prediction02, squared = False)
##print(error)

##regressor2 = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs = -1, min_samples_leaf = 1, min_samples_split = 2, max_depth = None, max_features = 'auto', bootstrap = True)
##regressor2.fit(train, y2_test)

##prediction2 = regressor2.predict(train)
##error = metrics.mean_squared_error(y2_test, prediction2, squared = False)
##print(error)

##predictions2 = regressor2.predict(test)
##df_predictions['target_benzene'] = predictions2








