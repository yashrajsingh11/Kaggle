import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def accuracy(cm):
    diagonal = cm.trace()
    element = cm.sum()
    return diagonal/element

train_df = pd.read_csv('train.csv')
Y = train_df['label']
train_df.drop('label', inplace = True, axis = 1)
X = train_df.to_numpy()
y = Y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(train_df, Y, test_size = 0.2)

clf = MLPClassifier(solver = 'adam',  activation = 'relu', hidden_layer_sizes = (200,100))
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print(prediction)
acc = confusion_matrix(y_test, prediction)
accuracy = accuracy(acc)
print(accuracy)

test_df = pd.read_csv("test.csv")
arr_test = test_df.to_numpy()
df_predictions = pd.DataFrame()
index = np.zeros(28000, np.int64)
predictions = np.zeros(28000, np.int8)

predictions = clf.predict(arr_test)
for i in range(0,28000):
    index[i] = i + 1
    
df_predictions['ImageId'] = index
df_predictions['Label'] = predictions
df_predictions.to_csv('Predictions.csv', index = False)   