# Using Neural Networks to Recognize
# Handwritten Digits Without Using
# Any External Machine Learning Library

# Importing Libraries

import pandas as pd
import numpy as np

def sigmoid(t):
    return (1/(1 + np.exp(-t)))

def getOutputArray(t):
    if(t == 0):
        return np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]], np.int8)
    elif(t == 1):
        return np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]], np.int8)
    elif(t == 2):
        return np.array([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]], np.int8)
    elif(t == 3):
        return np.array([[0], [0], [0], [1], [0], [0], [0], [0], [0], [0]], np.int8)
    elif(t == 4):
        return np.array([[0], [0], [0], [0], [1], [0], [0], [0], [0], [0]], np.int8)
    elif(t == 5):
        return np.array([[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]], np.int8)
    elif(t == 6):
        return np.array([[0], [0], [0], [0], [0], [0], [1], [0], [0], [0]], np.int8)
    elif(t == 7):
        return np.array([[0], [0], [0], [0], [0], [0], [0], [1], [0], [0]], np.int8)
    elif(t == 8):
        return np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1], [0]], np.int8)
    elif(t == 9):
        return np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [1]], np.int8)

train_df = pd.read_csv("train.csv")

# Model

thetaLayer1 = np.random.rand(200, 784) * 0.01
thetaLayer2 = np.random.rand(100, 200) * 0.01
thetaLayer3 = np.random.rand(10, 100) * 0.01

for k in range(0, 30): 

    train_df = train_df.sample(frac=1).reset_index(drop=True)    
    Y = train_df['label']
    train_df.drop('label', inplace = True, axis = 1)
    arr_train = train_df.to_numpy()

    for j in range(0, 38000):
    
        deltaThetaLayer1 = np.zeros((200, 784), np.float64)     
        deltaThetaLayer2 = np.zeros((100, 200), np.float64)
        deltaThetaLayer3 = np.zeros((10, 100), np.float64)    
    
        deltaLayer2 = np.zeros((200,1), np.float64) 
        deltaLayer3 = np.zeros((100,1), np.float64) 
        deltaLayer4 = np.zeros((10,1), np.float64) 

        activationLayer1 = np.array(arr_train[j,:], np.float64)
        activationLayer1 = activationLayer1/255
        activationLayer1 = activationLayer1.reshape(784,1)

        activationLayer2 = np.zeros((200,1), np.float64) 
        activationLayer2 = np.matmul(thetaLayer1, activationLayer1)
        for i in range(0, 200):
            activationLayer2[i]= sigmoid(activationLayer2[i])            
              
        activationLayer3 = np.zeros((100,1), np.float64)
        activationLayer3 = np.matmul(thetaLayer2, activationLayer2)
        for i in range(0, 100):
            activationLayer3[i] = sigmoid(activationLayer3[i])  
        
        predictedOutput = np.zeros((10,1), np.float64)
        predictedOutput = np.matmul(thetaLayer3, activationLayer3)
        for i in range(0, 10): 
            predictedOutput[i]= sigmoid(predictedOutput[i])
        
        y = np.zeros((10,1), np.int8)
        y = getOutputArray(Y[j])
        
        deltaLayer4 = (predictedOutput - y) 
        deltaLayer3 = ((np.matmul(thetaLayer3.transpose(), deltaLayer4) * activationLayer3) * (1 - activationLayer3))
        deltaLayer2 = ((np.matmul(thetaLayer2.transpose(), deltaLayer3) * activationLayer2) * (1 - activationLayer2))
        
        deltaThetaLayer1 = np.matmul(deltaLayer2, activationLayer1.transpose())
        deltaThetaLayer2 = np.matmul(deltaLayer3, activationLayer2.transpose())
        deltaThetaLayer3 = np.matmul(deltaLayer4, activationLayer3.transpose())
        
        thetaLayer1 = thetaLayer1 - (deltaThetaLayer1 * 0.5)
        thetaLayer2 = thetaLayer2 - (deltaThetaLayer2 * 0.5)
        thetaLayer3 = thetaLayer3 - (deltaThetaLayer3 * 0.5)

    mySum = 0
    total = 4000
    for j in range(38000, 42000):
    
        activationLayer1 = np.array(arr_train[j,:], np.float64)
        activationLayer1 = activationLayer1/255
        activationLayer1 = activationLayer1.reshape(784,1)
    
        activationLayer2 = np.zeros((200,1), np.float64) 
        activationLayer2 = np.matmul(thetaLayer1, activationLayer1)
        for i in range(0, 200):
            activationLayer2[i]= sigmoid(activationLayer2[i])            
                    
        activationLayer3 = np.zeros((100,1), np.float64)
        activationLayer3 = np.matmul(thetaLayer2, activationLayer2)
        for i in range(0, 100):
            activationLayer3[i] = sigmoid(activationLayer3[i])  
            
        predictedOutput = np.zeros((10,1), np.float64)
        predictedOutput = np.matmul(thetaLayer3, activationLayer3)
        for i in range(0, 10): 
            predictedOutput[i]= sigmoid(predictedOutput[i])
       
        if(predictedOutput.argmax() == Y[j]):
            mySum = mySum + 1

    accuracy = mySum/total
    print("cross validation accuracy:")
    print(accuracy)
    
    train_df['label'] = Y

file1 = open("thetaLayer1.txt","w")
for row in thetaLayer1:
    np.savetxt(file1, row)
file1.close()

file2 = open("thetaLayer2.txt","w")
for row in thetaLayer2:
    np.savetxt(file2, row)
file2.close()

file3 = open("thetaLayer3.txt","w")
for row in thetaLayer3:
    np.savetxt(file3, row)
file3.close()

test_df = pd.read_csv("test.csv")
arr_test = test_df.to_numpy()

index = np.zeros(28, np.int64)
predictions = np.zeros(28, np.int8)
df_predictions = pd.DataFrame()

for j in range(0, 28000):
    
        activationLayer1 = np.array(arr_test[j,:], np.float64)
        activationLayer1 = activationLayer1/255
        activationLayer1 = activationLayer1.reshape(784,1)
    
        activationLayer2 = np.zeros((200,1), np.float64) 
        activationLayer2 = np.matmul(thetaLayer1, activationLayer1)
        for i in range(0, 200):
            activationLayer2[i]= sigmoid(activationLayer2[i])            
                    
        activationLayer3 = np.zeros((100,1), np.float64)
        activationLayer3 = np.matmul(thetaLayer2, activationLayer2)
        for i in range(0, 100):
            activationLayer3[i] = sigmoid(activationLayer3[i])  
            
        predictedOutput = np.zeros((10,1), np.float64)
        predictedOutput = np.matmul(thetaLayer3, activationLayer3)
        for i in range(0, 10): 
            predictedOutput[i]= sigmoid(predictedOutput[i])
       
        predictions[j] = predictedOutput.argmax()
        index[j] = j + 1

df_predictions['ImageId'] = index
df_predictions['Label'] = predictions
df_predictions.to_csv('Predictions.csv', index = False)   
