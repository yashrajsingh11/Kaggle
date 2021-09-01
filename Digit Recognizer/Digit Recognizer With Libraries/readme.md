# Digit Recognizer

### Libraries used - 

1. Pandas (For Data Handling, Pre-processing)
2. Numpy (For Data Handling, Pre-processing)
3. Scikit-Learn (For Model Training)

### My Approach - 

The approach was similar to the one which I used in predicting the output without using any ML Libraries. The only difference was in model training.

1. Model -

For this I used the Scikit Learn's [Multi Layer Perceptron Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)(MLPClassifier) with two hidden layers of 200 nodes and 100 nodes each, along with 784 input node corresponding to each pixel and 10 output nodes corresponding to each digit that can be recignized.

2. Predictions -

The highest accuracy score I reached using this model was 0.953.