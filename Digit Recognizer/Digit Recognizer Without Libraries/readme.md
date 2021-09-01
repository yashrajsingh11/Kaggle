# Digit Recognizer

### Libraries used - 

1. Pandas (For Data Handling, Pre-processing and Model Training)
2. Numpy (For Data Handling, Pre-processing and Model Training)

Note- I didnt use any external Machine Learning Libraries instead I tried to create the model completely by myself.  

### My Approach - 

1. Data-Preprocessing -

The dataset provided on the [Kaggle Website](https://www.kaggle.com/c/digit-recognizer/data) was quite simple. It had 784 columns containing the value of each pixel. It didnt require any pre-processing apart from normalizing the values to lie in between 0 and 1.

2. Model -

I created my own neural network with 4 layers. The input layer with 784 input nodes corresponding to each pixel. Two hidden layers with 200 and 100 nodes each and the output layer with 10 output nodes which predicted the digit which the model recognized.

3. Predictions -

The highest accuracy score I reached using this model was 0.89 (which was quite good considering I didnt use any external libraries or help to train or create the model).

