# Titanic - Machine Learning From Disaster Without ML Libraries

### Prediction Model -

Logistic Regression: This was my first-ever Machine Learning Model so I tried to keep it as simple as possible. Hence I used this model.

### Libraries used - 

1. Pandas (For Data Handling and Pre-processing)
2. Numpy (For Data Handling and Pre-processing)
3. Matplotlib (For Data Analysis)

Note- I didnt use any external Machine Learning Libraries instead I tried to create the model completely by myself.  

### My Approach - 

1. Identifying Features -
 
Looking at the training data provided on the [Kaggle Website](https://www.kaggle.com/c/titanic/data). It was clear that sex, age, ticket-class were the main factors affecting the survival rate of a person. I plotted some graphs to confirm this. After my first submission, I tried to add some more features and with the help of [the Internet](https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8), I replaced age with age-groups and added relatives which is 'SibSp' + 'parch' which increased the accuracy score a bit.

2. Data-Preprocessing - 

The Age features had some missing values which I replaced with the mean of all ages.
The sex features was converted from 'female' to 1 and from 'male' to 0.

3. Predictions -

My first submission was given an accuracy score of 0.75358 (which I consider quite good since it was the first ever ML problem and also I was using my own logistic regression model for training insted of some optimized training model from ML Libraries) . The second submission with improved features was given an accuracy score of 0.76794.

