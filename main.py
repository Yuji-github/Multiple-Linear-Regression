import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer # for encoding categorical to numerical
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split # for splitting data into trains and tests
from sklearn.linear_model import LinearRegression # for training and predicting

def MLR():
    # import data
    data = pd.read_csv('50_Startups.csv')
    print(data.head(3))

    # independent variables (x)
    x = data.iloc[:, :-1].values

    # dependent (results) variable (y)
    y = data.iloc[:, -1].values
    print(x)
    print(y)

    # encoding categorical data
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    x = np.array(ct.fit_transform(x))

    # states moves to index 0
    # NY 0.0.1
    # CA 1.0.0
    # Fl 0.1.0
    print(x)

    # splitting data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

    # Multiple Linear Regression does NOT need feature scaling
    # because each coffcient corresponds each feature

    # Training Multiple Linear Regression on Train
    regressor = LinearRegression() # handle dummy traps and select best models like backward elimination automatically
    regressor.fit(x_train, y_train)

    # Predicting test results (profits) by comparing actual profits
    # because multiple linear regression do not need to get plotted

    y_pred = regressor.predict(x_test) # as the same as simple linear
    np.set_printoptions(precision=2) # exact by 2 decimal numbers
    print('Prediction vs Real Profits')
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1)) # axis 1 is horizontal
    # np.concatenate((array1, array2, arrayN), axis, and so on) axis = 0 is vertical
    # this time is just comparing prediction(ML) vs actual profits


    print('Predict: CA, R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000')
    print('The profits will be %d dollars per year' %regressor.predict([[1,0,0, 16000, 13000, 3000000]]))




if __name__ == '__main__':
    MLR()