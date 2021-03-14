from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

import pandas as pd
import seaborn as sns


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data_df = pd.read_csv('housing.csv', header=None,
                      delim_whitespace=True, names=column_names)


X = data_df[["LSTAT"]]

y = data_df['MEDV']


class Regression:

    def linearRegression(self):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4)
        lr = LinearRegression()
        model1 = lr.fit(X_train, y_train)
        predicted = model1.predict(X_test)
        # print(predicted)
        # print(y_test)
        #plt.scatter(y_test, predicted)
        sns.regplot(y_test, predicted)
        print("RMSE: ", np.sqrt(mean_squared_error(y_test, predicted)))
        print("R**2 score is: ", r2_score(y_test, predicted))
        plt.title("Linear Regression")
        plt.xlabel("MEDV")
        plt.ylabel("PREDICTED")
        plt.show()

    def ploynomailRegression(self, degree):

        # print(data_df.tail())

        df2 = pd.DataFrame()

        df2['LSTAT'], df2['MEDV'] = data_df['LSTAT'], data_df['MEDV']

        X_train, X_test, y_train, y_test = train_test_split(
            df2['MEDV'], df2['LSTAT'])

        X_train_df, X_test_df = pd.DataFrame(X_train), pd.DataFrame(X_test)

        poly = PolynomialFeatures(degree=degree)

        X_train_poly, X_test_poly = poly.fit_transform(
            X_train_df), poly.fit_transform(X_test_df)

        model = linear_model.LinearRegression()

        model = model.fit(X_train_poly, y_train)

        coefficient = model.coef_

        print(coefficient)

        intercept = model.intercept_

        x_axis = np.arange(5, 50, 0.1)

        response = intercept

        for i in range(1, degree+1):
            response = response + coefficient[i] * x_axis**i

        # response = intercept + coefficient[1] * \
        #     x_axis + coefficient[2] * x_axis**2

        plt.xlabel("MEDV")
        plt.ylabel("LSTAT")
        plt.title("POLYNOMIAL REGRESSION")

        prediction = model.predict(X_test_poly)

        print(r2_score(y_test, prediction))
        print(np.sqrt(mean_squared_error(y_test, prediction)))

        #print("R**2 score is:", r2_score)

        plt.scatter(df2['MEDV'], df2['LSTAT'], color='b')

        plt.plot(x_axis, response, color='r')

        plt.show()

    def multipleRegression(self):

        X = data_df[["LSTAT", "RM", "PTRATIO"]]

        y = data_df['MEDV']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4)
        lr = LinearRegression()
        model1 = lr.fit(X_train, y_train)

        predicted = model1.predict(X_test)

        print(predicted)

        print(y_test)

        #plt.scatter(y_test, predicted)

        sns.regplot(y_test, predicted)

        print("RMSE: ", np.sqrt(mean_squared_error(y_test, predicted)))
        print("R**2 score is: ", r2_score(y_test, predicted))
        r_squared = r2_score(y_test, predicted)
        adjusted_r_squared = 1 - \
            (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
        print("adjusted R**2 score is: ", adjusted_r_squared)
        plt.show()

        # print(X_train_poly)
if __name__ == "__main__":

    r = Regression()
    # r.linearRegression()
    #degree = 2
    # r.ploynomailRegression(degree)
    r.multipleRegression()
