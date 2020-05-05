from sys import argv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodel import LinearRegressionModel
from cross_k_validation import CrossValidation
from tabulate import tabulate

#reading Data
data=pd.read_csv("parkinsons_updrs.csv")
pd.set_option('display.width', 10000)
print (data.describe().T)

# Grouping the data
data.test_time = data.test_time.apply(np.abs)
data["day"] = data.test_time.astype(np.int64) # equals to test_time but only with int values
data = data.groupby(["subject#", "day"]).mean()

data_crossValidation = data

# Splitting into training/testing data
data_train = data.loc[data.index.get_level_values('subject#') <= 36,
                     data.columns.difference(["day","age","sex","test_time"])]

data_test = data.loc[data.index.get_level_values('subject#') > 36,
                    data.columns.difference(["day","age","sex","test_time"])]

# Normalize data_train and data_test
data_train_norm = (data_train - data_train.mean()) / data_train.std()
data_test_norm = (data_test - data_train.mean()) / data_train.std()

# weights
np.random.seed(1234)
random_weights = np.random.rand(len(data_train_norm.columns) - 1)

# Auxiliary functions
def _gradient (X, y, weights):
    """
    gradient = -2 * (X^T)*y + 2 * (X^T)*X*w
    """
    return -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(weights)

def _hessian (X):
    """
    hessian = 4 * (X^T) * X
    """
    return 4 * X.T.dot(X)

def _ridge_gradient(X, y, weights, _lambda):
    """
    _ridge_gradient = 2 * X.T.dot(X).dot(weights) - 2 * X.T.ymeas + 2 *_lambda * weights
    """
    return 2 * X.T.dot(X).dot(weights) + 2 * _lambda * weights

# Algorithms

class MinimumSquareError (LinearRegressionModel):
    """
    Minimum Square Error Algorithm
    e(w) = || y - X*w || ^ 2
    """
    def train (self):
        self.weights = np.dot(np.linalg.pinv(self.X_train), self.y_train) # ()[X^T.X]^-1)X^T = linal.pinv(X_Train)
        self.e_train = np.linalg.norm(self.y_train - self.X_train.dot(self.weights))**2
        self.y_hat_train = self.X_train.dot(self.weights)

class GradientAlgorithm (LinearRegressionModel):
    """
    The Gradient Algorithm
    """
    def initializations(self):
        # To store the last weights used
        self.previous_weights = np.ones(len(self.x_cols))
        # learning_coefficient > 0
        self.learning_coefficient = 1.0e-5
        self.e_train = np.linalg.norm(self.X_train.dot(self.weights) - self.y_train)**2
        self.gradient = _gradient(self.X_train, self.y_train, self.weights)
        # Store every value of e_train
        self.e_history = []
        # Number of iterations and Value to stop the cycle
        self.iterations = 1e4
        self.stop_value = 1e-8

    def train (self):
        i = 0
        self.initializations()

        while np.linalg.norm(self.weights-self.previous_weights) > self.stop_value and i < self.iterations:
            i += 1
            self.e_history += [self.e_train]
            self.previous_weights = self.weights
            #Updating the guess
            self.weights = self.weights - self.learning_coefficient * self.gradient
            # Calculating new e_train and consequently the new gradient
            self.e_train = np.linalg.norm(self.X_train.dot(self.weights) - self.y_train)**2
            self.gradient = _gradient(self.X_train, self.y_train, self.weights)
        # end-while

        self.y_hat_train = self.X_train.dot(self.weights)

class SteepestDescentAlgorithm (LinearRegressionModel):
    """
    Steepest Descent Algorithm
    """
    def initializations(self):
        # To store previous weights
        self.previous_weights = np.ones(len(self.x_cols))
        self.y_hat_train = self.X_train.dot(self.previous_weights)
        self.e_train = np.linalg.norm(self.X_train.dot(self.weights) - self.y_train)**2
        self.gradient = _gradient(self.X_train, self.y_train, self.weights)
        self.hessian = _hessian(self.X_train)
        # Store every value of e_train
        self.e_history = []
        # Number of iterations and stop value
        self.iterations = 1e4
        self.stop_value = 1e-8

    def train (self):
        i = 0
        self.initializations()

        while np.linalg.norm(self.weights-self.previous_weights) > self.stop_value and i < self.iterations:
            i += 1
            self.e_history += [self.e_train]
            self.previous_weights = self.weights
            # Calculating new learning_coefficient
            # (|| gradient || ^ 2) / (gradient^T).hessian.gradient
            learning_coefficient = (np.linalg.norm(self.gradient)**2)/(self.gradient.T.dot(self.hessian).dot(self.gradient))
            # Updating guess
            self.weights = self.weights - learning_coefficient * self.gradient
            # Calculating new e_train and consequently the new gradient
            self.e_train = np.linalg.norm(self.y_train - self.X_train.dot(self.weights))**2
            self.gradient = _gradient(self.X_train, self.y_train, self.weights)
        # end-while

        self.y_hat_train = self.X_train.dot(self.weights)


class RidgeRegression (LinearRegressionModel):
    """
    Ridge Regression
    """
    def train(self, _lambda=-0.2):
        # Weights = ((X^T.X +  lambda*I)^-1).(X^t).y_train
        identity = _lambda * np.eye(self.X_train.shape[1])
        self.weights = np.dot(np.linalg.inv(self.X_train.T.dot(self.X_train) + identity), self.X_train.T.dot(self.y_train))
        self.e_train = np.linalg.norm(self.y_train - self.X_train.dot(self.weights))**2
        self.y_hat_train = self.X_train.dot(self.weights)

class PCR(LinearRegressionModel):
    """
    Principal Component Regression
    """
    def train(self, L=0):
        self.X = self.X_train
        # Number of patients
        self.N = self.X_train.shape[0]
        # Number of Features
        self.F = self.X_train.shape[1]
        # Rx = (1/N) * X^T.X -> F x F
        self.Rx = (1.0/ self.N) * (self.X.T.dot(self.X))
        # Eigenvectors and Eigenvalues
        self.eig_values, self.U = np.linalg.eig(self.Rx)
        # Z = X.U -> transfomation Linear of X_train
        self.Z = self.X.dot(self.U)
        # The diagonal matrix with the largest L eigenvalues
        self.diagonal =np.diag(self.eig_values)[0:L-1,0:L-1]
        # Updating size of U
        self.U = self.U[:,:L-1]
        # weights = (1/N)*(U.(A^-1).U.T). X.T.y
        self.weights = (1.0/self.N) * (self.U.dot(np.linalg.inv(self.diagonal).dot(self.U.T))).dot(self.X.T.dot(self.y_train))
        # E_train and Y_hat_train
        self.e_train = np.linalg.norm(self.y_train - self.X_train.dot(self.weights))**2
        self.y_hat_train = self.X_train.dot(self.weights)

def start (y_col):

    x_cols = data_train_norm.columns.difference([y_col])
    # train data
    X_train = data_train_norm[x_cols].values
    y_train = data_train_norm[y_col].values
    # test data
    X_test = data_test_norm[x_cols].values
    y_test = data_test_norm[y_col].values

    mse = MinimumSquareError(X_train, y_train, X_test, y_test, y_col, x_cols)
    mse.train()
    mse.test()

    gd = GradientAlgorithm(X_train, y_train, X_test, y_test, y_col, x_cols, random_weights)
    gd.train()
    gd.test()

    sda = SteepestDescentAlgorithm(X_train, y_train, X_test, y_test, y_col, x_cols, random_weights)
    sda.train()
    sda.test()

    ridge = RidgeRegression(X_train, y_train, X_test, y_test, y_col, x_cols)
    ridge.train()
    ridge.test()

    pcr =  PCR(X_train, y_train, X_test, y_test, y_col, x_cols)
    pcr.train()
    pcr.test()

    print (mse.e_train,mse.e_test,gd.e_train,gd.e_test,sda.e_train,sda.e_test,ridge.e_train,ridge.e_test,pcr.e_train,pcr.e_test)

    return mse, gd, sda, ridge, pcr

mse, gd, sda, ridge, pcr = start("total_UPDRS")
#mse, gd, sda, ridge, pcr = start("Jitter(%)")

# Plots

#y_hat_train versus y_train

def y_hat_VS_y ():
    plt.figure(figsize=(13,6))
    plt.plot(mse.y_train, mse.y_train, linewidth=0.2)
    plt.scatter(mse.y_hat_train, mse.y_train, label="Minimum square error", marker="o", color="red", alpha=0.4)
    plt.scatter(gd.y_hat_train, gd.y_train, label="Gradient descent", marker="x", color="green", alpha=0.4)
    plt.scatter(sda.y_hat_train, sda.y_train, label="Steepest descent", marker=".", color="blue", alpha=0.4)
    plt.scatter(ridge.y_hat_train, ridge.y_train, label="Ridge regression", marker="*", color="grey", alpha=0.4)
    plt.scatter(pcr.y_hat_train, pcr.y_train, label="Principal component regression", marker="+", color="brown", alpha=0.4)
    plt.title("yhat train versus y train")
    plt.xlabel("y_train")
    plt.ylabel("yhat_train")
    plt.legend(loc=2)
    plt.show()

    plt.figure(figsize=(13,6))
    plt.plot(mse.y_test, mse.y_test, linewidth=0.2)
    plt.scatter(mse.y_hat_test, mse.y_test, label="Minimum square error", marker="o", color="red", alpha=0.4)
    plt.scatter(gd.y_hat_test, gd.y_test, label="Gradient descent", marker="x", color="green", alpha=0.4)
    plt.scatter(sda.y_hat_test, sda.y_test, label="Steepest descent", marker=".", color="blue", alpha=0.4)
    plt.scatter(ridge.y_hat_test, ridge.y_test, label="Ridge regression", marker="*", color="grey", alpha=0.4)
    plt.scatter(pcr.y_hat_test, pcr.y_test, label="Principal component regression", marker="+", color="brown", alpha=0.4)
    plt.title("yhat test versus y test")
    plt.xlabel("y_test")
    plt.ylabel("yhat_test")
    plt.legend(loc=2)
    plt.show()

y_hat_VS_y()


def histogram50_errors ():

    plt.figure(figsize=(13,6))
    plt.hist(mse.y_hat_train - mse.y_train, bins=50, label="Minimum square error", alpha=0.4)
    plt.hist(gd.y_hat_train - gd.y_train, bins=50, label="Gradient descent", alpha=0.4)
    plt.hist(sda.y_hat_train - sda.y_train, bins=50, label="Steepest descent", alpha=0.4)
    plt.hist(ridge.y_hat_train - ridge.y_train, bins=50, label="Ridge regression", alpha=0.4)
    plt.hist(pcr.y_hat_train - pcr.y_train, bins=50, label="Principal component regression", alpha=0.4)
    plt.title("histograms of y train-yhat train")
    plt.xlabel("Error")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    plt.show()


    plt.figure(figsize=(13,6))
    plt.hist(mse.y_hat_test - mse.y_test, bins=50, label="Minimum square error", alpha=0.4)
    plt.hist(gd.y_hat_test - gd.y_test, bins=50, label="Gradient descent", alpha=0.4)
    plt.hist(sda.y_hat_test - sda.y_test, bins=50, label="Steepest descent", alpha=0.4)
    plt.hist(ridge.y_hat_test - ridge.y_test, bins=50, label="Ridge regression", alpha=0.4)
    plt.hist(pcr.y_hat_test - pcr.y_test, bins=50, label="Principal component regression", alpha=0.4)
    plt.title("histograms of y test-yhat test")
    plt.xlabel("Error")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    plt.show()

histogram50_errors()


def weights():
    plt.figure(figsize=(12,6))
    plt.plot(mse.weights, marker=".", label="Minimum square error")
    plt.plot(gd.weights, marker="o", label="Gradient descent")
    plt.plot(sda.weights, marker="x", label="Steepest descent")
    plt.plot(ridge.weights, marker="+", label="Ridge regression")
    plt.plot(pcr.weights, marker="*", label="Principal component regression")
    plt.xticks(range(len(mse.weights)), mse.x_cols, rotation="vertical")
    plt.title("values of w")
    plt.xlabel("Feature")
    plt.ylabel("Weight")
    plt.legend(loc=0)
    plt.show()

weights()

# Cross Validation

def cross_k_validation(y_col):

    cross_k = CrossValidation(data_crossValidation, y_col, 5, 42)

    errors = []

    for i in range(0, cross_k.K):

        mse = MinimumSquareError(
            cross_k.experiences[i][0],
            cross_k.experiences[i][1],
            cross_k.experiences[i][2],
            cross_k.experiences[i][3],
            y_col,
            cross_k.x_cols)
        mse.train()
        mse.test()

        gd = GradientAlgorithm(
             cross_k.experiences[i][0],
             cross_k.experiences[i][1],
             cross_k.experiences[i][2],
             cross_k.experiences[i][3],
             y_col,
             cross_k.x_cols,
             random_weights)
        gd.train()
        gd.test()

        sda = SteepestDescentAlgorithm(
             cross_k.experiences[i][0],
             cross_k.experiences[i][1],
             cross_k.experiences[i][2],
             cross_k.experiences[i][3],
             y_col,
             cross_k.x_cols,
             random_weights)
        sda.train()
        sda.test()

        ridge = RidgeRegression(
              cross_k.experiences[i][0],
              cross_k.experiences[i][1],
              cross_k.experiences[i][2],
              cross_k.experiences[i][3],
              y_col,
              cross_k.x_cols)
        ridge.train()
        ridge.test()

        pcr =  PCR(
              cross_k.experiences[i][0],
              cross_k.experiences[i][1],
              cross_k.experiences[i][2],
              cross_k.experiences[i][3],
              y_col,
              cross_k.x_cols)
        pcr.train()
        pcr.test()

        errors.append([i + 1, mse.e_train, mse.e_test,
                       gd.e_train, gd.e_test,
                       sda.e_train, sda.e_test,
                       ridge.e_train, ridge.e_test,
                       pcr.e_train, pcr.e_test])


    print (tabulate(errors, headers=['Experience',
                           'mse.e_train','mse.e_test',
                           'gd.e_train','gd.e_test',
                           'sda.e_train','sda.e_test',
                           'ridge.e_train','ridge.e_test',
                           'pcr.e_train','pcr.e_test']))

cross_k_validation("Jitter(%)")
cross_k_validation("total_UPDRS")
