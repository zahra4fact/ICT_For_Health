import numpy as np

class LinearRegressionModel(object):

    def __init__(self, X_train, y_train, X_test, y_test, y_col, x_cols, weights=[]):

        self.y_col = y_col # represents the column that we want to predict
        self.x_cols = x_cols

        # train data
        self.X_train = X_train
        self.y_train = y_train

        # test data
        self.X_test = X_test
        self.y_test = y_test

        self.e_train = 0.0
        self.e_test = 0.0

        self.y_hat_train = 0.0
        self.y_hat_test = 0.0

        self.weights = weights

    def train(self):
        pass

    def test (self):
      self.y_hat_test = self.X_test.dot(self.weights)
      self.e_test = np.linalg.norm(self.y_hat_test - self.y_test)**2
