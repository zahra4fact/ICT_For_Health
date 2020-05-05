import numpy as np

class CrossValidation(object):
    """
    K-fold cross validation technique 
    """
    def __init__(self, data, y_col, K, number_patients):

        self.data = data
        self.y_col = y_col
        self.K = K
        self.number_patients = number_patients

        self.patient_id = 1

        self.subsets = []
        self.length_subset = self.number_patients // self.K
        self.rest = self.number_patients % self.K

        self.experiences = []

        self.x_cols = data.columns.difference(["day","age","sex","test_time",y_col])

        # Creating subsets
        for i in range(0, self.K):

            current_subset = []

            for j in range(0, self.length_subset):
                current_subset.append(self.patient_id)
                self.patient_id += 1

            if self.rest != 0:
                current_subset.append(self.patient_id)
                self.patient_id += 1
                self.rest -= 1

            self.subsets.append(current_subset)

        for k in range(0, self.K):

            aux_subsets = self.subsets[:]

            test_subset = aux_subsets[k]
            del aux_subsets[k]
            train_subset = sum(aux_subsets, [])

            data_train = self.data.loc[train_subset, self.data.columns.difference(["day","age","sex","test_time"])].values

            # Normalize train and test data
            y_test = (self.data[y_col].loc[test_subset].values - data_train.mean()) / data_train.std()
            X_test = (self.data.loc[test_subset, self.data.columns.difference(["day","age","sex","test_time",y_col])].values - data_train.mean()) / data_train.std()

            y_train = (self.data[y_col].loc[train_subset] - data_train.mean()) / data_train.std()
            X_train = (self.data.loc[train_subset, self.data.columns.difference(["day","age","sex","test_time",y_col])].values - data_train.mean()) / data_train.std()

            self.experiences.append([X_train,y_train,X_test,y_test])
