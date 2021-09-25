import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """


    def __init__(self):
        pass


    def train(self, X, y):
        """
        Train the classifier.
        For k-nearest neighbors this is just memorizing the training data.
        
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
             consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        
        self.train_X = X
        self.train_y = y


    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.
        
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        # https://mlxai.github.io/2017/01/03/finding-distances-between-data-points-with-numpy.html
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loops(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)


    def compute_distances_no_loops(self, X):
        '''Compute the l2 distance between all test points and all training points'''
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        M = np.dot(X, self.train_X.T) # M = X.dot(self.train_X.T) .T为转置
        X_square = np.square(X).sum(axis=1)
        train_X_square = np.square(self.train_X).sum(axis=1)

        dists = np.sqrt(np.matrix(X_square).T + np.matrix(train_X_square) - 2 * M)

        return dists


    def compute_distances_one_loops(self, X):
        '''Compute the l2 distance between the i_th test point and all training points'''
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum(np.square(X[i, :] - self.train_X), axis=1))

        '''
        np.sum(arr,axis)
        当axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行
        当axis为1时,是压缩列,即将每一行的元素相加,将矩阵压缩为一列,以行向量的形式表示
        '''

        return dists


    def compute_distances_two_loops(self, X):
        '''Compute the l2 distance between the i_th test point and the j_th training point'''
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum(np.square(X[i, :] - self.train_X[j, :])))

        return dists


    def predict_labels(self, dists, k=1):
        '''
        Use the distance matrix to find the k nearest neighbors of the i_th testing point
        Use self.y_train to find the labels of these neighbors. 
        Store these labels in closest_y. 
        '''
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            closest_y = []
            '''argsort函数返回的是数组值从小到大的索引值'''
            closest_y = self.y_train[np.argsort(dists[i])][0:k]
            '''max返回closest_y.count(value)中最大的，即大多数投票'''
            y_pred[i] = max(closest_y, key=list(closest_y).count)

        return y_pred