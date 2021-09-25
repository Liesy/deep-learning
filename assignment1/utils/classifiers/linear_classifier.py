import numpy as np
from utils.classifiers.linear_svm import *
from utils.classifiers.softmax import *


class LinearClassifier(object):

    def __init__(self):
        self.W = None


    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=100, verbose=False):
        '''
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        '''
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # y的取值范围为0-(k-1),则共有k个类
        
        # initialize the W
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for iter in range(num_iters):
            batch_idx = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad # sgd

            if verbose and iter % 10 == 0:
                print("iteration %d/%d : loss = %f" % (iter, num_iters, loss))

        return loss_history


    def predict(self, X):
        """
        Inputs:
        - X: A numpy array of shape (N, D) containing training data
          there are N training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])

        scores = np.dot(X, self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred


    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
