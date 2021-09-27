import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    Inputs have dimension D, there are C classes, and we operate on minibatches of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = np.dot(X[i], W)
        for j in range(num_classes):
            if j == y[i]: # 注意这里是j==y[i]，不是j==i，这Bug找个半个小时
                continue
            margin = scores[j] - scores[y[i]] + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i] * 1.0
                dW[:, y[i]] -= X[i] * 1.0
    
    # the loss is a sum over all training examples, we want it to be an average instead
    loss /= num_train

    # add l2 regularization to the loss.
    loss += reg * np.sum(W * W)

    dW = dW / num_train + 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # (D, C)
    num_train = X.shape[0]

    # compute the loss
    scores = np.dot(X, W) # (N, C)
    margin = np.maximum(0, scores - scores[range(num_train), y].reshape(-1, 1) + 1) # 逐位比较取其大者,会用到broadcast机制
    margin[range(num_train), y] = 0 # let i != j
    data_loss = np.sum(margin) * 1.0 / num_train
    reg_loss = reg * np.sum(W * W)
    loss = data_loss + reg_loss

    # compute the gradient
    X_effect = (margin > 0).astype('float') # 每个样本i在非y[i]的类上产生X[i]的梯度
    X_effect[range(num_train), y] -= np.sum(X_effect, axis=1)
    dW = np.dot(X.T, X_effect)
    dW = dW / num_train + 2 * reg * W

    return loss, dW