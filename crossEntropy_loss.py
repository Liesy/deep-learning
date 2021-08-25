import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
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
        scores = np.dot(X, W) # (1, C)
        exp_normalized_scores = np.exp(scores - np.max(scores))
        loss -= np.log(exp_normalized_scores[y[i]] * 1.0 / np.sum(exp_normalized_scores))

        for j in range(num_classes):
            dW[:, j] += (exp_normalized_scores[j] * 1.0 / np.sum(exp_normalized_scores)) * X[i]
            if j == y[i]:
                dW[:, j] -= X[i]

    loss = loss / num_train + reg * np.sum(np.square(W))
    dW = dW / num_train + 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    num_train = X.shape[0]

    scores = np.dot(X, W) # (N, C)
    exp_normalized_scores = np.exp((scores.T - np.max(scores, axis=1)).T)

    # compute the loss
    loss = np.sum(-np.log(exp_normalized_scores[range(num_train), y] * 1.0 / np.sum(exp_normalized_scores, axis=1)))
    loss = loss / num_train + reg + np.sum(np.square(W))

    # compute the gradient
    acc_effect = (exp_normalized_scores.T / np.sum(exp_normalized_scores, axis=1)).T
    acc_effect[range(num_train), y] -= 1.0
    dW = np.dot(X.T, acc_effect) / num_train + 2 * reg * W

    return loss, dW