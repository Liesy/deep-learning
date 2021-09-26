import numpy as np


class ThreeLayerNet(object):
    """
    A three-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, H)
        b2: Second layer biases; has shape (H,)
        W3: Third layer weights; has shape (H, C)
        b3: Third layer biases; has shape (C,)


        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a three layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = X.shape

        # Compute the forward pass
        layer_1 = np.maximum(0, np.dot(X, W1) + b1)  # (N,D)*(D,H)=(N,H)
        layer_2 = np.maximum(0, np.dot(layer_1, W2) + b2)  # (N,H)*(H,H)=(N,H)
        fcn_3 = np.dot(layer_2, W3) + b3  # (N,H)*(H,C)=(N,C)
        scores = fcn_3
        # softmax
        # (C,N)-[(1,N)->(C,N)]=(C,N)再转置为(N,C)
        exp_scores = np.exp((scores.T - np.max(scores, axis=1)).T)
        exp_scores_sum = np.sum(exp_scores, axis=1)  # (1,N)
        layer_3 = (exp_scores.T * 1.0 / exp_scores_sum).T # prob_allClass

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss. Use the Softmax classifier loss.
        prob_i = layer_3[range(N), y] # prob of correct class
        loss_sum = np.sum(-np.log(prob_i))
        regression = reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
        loss = loss_sum / N + regression

        # Backward pass: compute gradients
        grads = {}
        dW1 = 2 * reg * W1
        dW2 = 2 * reg * W2
        dW3 = 2 * reg * W3

        # grad of softmax https://zhuanlan.zhihu.com/p/25723112
        dsoftmax = layer_3
        dsoftmax[range(N), y] -= 1.0
        dsoftmax /= N

        db3 = np.dot(np.ones(N), dsoftmax)
        dW3 += np.dot(layer_2.T, dsoftmax)

        dlayer_2 = np.dot(dsoftmax, W3.T)
        dReLU2 = dlayer_2
        dReLU2[np.where(dReLU2 <= 0)] = 0.0

        db2 = np.dot(np.ones(N), dReLU2)
        dW2 += np.dot(layer_1.T, dReLU2)

        dlayer_1 = np.dot(dReLU2, W2.T)
        dReLU1 = dlayer_1
        dReLU1[np.where(dReLU1 <= 0)] = 0.0

        db1 = np.dot(np.ones(N), dReLU1)
        dW1 += np.dot(X.T, dReLU1)

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]  # (N,)
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            batch_idx = np.random.choice(num_train, batch_size)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y_batch, reg=reg)
            loss_history.append(loss)

            for name in self.params:
                self.params[name] -= learning_rate * grads[name]  # sgd

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this three-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        scores = self.loss(X) # (N,C)
        y_pred = np.argmax(scores, axis=1) # (1,N)

        return y_pred
