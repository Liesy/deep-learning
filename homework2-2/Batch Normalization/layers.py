import numpy as np
# reference: https://blog.csdn.net/stesha_chen/article/details/84999511

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0) # (D,)
        sample_var = np.var(x, axis=0) # (D,)
        x_hat = (x - sample_mean) / (np.sqrt(sample_var + eps))
        out = gamma * x_hat + beta
        cache = (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

    elif mode == 'test':
        out = (x - running_mean) * gamma / (np.sqrt(running_var + eps)) + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """

    x, mean, var, x_hat, eps, gamma, beta = cache
    N = x.shape[0]

    dgamma = np.sum(dout * x_hat, axis = 0)
    dbeta = np.sum(dout * 1.0, axis = 0)
    
    dx_hat = dout * gamma
    dx_hat_numerator = dx_hat / np.sqrt(var + eps)
    dx_hat_denominator = np.sum(dx_hat * (x - mean), axis = 0)

    dx_1 = dx_hat_numerator
    dvar = -0.5 * ((var + eps) ** (-1.5)) * dx_hat_denominator
    dmean = -1.0 * np.sum(dx_hat_numerator, axis = 0) + dvar * np.mean(-2.0 * (x - mean), axis = 0)

    dx_var = dvar * 2.0 / N * (x - mean)
    dx_mean = dmean * 1.0 / N
    dx = dx_1 + dx_var + dx_mean

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """

    x, mean, var, x_hat, eps, gamma, beta = cache
    N = x.shape[0]

    dbeta = np.sum(dout, axis = 0)
    dgamma = np.sum(x_hat * dout, axis = 0)
    dx_norm = dout * gamma
    dv = np.sum((x - mean) * -0.5 * (var + eps) ** -1.5 * dx_norm, axis = 0)
    dm = np.sum(dx_norm * -1 * (var + eps)** -0.5, axis = 0) + np.sum(dv * (x - mean) * -2 / N, axis = 0)
    dx = dx_norm / (var + eps) ** 0.5 + dv * 2 * (x - mean) / N + dm / N

    return dx, dgamma, dbeta