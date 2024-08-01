import numpy as np


class BatchNormLayer:
  def __init__(self, size, epsilon=1e-5, momentum=0.9):
    self.epsilon = epsilon
    self.momentum = momentum
    self.size = size
    self.gamma = np.ones((size, 1))
    self.beta = np.zeros((size, 1))
    self.running_mean = np.zeros((size, 1))
    self.running_var = np.ones((size, 1))

  def forward(self, Z, training=True):
    if training:
      self.Z = Z
      self.mu = np.mean(Z, axis=1, keepdims=True)
      self.var = np.var(Z, axis=1, keepdims=True)
      self.Z_norm = (Z - self.mu) / np.sqrt(self.var + self.epsilon)
      self.Z_out = self.gamma * self.Z_norm + self.beta

      # Update running mean and variance
      self.running_mean = self.momentum * \
          self.running_mean + (1 - self.momentum) * self.mu
      self.running_var = self.momentum * \
          self.running_var + (1 - self.momentum) * self.var
    else:
      Z_norm = (Z - self.running_mean) / \
          np.sqrt(self.running_var + self.epsilon)
      self.Z_out = self.gamma * Z_norm + self.beta

    # print(f"BatchNorm: {self.Z_out}")
    return self.Z_out

  def backward(self, dZ, learning_rate):
    m = dZ.shape[1]

    dgamma = np.sum(dZ * self.Z_norm, axis=1, keepdims=True)
    dbeta = np.sum(dZ, axis=1, keepdims=True)

    dZ_norm = dZ * self.gamma
    dvar = np.sum(dZ_norm * (self.Z - self.mu) * -0.5 *
                  (self.var + self.epsilon) ** (-1.5), axis=1, keepdims=True)
    dmu = np.sum(dZ_norm * -1 / np.sqrt(self.var + self.epsilon), axis=1,
                 keepdims=True) + dvar * np.mean(-2 * (self.Z - self.mu), axis=1, keepdims=True)
    dZ = dZ_norm / np.sqrt(self.var + self.epsilon) + \
        dvar * 2 * (self.Z - self.mu) / m + dmu / m

    # Update gamma and beta
    self.gamma -= learning_rate * dgamma
    self.beta -= learning_rate * dbeta

    return dZ
