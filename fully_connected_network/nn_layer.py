import numpy as np
import activation
from batch_norm_layer import BatchNormLayer


class NNLayer:
  def __init__(self, input_size, output_size, activation='relu', network_loss=None, keep_prob=1, batch_norm=False):
    self.input_size = input_size
    self.output_size = output_size
    self.activation = activation
    self.__network_loss = network_loss
    self.keep_prob = keep_prob
    self.batch_norm = batch_norm
    self.init_weights()
    if self.batch_norm:
      self.batch_norm_layer = BatchNormLayer(self.output_size)

  def init_weights(self):
    k = 1.
    if self.activation == 'relu':
      k = 2.

    # initialize weights with random values from normal distribution
    self.W = np.random.randn(
      self.output_size, self.input_size) * np.sqrt(k / self.input_size)
    self.b = np.zeros((self.output_size, 1))

    # initialize weights for momentum
    self.vdW = np.zeros_like(self.W)
    self.vdb = np.zeros_like(self.b)

    # initialize weights for Adam
    self.sdW = np.zeros_like(self.W)
    self.sdb = np.zeros_like(self.b)

  def forward(self, A_prev, training=False):
    self.A_prev = A_prev
    self.Z = np.dot(self.W, A_prev) + self.b

    if self.batch_norm:
      self.Z = self.batch_norm_layer.forward(self.Z, training)

    self.A = self.activation_fn(self.Z)

    if training and self.keep_prob < 1:
      # apply dropout to the activations of the previous layer
      self.A = self.A * np.random.binomial(
          1, self.keep_prob, size=self.A.shape)
      # scale the activations
      self.A = self.A / self.keep_prob

    return self.A

  def gradient_descent_update(self, dW, db, learning_rate, L2_reg=0, beta=0.9, beta2=0.999, train_iteration=1, optimization='gd'):
    eps = 1e-8
    if optimization == 'gd':
      self.W -= learning_rate * (dW + L2_reg * self.W)
      self.b -= learning_rate * db
    elif optimization == 'adam':
      self.vdW = beta * self.vdW + (1 - beta) * dW
      self.vdb = beta * self.vdb + (1 - beta) * db
      self.sdW = beta2 * self.sdW + (1 - beta2) * dW ** 2
      self.sdb = beta2 * self.sdb + (1 - beta2) * db ** 2
      # bias correction
      vdW = self.vdW / (1 - beta ** train_iteration)
      vdb = self.vdb / (1 - beta ** train_iteration)
      sdW = self.sdW / (1 - beta2 ** train_iteration)
      sdb = self.sdb / (1 - beta2 ** train_iteration)
      # update weights
      self.W = self.W - learning_rate * vdW / np.sqrt(sdW + eps)
      self.b = self.b - learning_rate * vdb / np.sqrt(sdb + eps)
    elif optimization == 'rmsprop':
      self.vdW = beta * self.vdW + (1 - beta) * dW ** 2
      self.vdb = beta * self.vdb + (1 - beta) * db ** 2
      # update weights
      self.W = self.W - learning_rate * dW / np.sqrt(self.vdW + eps)
      self.b = self.b - learning_rate * db / np.sqrt(self.vdb + eps)
    elif optimization == 'momentum':
      self.vdW = beta * self.vdW + (1 - beta) * dW
      self.vdb = beta * self.vdb + (1 - beta) * db
      self.W -= learning_rate * self.vdW
      self.b -= learning_rate * self.vdb
    else:
      raise ValueError(f"Unsupported optimization method: {optimization}")

  def backward(self, dA, learning_rate, L2_reg=0, beta1=0.9, beta2=0.999, train_iteration=0, optimization='gd'):
    m = self.A_prev.shape[1]

    if self.activation == 'softmax' or self.__network_loss == 'binary_cross_entropy':
      # we already computed the derivative in the nerual network backward pass method
      dZ = dA
    else:
      dZ = dA * self.activation_fn(self.Z, derivative=True)

    if self.batch_norm:
      dZ = self.batch_norm_layer.backward(dZ, learning_rate)

    dW = 1 / m * np.dot(dZ, self.A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(self.W.T, dZ)

    self.gradient_descent_update(
      dW, db, learning_rate,
      L2_reg=L2_reg,
      beta=beta1,
      beta2=beta2,
      train_iteration=train_iteration + 1,
      optimization=optimization
    )

    return dA_prev

  def activation_fn(self, x, derivative=False):
    return activation.activation_fn(self.activation, derivative)(x)


class Layer:
  def __init__(self, neurons: int, activation='relu', keep_prob=1.0, batch_norm=False):
    self.neurons = neurons
    self.activation = activation
    self.keep_prob = keep_prob
    self.batch_norm = batch_norm


class OutputLayer(Layer):
  pass


def connect_layers(layers: list[Layer], loss=None):
  assert len(layers) > 1, "At least 2 layers are required"
  nn_layers = [
    NNLayer(input_size=layers[0].neurons, output_size=layers[1].neurons,
            activation=layers[1].activation, keep_prob=layers[0].keep_prob),
  ]
  for i in range(1, len(layers) - 1):
    nn_layers.append(
      NNLayer(
        input_size=layers[i].neurons,
        output_size=layers[i + 1].neurons,
        activation=layers[i + 1].activation,
        network_loss=loss,
        keep_prob=layers[i].keep_prob,
        batch_norm=layers[i].batch_norm
      )
    )
  return nn_layers
