import numpy as np


def sigmoid(x):
  x = np.clip(x, -709, 709)  # Clip input to avoid overflow
  return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
  return sigmoid(x) * (1 - sigmoid(x))


def ReLu(v):
  # clip input to avoid overflow
  # v = np.clip(v, 1e-15, 1 - 1e-1 5)
  return np.maximum(0, v)


def d_ReLu(x):
  return np.where(x > 0, 1, 0)


def leaky_ReLu(x, alpha=0.01):
  v = np.maximum(alpha * x, x)
  v = np.clip(v, 1e-15, 1 - 1e-15)
  return v


def d_leaky_ReLu(x, alpha=0.01):
  return np.where(x > 0, 1, alpha)


def tanh(x):
  return np.tanh(x)


def d_tanh(x):
  return 1 - np.tanh(x) ** 2


def linear(x):
  x = np.clip(x, 1e-15, 1 - 1e-15)
  return x


def d_linear(x):
  return 1


def softmax(x, T=1, col=False):
  axis = 1 if col else 0
  clip_value = 10.0
  x = x - x.max(axis=axis)
  x = np.clip(x, -clip_value, clip_value)
  exp_xrel = np.exp(x / T)
  return exp_xrel / exp_xrel.sum(axis=axis)


def d_softmax(x):
  s = softmax(x)
  return s * (1 - s)


def activation_fn(activation: str, derivative=False):
  a = {
      'relu': [ReLu, d_ReLu],
      'leaky_relu': [leaky_ReLu, d_leaky_ReLu],
      'tanh': [tanh, d_tanh],
      'sigmoid': [sigmoid, d_sigmoid],
      'linear': [linear, d_linear],
      'softmax': [softmax, d_softmax],
  }
  return a[activation][derivative]
