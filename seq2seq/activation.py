import jax.numpy as np


def sigmoid(x):
  x = np.clip(x, -709, 709)  # Clip input to avoid overflow
  return 1 / (1 + np.exp(-x))


def softmax(x, T=1, col=False):
  axis = 1 if col else 0
  clip_value = 10.0
  x = x - x.max(axis=axis)
  x = np.clip(x, -clip_value, clip_value)
  exp_xrel = np.exp(x / T)
  return exp_xrel / exp_xrel.sum(axis=axis)
