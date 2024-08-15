import jax.numpy as np
from activation import softmax, sigmoid

# ============ Bahdanau-Attention ===============

def attention_score(params, hi, sj):
  W, V = params['W'], params['V']
  alpha = np.dot(V.T, np.tanh(np.dot(W, np.vstack([hi, sj]))))
  return alpha

def attention_align(params, encoder_states, sj):
  alpha = []
  for hi in encoder_states:
    alpha.append(attention_score(params, hi, sj))
  alpha = np.array(alpha)
  alpha = softmax(alpha)
  # alpha = sigmoid(alpha)
  return np.array(alpha)
