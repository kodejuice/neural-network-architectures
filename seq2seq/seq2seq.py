import json
import os
from jax import grad
import jax.numpy as np
from numpy import random as npr

from attention import attention_align
from activation import softmax


def init_params(input_size, hidden_size, output_size, attention_size=None, filename='seq2seq_model.json'):
  attention_size = attention_size or hidden_size
  i, h, o, a = input_size, hidden_size, output_size, attention_size
  weights = load_weights(filename, i, h, o)
  if weights:
    print(f"Loaded weights from {filename}")
    return weights

  return {
    'encoder': {
      # encoder states
      'Wx': npr.randn(h, i) * np.sqrt(2. / (h + i)),
      'Wh': npr.randn(h, h) * np.sqrt(2. / (h + h)),
      'bh': np.zeros((h, 1)),
    },

    'decoder': {
      # decoder states
      'Wh': npr.randn(h, o + h + h) * np.sqrt(2. / (o + h + h)),
      'bh': np.zeros((h, 1)),
      'Wy': npr.randn(o, h) * np.sqrt(2. / (o + h)),
      'by': np.zeros((o, 1))
    },

    'attention': {
      # attention
      'W': npr.randn(a, h + h) * np.sqrt(2. / (a + h + h)),
      'V': npr.randn(a, 1),
    }
  }


def mse_loss(y_pred, y_true):
  return np.sum((y_pred - y_true)**2)


def cross_entropy_loss(y_pred, y_true):
  epsilon = 1e-15
  y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
  return -np.sum(y_true * np.log(y_pred))


def rnn_encode_layer(encoder_params, x, h_prev):
  p = encoder_params
  Wx, Wh, bh = p['Wx'], p['Wh'], p['bh']
  h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h_prev) + bh)
  return h


def rnn_decode_layer(params, y_prev, h_prev, context_vector):
  p = params
  Wh, bh, Wy, by = p['Wh'], p['bh'], p['Wy'], p['by']
  v = np.vstack((y_prev, h_prev, context_vector))
  h = np.tanh(Wh @ v + bh)
  y = np.dot(Wy, h) + by
  return y, h


def forward(params, X_sequence):
  encoder_params = params['encoder']
  h = np.zeros((encoder_params['Wh'].shape[0], 1))
  encoder_states = []
  for x in X_sequence:
    h = rnn_encode_layer(encoder_params, x, h)
    encoder_states.append(h)

  Y = []
  encoder_states = np.array(encoder_states)
  attention_params = params['attention']
  decoder_params = params['decoder']
  y = np.zeros((decoder_params['Wy'].shape[0], 1))
  for _ in range(len(X_sequence)):
    h_prev = h
    weights = attention_align(attention_params, encoder_states, h_prev)
    context_vector = np.sum(np.array(weights * encoder_states), axis=0)
    y, h = rnn_decode_layer(decoder_params, y, h, context_vector)
    Y.append(y)

  return np.array(Y)


def predict(params, X_sequence):
  return forward(params, X_sequence)


def sequence_loss(params, X_sequence, Y_sequence, loss_func, apply_softmax=False):
  outputs = forward(params, X_sequence)
  if apply_softmax:
    outputs = [softmax(y) for y in outputs]
  total_loss = sum(loss_func(y_pred, y_true)
                   for y_pred, y_true in zip(outputs, Y_sequence))
  return total_loss / len(X_sequence)


grad_sequence_loss = grad(sequence_loss)


def train(X_sequences, Y_sequences, input_size, hidden_size, epochs, learning_rate=0.01, loss='mse', apply_softmax=False, periodic_callback=None, decay_rate=0.0001, model_filename='seq2seq_model.json', params=None):
  output_size = Y_sequences[0][0].shape[0]
  params = params or init_params(input_size, hidden_size,
                                 output_size, hidden_size, model_filename)

  loss_func = mse_loss if loss == 'mse' else cross_entropy_loss

  for epoch in range(1, epochs + 1):
    LR = learning_rate / (1 + decay_rate * epoch)

    total_loss = 0
    for X_seq, Y_seq in zip(X_sequences, Y_sequences):
      loss = sequence_loss(params, X_seq, Y_seq, loss_func, apply_softmax)
      total_loss += loss

      grads = grad_sequence_loss(
        params, X_seq, Y_seq, loss_func, apply_softmax)

      for model in params:
        for key in params[model]:
          _grad = grads[model][key]
          np.clip(_grad, -5, 5)
          params[model][key] -= LR * _grad

    if epoch % 5 == 0:
      if periodic_callback:
        periodic_callback()

      save_weights(params, model_filename, input_size,
                   hidden_size, output_size)

    print(f"Epoch {epoch}, Loss: {total_loss / len(X_sequences)}")

  return params


def save_weights(params, filename, input_size, hidden_size, output_size):
  metadata = {
      'input_size': input_size,
      'hidden_size': hidden_size,
      'output_size': output_size
  }
  weights = {k: {k2: v2.tolist() for k2, v2 in v.items()}
             for k, v in params.items()}
  data = {'metadata': metadata, 'weights': weights}
  filename = filename.replace('.json', '') + '.json'
  with open(filename, 'w') as f:
    json.dump(data, f, indent=1)
  print(f"Saved weights to {filename}")


def load_weights(filename, input_size, hidden_size, output_size):
  filename = filename.replace('.json', '') + '.json'
  if not os.path.exists(filename):
    return False
  with open(filename, 'r') as f:
    data = json.load(f)

  saved_metadata = data['metadata']
  if (saved_metadata['input_size'] != input_size or
      saved_metadata['hidden_size'] != hidden_size or
          saved_metadata['output_size'] != output_size):
    print("Saved weights dimensions do not match current model dimensions, ignoring")
    f = filename.replace('.json', '')
    os.rename(filename, f + '_old.json')
    return False

  params = {k: {k2: np.array(v2) for k2, v2 in v.items()}
            for k, v in data['weights'].items()}
  return params


class Seq2Seq:
  def __init__(self, input_size, hidden_size, output_size, model_filename, loss='mse', apply_softmax=False):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.model_filename = model_filename
    self.apply_softmax = apply_softmax
    self.loss = loss
    self.params = self.initialize_parameters()

  def sequence_loss(self, X_sequence, Y_sequence):
    loss = mse_loss if self.loss == 'mse' else cross_entropy_loss
    return sequence_loss(self.params, X_sequence, Y_sequence, loss, self.apply_softmax)

  def initialize_parameters(self):
    return init_params(self.input_size, self.hidden_size, self.output_size, self.hidden_size, self.model_filename)

  def train(self, X_sequences, Y_sequences, learning_rate=0.01, decay_rate=0.0001, epochs=100, periodic_callback=None):
    self.params = train(
      X_sequences,
      Y_sequences,
      self.input_size,
      self.hidden_size,
      epochs=epochs,
      learning_rate=learning_rate,
      model_filename=self.model_filename,
      apply_softmax=self.apply_softmax,
      decay_rate=decay_rate,
      loss=self.loss,
      periodic_callback=periodic_callback,
      params=self.params,
    )
    save_weights(self.params, self.model_filename,
                 self.input_size, self.hidden_size, self.output_size)

  def predict(self, X_sequence):
    return predict(self.params, X_sequence)
