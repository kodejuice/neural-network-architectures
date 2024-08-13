import os
import json
import numpy as np

"""
Sequence to Sequence architecture

Assumes that the input and output are of the same length
"""

# TODO: remove
try:
  from .. import activation
except:
  import activation


def mse_loss(y_pred, y_true):
  return np.sum((y_pred - y_true)**2)


def mse_loss_derivative(y_pred, y_true):
  return 2 * (y_pred - y_true)


def cross_entropy_loss(y_pred, y_true):
  epsilon = 1e-15
  y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
  return -np.sum(y_true * np.log(y_pred))


def cross_entropy_loss_derivative(y_pred, y_true):
  epsilon = 1e-15
  y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
  return (y_pred - y_true)  # assuming softmax activation


def loss_func(loss, deriv=False):
  L = {
    'mse': [mse_loss, mse_loss_derivative],
    'cross_entropy': [cross_entropy_loss, cross_entropy_loss_derivative],
  }
  return L[loss][deriv]


class EncoderRNNLayer:
  def __init__(self, input_size, hidden_size, activation='tanh', optimization='gd') -> None:
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.activation = activation
    self.optimization = optimization
    self.initialize_weights(input_size, hidden_size)

  def initialize_weights(self, input_size, hidden_size):
    # Xavier initialization
    self.Wx = np.random.randn(hidden_size, input_size) * \
        np.sqrt(2. / (hidden_size + input_size))
    self.Wh = np.random.randn(hidden_size, hidden_size) * \
        np.sqrt(2. / (hidden_size + hidden_size))
    self.bh = np.zeros((hidden_size, 1))
    self.reset_hidden_state()

  def reset_hidden_state(self):
    self.hidden_states = [np.zeros((self.hidden_size, 1))]  # initial state

  def activation_func(self, x):
    return activation.activation_fn(self.activation)(x)

  def activation_derivative(self, x):
    return activation.activation_fn(self.activation, derivative=True)(x)

  def forward(self, x):
    if not isinstance(x, np.ndarray):
      x = np.array(x)

    x = x.reshape(-1, 1)  # make a column vector
    h_prev = self.hidden_states[-1]

    h = self.activation_func(self.Wx @ x + self.Wh @ h_prev + self.bh)

    self.hidden_states.append(h)

    return h

  def backward(self, dh_next, x, h, h_prev, learning_rate):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    x = x.reshape(-1, 1)
    h_prev = h_prev.reshape(-1, 1)

    dhraw = self.activation_derivative(h) * dh_next

    dWx = dhraw @ x.T
    dWh = dhraw @ h_prev.T
    dbh = dhraw

    dh_prev = self.Wh.T @ dhraw

    # Gradient clipping
    clip_value = 5.0
    for grad in [dWx, dWh, dbh, dh_prev]:
      np.clip(grad, -clip_value, clip_value, out=grad)

    # Update weights and biases
    self.gradient_descent_update(dWx, dWh, dbh, learning_rate)

    return dh_prev

  def gradient_descent_update(self, dWx, dWh, dbh, learning_rate):
    LR = learning_rate
    if self.optimization == 'gd':
      self.Wx -= LR * dWx
      self.Wh -= LR * dWh
      self.bh -= LR * dbh
    else:
      raise ValueError(f"Unsupported optimization method: {self.optimization}")


class DecoderRNNLayer:
  def __init__(self, hidden_size, output_size, activation='tanh', optimization='gd') -> None:
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.activation = activation
    self.optimization = optimization
    self.initialize_weights(hidden_size, output_size)

  def initialize_weights(self, hidden_size, output_size):
    # Xavier initialization
    self.Wh = np.random.randn(hidden_size, hidden_size) * \
        np.sqrt(2. / (hidden_size + hidden_size))
    self.Wy = np.random.randn(output_size, hidden_size) * \
        np.sqrt(2. / (output_size + hidden_size))
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))
    self.reset_hidden_state()

  def reset_hidden_state(self):
    self.hidden_states = [np.zeros((self.hidden_size, 1))]  # initial state

  def activation_func(self, x):
    return activation.activation_fn(self.activation)(x)

  def activation_derivative(self, x):
    return activation.activation_fn(self.activation, derivative=True)(x)

  def forward(self, h_prev):
    h = self.activation_func(self.Wh @ h_prev + self.bh)
    y = self.Wy @ h + self.by

    self.hidden_states.append(h)
    return y, h

  def backward(self, dY, dh_next, h, h_prev, learning_rate):
    dWy = dY @ h.T
    dby = dY

    dh = (self.Wy.T @ dY) + dh_next

    dhraw = self.activation_derivative(h) * dh

    dWh = dhraw @ h_prev.T
    dbh = dhraw

    dh_prev = self.Wh.T @ dhraw

    # Gradient clipping
    clip_value = 5.0

    for grad in [dWy, dby, dWh, dbh, dh_prev]:
      np.clip(grad, -clip_value, clip_value, out=grad)

    # Update weights and biases
    self.gradient_descent_update(dWh, dWy, dbh, dby, learning_rate)

    return dh_prev

  def gradient_descent_update(self, dWh, dWy, dbh, dby, learning_rate):
    LR = learning_rate

    if self.optimization == 'gd':
      self.Wh -= LR * dWh
      self.Wy -= LR * dWy
      self.bh -= LR * dbh
      self.by -= LR * dby
    else:
      raise ValueError(f"Unsupported optimization method: {self.optimization}")


class EncoderDecoderRNN:
  def __init__(self, input_size, hidden_size, output_size, loss='mse', apply_softmax=False, activation='tanh', optimization='gd', model_file_name='rnn_seq2seq_model_weights') -> None:

    self.encoder = EncoderRNNLayer(
      input_size, hidden_size, activation, optimization)
    self.decoder = DecoderRNNLayer(
      hidden_size, output_size, activation, optimization)
    self.loss = loss
    self.apply_softmax = apply_softmax
    self.model_file_name = model_file_name
    self.load_weights_from_file()

  def loss_function(self, y_pred, y_true):
    return loss_func(self.loss)(y_pred, y_true)

  def loss_function_derivative(self, y_pred, y_true):
    return loss_func(self.loss, True)(y_pred, y_true)

  def forward(self, X_sequence):
    encoder_states = []
    for x in X_sequence:
      h = self.encoder.forward(x)
      encoder_states.append(h)

    Y = []
    h = encoder_states[-1]
    for _ in range(len(X_sequence)): # change to determine length of output sequence
      y, h = self.decoder.forward(h)
      if self.apply_softmax:
        Y.append(self.softmax(y))
      else:
        Y.append(y)
    return Y

  def softmax(self, a):
    return activation.activation_fn('softmax')(a)

  def predict(self, X_sequence):
    self.encoder.reset_hidden_state()
    self.decoder.reset_hidden_state()
    return self.forward(X_sequence)

  def backward(self, X_sequence, dY_sequence, learning_rate):
    n = len(dY_sequence)
    dh_next_decoder = np.zeros_like(self.decoder.hidden_states[-1])

    for t in reversed(range(n)):
      dy = dY_sequence[t]
      h, h_prev = self.decoder.hidden_states[t +
                                             1], self.decoder.hidden_states[t]
      dh_next_decoder = self.decoder.backward(
        dy, dh_next_decoder, h, h_prev, learning_rate)

    dh_next_encoder = dh_next_decoder
    for t in reversed(range(n)):
      x = X_sequence[t]
      h = self.encoder.hidden_states[t + 1]
      h_prev = self.encoder.hidden_states[t]
      dh_next_encoder = self.encoder.backward(
        dh_next_encoder, x, h, h_prev, learning_rate)

    self.encoder.reset_hidden_state()
    self.decoder.reset_hidden_state()

  def dataset_loss(self, X_sequences, Y_sequences):
    total_loss = 0
    for X_seq, Y_seq in zip(X_sequences, Y_sequences):

      self.encoder.reset_hidden_state()
      self.decoder.reset_hidden_state()
      outputs = self.forward(X_seq)
      loss = 0
      for y_pred, y_true in zip(outputs, Y_seq):
        loss += self.loss_function(y_pred, y_true)
      total_loss += loss / len(X_seq)
    return total_loss

  def train(self, X_sequences, Y_sequences, epochs, learning_rate, periodic_callback=None, new_dataset=None, decay_rate=0.0001):
    if periodic_callback:
      periodic_callback()
      print('')

    for epoch in range(1, epochs):
      LR = learning_rate / (1 + decay_rate * epoch)

      total_loss = 0
      for X_seq, Y_seq in zip(X_sequences, Y_sequences):

        self.encoder.reset_hidden_state()
        self.decoder.reset_hidden_state()

        outputs = self.forward(X_seq)

        loss = 0
        dY_sequence = []
        for y_pred, y_true in zip(outputs, Y_seq):
          y_true = np.array(y_true).reshape(-1, 1)
          loss += self.loss_function(y_pred, y_true)
          dY_sequence.append(self.loss_function_derivative(y_pred, y_true))

        total_loss += loss / len(outputs)

        self.backward(X_seq, dY_sequence, LR)

      if epoch % 10 == 0:
        self.output_weights_to_file()
        if periodic_callback:
          print('')
          periodic_callback()
          print('')

      if epoch % 30 == 0:
        if new_dataset:
          X_sequences, Y_sequences = new_dataset()

      print(f"Epoch {epoch+1}, Loss: {total_loss}, LR: {LR}")

  def load_weights_from_file(self):
    plain_model_file_name = self.model_file_name.replace('.json', '')
    model_file_name = plain_model_file_name + '.json'
    if os.path.exists(model_file_name):
      with open(model_file_name, 'r') as f:
        weights = json.load(f)
        encoder_Wh = np.array(weights['encoder_Wh'])
        encoder_bh = np.array(weights['encoder_bh'])
        if encoder_Wh.shape != self.encoder.Wh.shape or encoder_bh.shape != self.encoder.bh.shape:
          print('Shapes of saved model weights dont match, ignoring saved model')
          os.rename(f'{plain_model_file_name}.json',
                    f'{plain_model_file_name} (old).json')
          return

        self.encoder.Wh = encoder_Wh
        self.encoder.Wx = np.array(weights['encoder_Wx'])
        self.encoder.bh = encoder_bh

        self.decoder.Wh = np.array(weights['decoder_Wh'])
        self.decoder.bh = np.array(weights['decoder_bh'])
        self.decoder.Wy = np.array(weights['decoder_Wy'])
        self.decoder.by = np.array(weights['decoder_by'])

  def output_weights_to_file(self):
    model_file_name = self.model_file_name.replace('.json', '') + '.json'
    weights = {
        'encoder_Wh': self.encoder.Wh.tolist(),
        'encoder_Wx': self.encoder.Wx.tolist(),
        'encoder_bh': self.encoder.bh.tolist(),
        #
        'decoder_Wh': self.decoder.Wh.tolist(),
        'decoder_Wy': self.decoder.Wy.tolist(),
        'decoder_bh': self.decoder.bh.tolist(),
        'decoder_by': self.decoder.by.tolist()
    }
    with open(model_file_name, 'w') as f:
      json.dump(weights, f, indent=1)
