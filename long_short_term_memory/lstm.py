import numpy as np
import os
import json

# TODO: remove
try:
  from . import activation
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
  return L[loss][int(deriv)]


class LSTMLayer:
  def __init__(self, input_size, hidden_size, output_size) -> None:
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.initialize_weights(input_size, hidden_size, output_size)

  def weight_matrix(self, r, c) -> np.ndarray:
    return np.random.randn(r, c)

  def initialize_weights(self, input_size, hidden_size, output_size):
    # Initialize weights and biases
    self.Wf = self.weight_matrix(hidden_size, hidden_size + input_size)
    self.Wi = self.weight_matrix(hidden_size, hidden_size + input_size)
    self.Wc = self.weight_matrix(hidden_size, hidden_size + input_size)
    self.Wo = self.weight_matrix(hidden_size, hidden_size + input_size)
    self.Wy = self.weight_matrix(output_size, hidden_size)

    self.bf = np.zeros((hidden_size, 1))
    self.bi = np.zeros((hidden_size, 1))
    self.bc = np.zeros((hidden_size, 1))
    self.bo = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))

    self.reset_state()

  def reset_state(self):
    self.hidden_states = [np.zeros((self.hidden_size, 1))]
    self.cell_states = [np.zeros((self.hidden_size, 1))]
    self.forget_gate_states = []
    self.input_gate_states = []
    self.cell_gate_states = []
    self.output_gate_states = []

  def forward(self, x, h_prev, c_prev):
    if not isinstance(x, np.ndarray):
      x = np.array(x)

    x = x.reshape(-1, 1)

    concat = np.vstack((h_prev, x))

    f = activation.sigmoid(self.Wf @ concat + self.bf)
    i = activation.sigmoid(self.Wi @ concat + self.bi)
    c_tilde = activation.tanh(self.Wc @ concat + self.bc)
    o = activation.sigmoid(self.Wo @ concat + self.bo)

    c = f * c_prev + i * c_tilde
    h = o * activation.tanh(c)

    y = self.Wy @ h + self.by

    self.forget_gate_states.append(f)
    self.input_gate_states.append(i)
    self.cell_gate_states.append(c_tilde)
    self.output_gate_states.append(o)
    self.hidden_states.append(h)
    self.cell_states.append(c)

    return y, h, c

  def backward(self, dY, dh_next, dc_next, x, h, c, c_prev, f, i, c_tilde, o, learning_rate):
    if not isinstance(x, np.ndarray):
      x = np.array(x)

    x = x.reshape(-1, 1)

    # Derivative of output layer
    dWy = dY @ h.T
    dby = dY
    dh = self.Wy.T @ dY + dh_next

    # Backpropagate through time
    do = dh * activation.tanh(c)
    dc = dc_next + (dh * o) * (1 - activation.tanh(c)**2)
    dc_tilde = dc * i
    di = dc * c_tilde
    df = dc * c_prev

    # Derivative of gate activations
    do_input = do * o * (1 - o)
    di_input = di * i * (1 - i)
    df_input = df * f * (1 - f)
    dc_tilde_input = dc_tilde * (1 - c_tilde**2)

    # Concatenate h and x for weight updates
    concat = np.vstack((h, x))

    # Compute weight gradients
    dWf = df_input @ concat.T
    dWi = di_input @ concat.T
    dWc = dc_tilde_input @ concat.T
    dWo = do_input @ concat.T

    # Compute bias gradients
    dbf = df_input
    dbi = di_input
    dbc = dc_tilde_input
    dbo = do_input

    # Compute gradients for next time step
    dconcat = self.Wf.T @ df_input + self.Wi.T @ di_input + \
        self.Wc.T @ dc_tilde_input + self.Wo.T @ do_input
    dh_prev = dconcat[:self.hidden_size]
    dc_prev = f * dc

    # Gradient clipping
    clip_value = 5.0
    for grad in [dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby]:
      np.clip(grad, -clip_value, clip_value, out=grad)

    # Update weights and biases
    self.gradient_descent_update(
      dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby, learning_rate)

    return dh_prev, dc_prev

  def backprop_through_time(self, input_sequence, dY_sequence, learning_rate):
    n = len(dY_sequence)
    dh_next = np.zeros_like(self.hidden_states[-1])
    dc_next = np.zeros_like(self.cell_states[-1])

    for t in reversed(range(n)):
      dy = dY_sequence[t]
      x = input_sequence[t]
      h = self.hidden_states[t + 1]
      c = self.cell_states[t + 1]
      c_prev = self.cell_states[t]
      f = self.forget_gate_states[t]
      i = self.input_gate_states[t]
      c_tilde = self.cell_gate_states[t]
      o = self.output_gate_states[t]

      dh_next, dc_next = self.backward(
        dy, dh_next, dc_next, x, h, c, c_prev, f, i, c_tilde, o, learning_rate)

    self.reset_state()

  def gradient_descent_update(self, dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby, learning_rate):
    LR = learning_rate
    self.Wf -= LR * dWf
    self.Wi -= LR * dWi
    self.Wc -= LR * dWc
    self.Wo -= LR * dWo
    self.Wy -= LR * dWy
    self.bf -= LR * dbf
    self.bi -= LR * dbi
    self.bc -= LR * dbc
    self.bo -= LR * dbo
    self.by -= LR * dby


class LSTMNetwork:
  def __init__(self, input_size, hidden_size, output_size, loss='mse', apply_softmax=False, model_file_name='lstm_model_weights') -> None:
    self.lstm_layer = LSTMLayer(input_size, hidden_size, output_size)
    self.loss = loss
    self.apply_softmax = apply_softmax
    self.model_file_name = model_file_name
    self.load_weights_from_file()

  def loss_function(self, y_pred, y_true):
    return loss_func(self.loss)(y_pred, y_true)

  def loss_function_derivative(self, y_pred, y_true):
    return loss_func(self.loss, True)(y_pred, y_true)

  def forward(self, X_sequence):
    Y = []
    h = np.zeros((self.lstm_layer.hidden_size, 1))
    c = np.zeros((self.lstm_layer.hidden_size, 1))
    for x in X_sequence:
      y, h, c = self.lstm_layer.forward(x, h, c)
      if self.apply_softmax:
        Y.append(self.softmax(y))
      else:
        Y.append(y)
    return Y

  def softmax(self, a):
    return activation.activation_fn('softmax')(a)

  def predict(self, X_sequence):
    self.lstm_layer.reset_state()
    return self.forward(X_sequence)

  def backward(self, input_sequence, dY_sequence, learning_rate):
    self.lstm_layer.backprop_through_time(
      input_sequence, dY_sequence, learning_rate)

  def dataset_loss(self, X_sequences, Y_sequences):
    total_loss = 0
    for X_seq, Y_seq in zip(X_sequences, Y_sequences):
      self.lstm_layer.reset_state()
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
        self.lstm_layer.reset_state()

        outputs = self.forward(X_seq)

        loss = 0
        dY_sequence = []
        for y_pred, y_true in zip(outputs, Y_seq):
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

      print(f"Epoch {epoch}, Loss: {total_loss}, LR: {LR}")

  def load_weights_from_file(self):
    model_file_name = self.model_file_name.replace('.json', '') + '.json'
    if os.path.exists(model_file_name):
      with open(model_file_name, 'r') as f:
        weights = json.load(f)
        Wf = np.array(weights['Wf'])
        bf = np.array(weights['bf'])
        if Wf.shape != self.lstm_layer.Wf.shape or bf.shape != self.lstm_layer.bf.shape:
          print('Shapes of saved model dont match')
          os.rename(f'{model_file_name}.json', f'{model_file_name} (old).json')
          return

        self.lstm_layer.Wf = Wf
        self.lstm_layer.Wi = np.array(weights['Wi'])
        self.lstm_layer.Wc = np.array(weights['Wc'])
        self.lstm_layer.Wo = np.array(weights['Wo'])
        self.lstm_layer.Wy = np.array(weights['Wy'])
        self.lstm_layer.bf = bf
        self.lstm_layer.bi = np.array(weights['bi'])
        self.lstm_layer.bc = np.array(weights['bc'])
        self.lstm_layer.bo = np.array(weights['bo'])
        self.lstm_layer.by = np.array(weights['by'])

  def output_weights_to_file(self):
    model_file_name = self.model_file_name.replace('.json', '') + '.json'
    weights = {
        'Wf': self.lstm_layer.Wf.tolist(),
        'Wi': self.lstm_layer.Wi.tolist(),
        'Wc': self.lstm_layer.Wc.tolist(),
        'Wo': self.lstm_layer.Wo.tolist(),
        'Wy': self.lstm_layer.Wy.tolist(),
        'bf': self.lstm_layer.bf.tolist(),
        'bi': self.lstm_layer.bi.tolist(),
        'bc': self.lstm_layer.bc.tolist(),
        'bo': self.lstm_layer.bo.tolist(),
        'by': self.lstm_layer.by.tolist()
    }
    with open(model_file_name, 'w') as f:
      json.dump(weights, f)
