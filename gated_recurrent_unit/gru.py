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


class GRULayer:
  def __init__(self, input_size, hidden_size, output_size) -> None:
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.initialize_weights(input_size, hidden_size, output_size)

  def weight_matrix(self, r, c) -> np.ndarray:
    return np.random.randn(r, c)

  def initialize_weights(self, input_size, hidden_size, output_size):
    # Initialize weights and biases
    self.Wr = self.weight_matrix(hidden_size, hidden_size)
    self.Ur = self.weight_matrix(hidden_size, input_size)
    self.Wz = self.weight_matrix(hidden_size, hidden_size)
    self.Uz = self.weight_matrix(hidden_size, input_size)
    self.Wh = self.weight_matrix(hidden_size, hidden_size)
    self.Uh = self.weight_matrix(hidden_size, input_size)
    self.Wy = self.weight_matrix(output_size, hidden_size)
    self.br = np.zeros((hidden_size, 1))
    self.bz = np.zeros((hidden_size, 1))
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))
    self.reset_hidden_state()

  def reset_hidden_state(self):
    self.hidden_states = [
      # ht
      np.zeros((self.hidden_size, 1))  # initial state
    ]
    self.candidate_states = [
      # ht_candidate
      np.zeros((self.hidden_size, 1))
    ]
    self.read_gate_states = [
      # rt
      np.zeros((self.hidden_size, 1))
    ]
    self.forget_gate_states = [
      # zt
      np.zeros((self.hidden_size, 1))
    ]

  def foward(self, x, h_prev):
    if x is not np.ndarray:
      x = np.array(x)

    x = x.reshape(-1, 1)  # make a column vector

    rt = activation.sigmoid(self.Wr @ h_prev + self.Ur @ x + self.br)
    zt = activation.sigmoid(self.Wz @ h_prev + self.Uz @ x + self.bz)

    h_candidate = activation.tanh(
      self.Wh @ (rt * h_prev) + self.Uh @ x + self.bh)
    ht = zt * h_prev + (1 - zt) * h_candidate

    y = self.Wy @ ht + self.by

    self.hidden_states.append(ht)
    self.read_gate_states.append(rt)
    self.forget_gate_states.append(zt)
    self.candidate_states.append(h_candidate)
    return y, ht

  def backward(self, dY, dh_next, x, h, h_prev, h_candidate, rt, zt, learning_rate):
    if x is not np.ndarray:
      x = np.array(x)

    x = x.reshape(-1, 1)
    h_prev = h_prev.reshape(-1, 1)

    # Derivative of output layer
    dWy = dY @ h.T
    dby = dY

    # Backpropagate into hidden state
    dh = (self.Wy.T @ dY) + dh_next

    # Derivative of the candidate hidden state
    dtanh = dh * (1 - zt) * (1 - h_candidate**2)
    dWh = dtanh @ (rt * h_prev).T
    dUh = dtanh @ x.T
    dbh = dtanh

    # Derivative of the reset gate
    drt = dtanh * (self.Wh @ h_prev)
    dr_sigma = drt * rt * (1 - rt)
    dWr = dr_sigma @ h_prev.T
    dUr = dr_sigma @ x.T
    dbr = dr_sigma

    # Derivative of the forget gate
    dh_dz = h_prev - h_candidate
    dz = dh * dh_dz
    dz_sigma = dz * zt * (1 - zt)
    dWz = dz_sigma @ h_prev.T
    dUz = dz_sigma @ x.T
    dbz = dz_sigma

    # previous hidden state gradient
    dhcandidate_dhprev = (self.Wh @ rt) * \
        (1 - h_candidate**2)  # dht_candidate/dht_1
    dz_dhprev = self.Wz @ (zt * (1 - zt))
    dh_dhprev = zt + (1 - zt) * dhcandidate_dhprev + h_candidate * -dz_dhprev
    dh_prev = dh * dh_dhprev

    # Gradient clipping
    clip_value = 5.0
    for grad in [dWh, dUh, dbh, dWr, dUr, dbr, dWz, dUz, dbz, dWy, dby]:
      np.clip(grad, -clip_value, clip_value, out=grad)

    # Update weights and biases
    self.gradient_descent_update(
      dWh, dUh, dWr, dUr, dWz, dUz, dWy, dbz, dbr, dby, dbh, learning_rate)

    return dh_prev

  def backprop_through_time(self, input_sequence, dY_sequence, learning_rate):
    n = len(dY_sequence)
    dh_next = np.zeros_like(self.hidden_states[-1])

    for t in reversed(range(n)):
      dy = dY_sequence[t]
      x = input_sequence[t]
      h = self.hidden_states[t + 1]
      r = self.read_gate_states[t + 1]
      z = self.forget_gate_states[t + 1]
      h_candidate = self.candidate_states[t + 1]
      h_prev = self.hidden_states[t]
      dh_next = self.backward(dy, dh_next, x, h, h_prev,
                              h_candidate, r, z, learning_rate)

    self.reset_hidden_state()

  def gradient_descent_update(self, dWh, dUh, dWr, dUr, dWz, dUz, dWy, dbz, dbr, dby, dbh, learning_rate):
    LR = learning_rate
    self.Wr -= LR * dWr
    self.Ur -= LR * dUr
    self.Wz -= LR * dWz
    self.Uz -= LR * dUz
    self.Wh -= LR * dWh
    self.Uh -= LR * dUh
    self.Wy -= LR * dWy
    self.by -= LR * dby
    self.br -= LR * dbr
    self.bz -= LR * dbz
    self.bh -= LR * dbh


class GRUNetwork:
  def __init__(self, input_size, hidden_size, output_size, loss='mse', apply_softmax=False, model_file_name='gru_model_weights') -> None:
    self.gru_layer = GRULayer(input_size, hidden_size, output_size)
    self.loss = loss
    self.apply_softmax = apply_softmax
    self.model_file_name = model_file_name
    self.load_weights_from_file()

  def loss_function(self, y_pred, y_true):
    return loss_func(self.loss)(y_pred, y_true)

  def loss_function_derivative(self, y_pred, y_true):
    return loss_func(self.loss, True)(y_pred, y_true)

  def foward(self, X_sequence):
    Y = []
    h = np.zeros((self.gru_layer.hidden_size, 1))
    for x in X_sequence:
      y, h = self.gru_layer.foward(x, h)
      if self.apply_softmax:
        Y.append(self.softmax(y))
      else:
        Y.append(y)
    return Y

  def softmax(self, a):
    return activation.activation_fn('softmax')(a)

  def predict(self, X_sequence):
    self.gru_layer.reset_hidden_state()
    return self.foward(X_sequence)

  def backward(self, input_sequence, dY_sequence, learning_rate):
    self.gru_layer.backprop_through_time(
      input_sequence, dY_sequence, learning_rate)

  def dataset_loss(self, X_sequences, Y_sequences):
    total_loss = 0
    for X_seq, Y_seq in zip(X_sequences, Y_sequences):
      self.gru_layer.reset_hidden_state()
      outputs = self.foward(X_seq)
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
        self.gru_layer.reset_hidden_state()

        outputs = self.foward(X_seq)

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

  def gru_weight_shapes(self):
    layer = self.gru_layer
    # gd
    h = f'|Wh|={layer.Wh.shape} |Uh|={layer.Uh.shape} |bh|={layer.bh.shape}'
    h += f' |Wr|={layer.Wr.shape} |Ur|={layer.Ur.shape} |br|={layer.br.shape}'
    h += f' |Wz|={layer.Wz.shape} |Uz|={layer.Uz.shape} |bz|={layer.bz.shape}'
    h += f' |Wy|={layer.Wy.shape} |by|={layer.by.shape}'
    return h

  def output_weights_to_file(self):
    weights_shapes = self.gru_weight_shapes()
    weights = {'gru_weight_shapes': weights_shapes, 'weights': []}
    model_file_name = self.model_file_name.replace('.json', '')

    gru = self.gru_layer
    W = {
      # gd
      'gd': [gru.Wh.tolist(), gru.Uh.tolist(), gru.bh.tolist(),
             gru.Wr.tolist(), gru.Ur.tolist(), gru.br.tolist(),
             gru.Wz.tolist(), gru.Uz.tolist(), gru.bz.tolist(),
             gru.Wy.tolist(), gru.by.tolist()
             ],
    }

    weights['weights'] = W
    with open(f'{model_file_name}.json', 'w') as f:
      f.write(json.dumps(weights, indent=1))
      print(f'Weights saved to {model_file_name}.json')

  def load_weights_from_file(self):
    model_file_name = self.model_file_name.replace('.json', '')
    if os.path.exists(f'{model_file_name}.json'):
      print('Loading weights from file...')
    else:
      return

    with open(f'{model_file_name}.json', 'r+') as f:
      try:
        model_weights = json.loads(f.read())
      except:
        print('Error: weights file is not valid JSON')
        f.write('{}')
        return

      if 'gru_weight_shapes' not in model_weights:
        print('Error: weights file has no gru_weight_shapes')
        os.rename(f'{model_file_name}.json', f'{model_file_name}_old.json')
        return

      if model_weights['gru_weight_shapes'] != self.gru_weight_shapes():
        print(
          'Error: weights file and current network weights shape do not match, ignoring')
        # rename old weights file
        os.rename(f'{model_file_name}.json', f'{model_file_name} (old).json')
        return

      W = model_weights['weights']
      layer = self.gru_layer
      # Set weights for gradient descent
      layer.Wh = np.array(W['gd'][0])
      layer.Uh = np.array(W['gd'][1])
      layer.bh = np.array(W['gd'][2])
      layer.Wr = np.array(W['gd'][3])
      layer.Ur = np.array(W['gd'][4])
      layer.br = np.array(W['gd'][5])
      layer.Wz = np.array(W['gd'][6])
      layer.Uz = np.array(W['gd'][7])
      layer.bz = np.array(W['gd'][8])
      layer.Wy = np.array(W['gd'][9])
      layer.by = np.array(W['gd'][10])
      print('Weights loaded successfully!')

