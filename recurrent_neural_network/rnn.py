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
  return L[loss][deriv]


def momentum_update(param: np.ndarray, grad: np.ndarray, m: np.ndarray, learning_rate, beta=0.9):
  m = beta * m + (1 - beta) * grad
  param -= learning_rate * m
  return param, m


class RNNLayer:
  def __init__(self, input_size, hidden_size, output_size, activation='tanh', optimization='gd') -> None:
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.activation = activation
    self.optimization = optimization
    self.initialize_weights(input_size, hidden_size, output_size)

  def initialize_weights(self, input_size, hidden_size, output_size):
    # Initialize weights and biases
    # Xavier initialization
    self.Wx = np.random.randn(hidden_size, input_size) * \
        np.sqrt(2. / (hidden_size + input_size))
    self.Wh = np.random.randn(
      hidden_size, hidden_size) * np.sqrt(2. / (hidden_size + hidden_size))
    self.Wy = np.random.randn(
      output_size, hidden_size) * np.sqrt(2. / (output_size + hidden_size))
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))

    # initializtion for momentum
    self.v_Wx = np.zeros_like(self.Wx)
    self.v_Wh = np.zeros_like(self.Wh)
    self.v_Wy = np.zeros_like(self.Wy)
    self.v_bh = np.zeros_like(self.bh)
    self.v_by = np.zeros_like(self.by)

    self.reset_hidden_state()

  def reset_hidden_state(self):
    self.hidden_states = [
      np.zeros((self.hidden_size, 1))  # initial state
    ]

  def activation_func(self, x):
    return activation.activation_fn(self.activation)(x)

  def activation_derivative(self, x):
    return activation.activation_fn(self.activation, derivative=True)(x)

  def foward(self, x, h_prev):
    if x is not np.ndarray:
      x = np.array(x)

    x = x.reshape(-1, 1)  # make a column vector

    h = self.activation_func(self.Wx @ x + self.Wh @ h_prev + self.bh)
    y = self.Wy @ h + self.by

    self.hidden_states.append(h)
    return y, h

  def backward(self, dY, dh_next, x, h, h_prev, learning_rate):
    if x is not np.ndarray:
      x = np.array(x)
    x = x.reshape(-1, 1)
    h_prev = h_prev.reshape(-1, 1)

    # gradients for output layer
    dWy = dY @ h.T
    dby = dY

    # gradient for hiddent state
    dh = (self.Wy.T @ dY) + dh_next

    # activation function derivate
    dhraw = self.activation_derivative(h) * dh

    # gradients for weights
    dWx = dhraw @ x.T
    dWh = dhraw @ h_prev.T
    dbh = dhraw

    # gradient for previous hidden state
    dh_prev = self.Wh.T @ dhraw
    # dx = self.Wx.T @ dhraw

    # Gradient clipping
    clip_value = 5.0
    for grad in [dWx, dWh, dWy, dbh, dby, dh_prev]:
      np.clip(grad, -clip_value, clip_value, out=grad)

    # Update weights and biases
    self.gradient_descent_update(dWx, dWh, dWy, dbh, dby, learning_rate)

    return dh_prev

  def backprop_through_time(self, input_sequence, dY_sequence, learning_rate):
    n = len(dY_sequence)
    dh_next = np.zeros_like(self.hidden_states[-1])

    for t in reversed(range(n)):
      dy = dY_sequence[t]
      x = input_sequence[t]
      h = self.hidden_states[t + 1]
      h_prev = self.hidden_states[t]
      dh_next = self.backward(dy, dh_next, x, h, h_prev, learning_rate)

    self.reset_hidden_state()

  def gradient_descent_update(self, dWx, dWh, dWy, dbh, dby, learning_rate):
    LR = learning_rate
    optimization = self.optimization
    if optimization == 'gd':
      self.Wx -= LR * dWx
      self.Wh -= LR * dWh
      self.Wy -= LR * dWy
      self.bh -= LR * dbh
      self.by -= LR * dby
    elif optimization == 'momentum':
      self.Wx, self.v_Wx = momentum_update(self.Wx, dWx, self.v_Wx, LR)
      self.Wh, self.v_Wh = momentum_update(self.Wh, dWh, self.v_Wh, LR)
      self.Wy, self.v_Wy = momentum_update(self.Wy, dWy, self.v_Wy, LR)
      self.bh, self.v_bh = momentum_update(self.bh, dbh, self.v_bh, LR)
      self.by, self.v_by = momentum_update(self.by, dby, self.v_by, LR)
    else:
      raise ValueError(f"Unsupported optimization method: {optimization}")


class RNNNetwork:
  def __init__(self, input_size, hidden_size, output_size, loss='mse', apply_softmax=False, activation='tanh', optimization='gd', model_file_name='model_weights') -> None:
    self.rnn_layer = RNNLayer(input_size, hidden_size,
                              output_size, activation, optimization)
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
    h = np.zeros((self.rnn_layer.hidden_size, 1))
    for x in X_sequence:
      y, h = self.rnn_layer.foward(x, h)
      if self.apply_softmax:
        Y.append(self.softmax(y))
      else:
        Y.append(y)
    return Y

  def softmax(self, a):
    return activation.activation_fn('softmax')(a)

  def predict(self, X_sequence):
    self.rnn_layer.reset_hidden_state()
    return self.foward(X_sequence)

  def backward(self, input_sequence, dY_sequence, learning_rate):
    self.rnn_layer.backprop_through_time(
      input_sequence, dY_sequence, learning_rate)

  def dataset_loss(self, X_sequences, Y_sequences):
    total_loss = 0
    for X_seq, Y_seq in zip(X_sequences, Y_sequences):
      self.rnn_layer.reset_hidden_state()
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
        self.rnn_layer.reset_hidden_state()

        outputs = self.foward(X_seq)

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

  def rnn_weight_shapes(self):
    layer = self.rnn_layer
    # gd
    h = f'|Wx|={layer.Wx.shape} |Wy|={layer.Wy.shape} |Wh|={layer.Wh.shape} |bh|={layer.bh.shape} |by|={layer.by.shape}'
    if layer.optimization == 'momentum':
      # momentum
      h += f' <> |v_Wx|={layer.v_Wx.shape} |v_Wy|={layer.v_Wy.shape} |v_Wh|={layer.v_Wh.shape} |v_bh|={layer.v_bh.shape} |v_by|={layer.v_by.shape}'
    return h

  def output_weights_to_file(self):
    weights_shapes = self.rnn_weight_shapes()
    weights = {'rnn_weight_shapes': weights_shapes, 'weights': []}
    model_file_name = self.model_file_name.replace('.json', '')

    rnn = self.rnn_layer
    W = {
      # gd
      'gd': [rnn.Wh.tolist(), rnn.Wx.tolist(), rnn.Wy.tolist(),
             rnn.bh.tolist(), rnn.by.tolist()],

      # momentum
      'momentum': None if rnn.optimization != 'momentum' else [
          [rnn.v_Wh.tolist(), rnn.v_Wx.tolist(), rnn.v_Wy.tolist(),
           rnn.v_bh.tolist(), rnn.v_by.tolist()]
      ]
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

      if 'rnn_weight_shapes' not in model_weights:
        print('Error: weights file has no rnn_weight_shapes')
        os.rename(f'{model_file_name}.json', f'{model_file_name}_old.json')
        return

      if model_weights['rnn_weight_shapes'] != self.rnn_weight_shapes():
        print(
          'Error: weights file and current network weights shape do not match, ignoring')
        # rename old weights file
        os.rename(f'{model_file_name}.json', f'{model_file_name} (old).json')
        return

      W = model_weights['weights']
      layer = self.rnn_layer
      # Set weights for gradient descent
      layer.Wh = np.array(W['gd'][0])
      layer.Wx = np.array(W['gd'][1])
      layer.Wy = np.array(W['gd'][2])
      layer.bh = np.array(W['gd'][3])
      layer.by = np.array(W['gd'][4])

      # Set weights for momentum optimization
      if layer.optimization == 'momentum' and W['momentum'] is not None:
        layer.v_Wh, layer.v_Wx, layer.v_Wy, layer.v_bh, layer.v_by = [
          np.array(w) for w in W['momentum'][0]]

      print('Weights loaded successfully!')

