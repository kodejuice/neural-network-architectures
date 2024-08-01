import numpy as np
import os
import json
from nn_layer import Layer, OutputLayer, connect_layers


class DeepNeuralNetwork:
  def __init__(self, layers: list[Layer], loss='mse', L2_reg=0.0, beta1=0.9, beta2=0.999, optimization='gd', model_file=None):
    self.loss_fn = loss
    self.layers = connect_layers(layers, loss)
    self.assertions()
    self.training = False
    self.L2_reg = L2_reg
    self.beta1 = beta1
    self.beta2 = beta2
    self.optimization = optimization
    self.model_file_name = model_file or 'model_weights.json'
    self.load_weights_from_file()

  def assertions(self):
    if self.loss_fn == 'binary_cross_entropy':
      assert self.layers[-1].activation == 'sigmoid', \
          'Last layer must be sigmoid for binary cross entropy'
      assert self.layers[-1].output_size == 1, \
          'Last layer must have 1 neuron for binary cross entropy'

    for layer in self.layers:
      assert layer.activation in [
          'relu',
          'leaky_relu',
          'tanh',
          'sigmoid',
          'linear',
          'softmax',
      ], \
          f"Unsupported activation function: '{layer.activation}'"

    assert self.loss_fn in ['cross_entropy', 'mse', 'binary_cross_entropy'], \
        f"Unsupported loss function: '{self.loss_fn}'"

  def cost(self, X, Y):
    """cost over all examples in a batch"""
    A_L = self.full_forward_pass(X)
    Y = np.array(Y).T

    assert A_L.shape == Y.shape, \
        "Invalid shapes, A_L: %s, Y: %s" % (A_L.shape, Y.shape)

    cost = 0
    if self.loss_fn == 'mse':
      cost = np.mean(np.mean(np.square(A_L - Y), axis=0))
    elif self.loss_fn == 'cross_entropy':
      A_L = np.clip(A_L, 1e-15, 1 - 1e-15)
      cost = np.mean(-np.sum(Y * np.log(A_L), axis=0))
    elif self.loss_fn == 'binary_cross_entropy':
      A_L = np.clip(A_L, 1e-15, 1 - 1e-15)
      cost = np.mean(-np.sum(Y * np.log(A_L) + (1 - Y)
                     * np.log(1 - A_L), axis=0))

    # add L2 regularization
    if self.L2_reg > 0 and self.optimization in ['gd']:
      l2_reg = 0
      for layer in self.layers:
        l2_reg += np.sum(np.square(layer.W))
      cost += (self.L2_reg / 2) * l2_reg

    return cost

  def predict(self, X):
    return self.single_forward_pass(X)

  def single_forward_pass(self, X):
    """Foward pass for a single input"""
    X = np.array(X).reshape((self.layers[0].input_size, 1))
    A = X
    for layer in self.layers:
      A = layer.forward(A)
    return A

  def full_forward_pass(self, X):
    """Foward pass for a batch of inputs"""
    # X_T = np.array(X).T  # make all examples be arranged in a column
    # """
    # X_T = [
    #   [example1_a, example2_a, ..., exampleN_a],
    #   [example1_b, example2_b, ..., exampleN_b],
    #   [example1_c, example2_c, ..., exampleN_c],
    # ]
    # """
    X = np.array(X).T
    A = X
    for layer in self.layers:
      A = layer.forward(A, self.training)
    return A

  def backward_pass(self, Y, learning_rate, iteration=0):
    # we must have run a foward pass before calling this method

    Y = np.array(Y)
    Y_T = Y.T  # reshape training labels to be arranged in a column
    A_L = self.layers[-1].A

    if self.layers[-1].activation == 'softmax' and self.loss_fn == 'cross_entropy':
      dA = A_L - Y_T
    elif self.layers[-1].activation == 'sigmoid' and self.loss_fn == 'binary_cross_entropy':
      dA = A_L - Y_T
    else:
      assert self.loss_fn == 'mse', 'Expected mse loss'
      assert self.layers[-1].activation != 'softmax', 'Use a different activation function other than softmax here'

      dA = 2 * (A_L - Y_T) * \
          self.layers[-1].activation_fn(self.layers[-1].Z, derivative=True)

    for layer in reversed(self.layers):
      dA = layer.backward(
        dA, learning_rate,
        L2_reg=self.L2_reg,
        beta1=self.beta1,
        beta2=self.beta2,
        train_iteration=iteration,
        optimization=self.optimization
      )

  def train(self, X, Y, epochs=900000, initial_learning_rate=0.01, batch_size=64, decay_rate=0.0001, generate_dataset_fn=None, periodic_callback=None):
    print('Initial cost:', self.cost(X, Y))
    if periodic_callback:
      periodic_callback()
      print('')

    if any(l.keep_prob < 1 for l in self.layers):
      print('Applying Dropout to some layers')
    if self.L2_reg > 0 and self.optimization in ['gd']:
      print('Applying L2 regularization')

    for i in range(1, epochs):
      # decay learning rate
      learning_rate = initial_learning_rate / (1 + decay_rate * i)

      # Mini-batch gradient descent
      for j in range(0, len(X), batch_size):
        X_batch = X[j:j + batch_size]
        Y_batch = Y[j:j + batch_size]

        self.training = True
        self.full_forward_pass(X_batch)
        self.backward_pass(Y_batch, learning_rate, iteration=j)
        self.training = False

      if i % 10 == 0:
        loss = self.cost(X, Y)
        print(f'Epoch {i}, Loss: {loss:.6f}, LR: {learning_rate:.6f}')

      if i % 100 == 0:
        if periodic_callback:
          periodic_callback()
          print('')

        self.output_weights_to_file()

        if generate_dataset_fn:
          X, Y = generate_dataset_fn()
        else:
          # shuffle dataset
          XY = list(zip(X, Y))
          np.random.shuffle(XY)
          X, Y = zip(*XY)

    print('Final cost:', self.cost(X, Y))

  def nn_layers_params(self):
    s = ''
    for layer in self.layers:
      s += f'({layer.input_size}x{layer.output_size}, {layer.activation}) -> '
    return s

  def output_weights_to_file(self):
    layers_params = self.nn_layers_params()
    weights = {'network_params_hash': layers_params, 'weights': []}
    model_file_name = self.model_file_name.replace('.json', '')

    with open(f'{model_file_name}.json', 'w') as f:
      for i, layer in enumerate(self.layers):
        params = {
          'W': layer.W.tolist(),
          'b': layer.b.tolist(),
        }
        if layer.batch_norm:
          params['batch_norm_params'] = {
            'gamma': layer.batch_norm_layer.gamma.tolist(),
            'beta': layer.batch_norm_layer.beta.tolist(),
            'running_mean': layer.batch_norm_layer.running_mean.tolist(),
            'running_var': layer.batch_norm_layer.running_var.tolist(),
          }

        weights['weights'] += [params]
      f.write(json.dumps(weights, indent=1))

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

      if 'network_params_hash' not in model_weights:
        print('Error: weights file has no network_params_hash')
        os.rename(f'{model_file_name}.json', f'{model_file_name}_old.json')
        return

      if model_weights['network_params_hash'] != self.nn_layers_params():
        print('Error: weights file and current network layers hash do not match, ignoring')
        # rename old weights file
        os.rename(f'{model_file_name}.json', f'{model_file_name} (old).json')
        return

      weights = model_weights['weights']
      for i, layer in enumerate(self.layers):
        layer.W = np.array(weights[i]['W'])
        layer.b = np.array(weights[i]['b'])
        if layer.batch_norm and 'batch_norm_params' in weights[i]:
          Wi = weights[i]
          layer.batch_norm_layer.gamma = np.array(
            Wi['batch_norm_params']['gamma'])
          layer.batch_norm_layer.beta = np.array(
            Wi['batch_norm_params']['beta'])
          layer.batch_norm_layer.running_mean = np.array(
            Wi['batch_norm_params']['running_mean'])
          layer.batch_norm_layer.running_var = np.array(
            Wi['batch_norm_params']['running_var'])
