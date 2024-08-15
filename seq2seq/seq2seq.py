from numpy import random as npr
from jax import grad
import jax.numpy as np


def init_params(input_size, hidden_size, output_size, attention_size=None):
  attention_size = attention_size or hidden_size
  i, h, o, a = input_size, hidden_size, output_size, attention_size
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


def mse_loss(y_pred, y_true):
  return np.sum((y_pred - y_true)**2)


def cross_entropy_loss(y_pred, y_true):
  epsilon = 1e-15
  y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
  return -np.sum(y_true * np.log(y_pred))


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
  # alpha = softmax(alpha)
  alpha = sigmoid(alpha)
  return np.array(alpha)
# ================================================


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

  return Y


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


def train(X_sequences, Y_sequences, input_size, hidden_size, epochs, learning_rate=0.01, loss='mse', apply_softmax=False, periodic_callback=None, decay_rate=0.0001):
  output_size = Y_sequences[0][0].shape[0]
  params = init_params(input_size, hidden_size, output_size)

  loss_func = mse_loss if loss == 'mse' or apply_softmax == False else cross_entropy_loss

  for epoch in range(epochs):
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

    if epoch % 10 == 0:
      if periodic_callback:
        periodic_callback(params)

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(X_sequences)}")

  return params


# ================================================================================


max_num = 50 >> 1
sequence_len = 10


def normalize(x): return x / max_num    # map number to 0...1
def normalize_seq(X): return np.array([normalize(x) for x in X])


def one_hot_encode(x):
  s = np.zeros(max_num).tolist()
  s[x] = 1
  s = np.array(s)
  return s.reshape(-1, 1)


def encode_seq(X): return [one_hot_encode(x) for x in X]
def decode_seq(Y): return [np.argmax(y) for y in Y]


def F(x):
  return (x * 3) % max_num


def generate_sequence(start, seq_len=None):
  seq = [start]
  for _ in range(seq_len):
    seq += [F(seq[-1])]
  return seq


train_sequences = [
  generate_sequence(i, sequence_len)
  for i in range(max_num)
]

X_train = np.array([normalize_seq(seq[:-1]) for seq in train_sequences])
Y_train = np.array([encode_seq(seq[1:]) for seq in train_sequences])


def periodic_test(params):
  rand_start = npr.randint(0, max_num)
  seq = np.array(generate_sequence(rand_start, sequence_len))
  X_test = normalize_seq(seq[:-1])
  Y_test = encode_seq(seq[1:])

  Y_pred = predict(params, X_test)
  Y_pred_val = np.array([0] + decode_seq(Y_pred))

  print('')
  print('X_test', seq[:-1], [seq[-1]])
  print('Y_pred', Y_pred_val)
  print('Validation loss', sequence_loss(
    params, X_test, Y_test, cross_entropy_loss, apply_softmax=True))
  correct_pred = Y_pred_val[-1] == seq[-1]
  print(
    f'predicted last number: {Y_pred_val[-1]}, actual: {seq[-1]} {"✅" if correct_pred else "❌"}')


params = train(
  X_train,
  Y_train,
  input_size=1,
  hidden_size=250,
  epochs=250,
  learning_rate=0.01,
  periodic_callback=periodic_test,
  apply_softmax=True
)
