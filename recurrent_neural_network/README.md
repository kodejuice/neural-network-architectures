# Recurrent Neural Network

![RNN architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/2880px-Recurrent_neural_network_unfold.svg.png)
[RNN - Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network)

## Sample Usage

### Learn the next number in a progression

```python
import rnn
import numpy as np

max_num = 50
sequence_len = 10


def normalize(x): return x / max_num    # map number to 0...1
def normalize_seq(X): return [normalize(x) for x in X]


def one_hot_encode(x):
  s = np.zeros(max_num)
  s[x] = 1
  return s.reshape(-1, 1)


def encode_seq(X): return [one_hot_encode(x) for x in X]
def decode_seq(Y): return [np.argmax(y) for y in Y]


# Example usage
input_size = 1
hidden_size = 250
output_size = max_num
rnn = RNNNetwork(input_size, hidden_size, output_size,
                  apply_softmax=True, loss='cross_entropy')

def F(x):
  return (x * 3) % max_num

def generate_sequence(start, seq_len):
  seq = [start]
  for _ in range(seq_len):
    seq += [F(seq[-1])]
  return seq

train_sequences = [
  generate_sequence(i, sequence_len)
  for i in range(max_num)
] * 3

X_train = np.array([normalize_seq(seq[:-1]) for seq in train_sequences])
Y_train = np.array([encode_seq(seq[1:]) for seq in train_sequences])

def periodic_test():
  rand_start = np.random.randint(0, max_num)
  # double the length of training data sequence length
  seq = np.array(generate_sequence(rand_start, sequence_len * 2))
  X_test = normalize_seq(seq[:-1])
  Y_test = encode_seq(seq[1:])

  Y_pred = rnn.predict(X_test)
  Y_pred_val = np.array([0] + decode_seq(Y_pred))

  print('')
  print('X_test', seq[:-1], [seq[-1]])
  print('Y_pred', Y_pred_val)
  print('Validation loss', rnn.loss_function(Y_pred, Y_test))
  correct_pred = Y_pred_val[-1] == seq[-1]
  print(
    f'predicted last number: {Y_pred_val[-1]}, actual: {seq[-1]} {"✅" if correct_pred else "❌"}')

rnn.train(
  X_train,
  Y_train,
  epochs=500,
  learning_rate=0.0001,
  decay_rate=0.00001,
  periodic_callback=periodic_test,
)
```
