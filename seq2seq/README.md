# RNN Encoder Decoder - Seq2Seq

## Seq2Seq With Attention

![RNN Encoder Decoder - Seq2Seq with attention](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*TPlS-uko-n3uAxbAQY_STQ.png)
[Source - A simple overview of RNN, LSTM and Attention Mechanism](https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b)

### Learn to reverse a list of numbers

This can be seen as a translation task

Requires [jax](https://github.com/google/jax):

- `pip install jax`

```python
import jax.numpy as np
from numpy import random as npr
from seq2seq import Seq2Seq

seq_length = 4  # Length of the sequence
num_samples = 1000  # Number of training samples
hidden_size = 50
learning_rate = 0.01
num_epochs = 90

# Generate training data
X = npr.sample((num_samples, seq_length))
Y = np.flip(X, axis=1)

# Create and train the model
model = Seq2Seq(
  input_size=1,
  hidden_size=hidden_size,
  output_size=1,
  loss='mse',
)

def periodic_test():
  # Test the model
  test_seq = npr.sample((1, seq_length))[0]
  reversed_seq = np.flip(test_seq)
  predicted = model.predict(test_seq)
  print('-' * 50)
  print("Test sequence:", test_seq)
  print('-' * 50)
  print("Predicted (reversed) sequence:", predicted.flatten())
  print("Actual reversed sequenced:    ", reversed_seq)
  print('-' * 50)
  print("Loss:", model.sequence_loss(test_seq, reversed_seq))
  print()


periodic_test()

model.train(
  X_sequences=X,
  Y_sequences=Y,
  epochs=num_epochs,
  learning_rate=learning_rate,
  periodic_callback=periodic_test,
)
```
