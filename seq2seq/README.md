# RNN Encoder Decoder - Seq2Seq

## Seq2Seq With Attention

![RNN Encoder Decoder - Seq2Seq with attention](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*TPlS-uko-n3uAxbAQY_STQ.png)
[Source - A simple overview of RNN, LSTM and Attention Mechanism](https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b)

### Learn to reverse a list of numbers

This can be seen as a translation task

Requires:

- `pip install jax`

```python
import jax.numpy as np
from numpy import random as npr
from seq2seq import Seq2Seq

# Set parameters
N = 10  # Maximum number in the sequence
seq_length = 5  # Length of the sequence
num_samples = 1000  # Number of training samples
hidden_size = 200
learning_rate = 0.01
num_epochs = 200

# Generate training data
X = npr.randint(1, N + 1, size=(num_samples, seq_length))
Y = np.flip(X, axis=1)

# Normalize inputs
X_normalized = X / N

# One-hot encode outputs
Y_onehot = np.eye(N)[Y - 1]

# Create and train the model
model = Seq2Seq(
  input_size=1,
  hidden_size=hidden_size,
  output_size=N,
  model_filename='reverse_number_model.json',
  loss='cross_entropy',
  apply_softmax=True
)

def periodic_test():
  # Test the model
  test_seq = npr.randint(1, N+1, size=(1, seq_length))
  test_seq_normalized = test_seq / N
  predicted = model.predict(test_seq_normalized[0])
  predicted_seq = np.argmax(predicted, axis=1) + 1

  print("Test sequence:", test_seq[0])
  print("Predicted (reversed) sequence:", predicted_seq.flatten())
  print("Actual reversed sequence:", np.flip(test_seq[0]))
  print()


model.train(
  X_sequences=X_normalized,
  Y_sequences=Y_onehot,
  epochs=num_epochs,
  learning_rate=learning_rate,
  periodic_callback=periodic_test,
)
```
