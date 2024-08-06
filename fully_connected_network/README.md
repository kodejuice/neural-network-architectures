# FeedFoward Neural Network

![Feedfoward neural network architecure](https://upload.wikimedia.org/wikipedia/commons/5/54/Feed_forward_neural_net.gif)

[Feedforward neural network - Wikipedia](https://en.wikipedia.org/wiki/Feedforward_neural_network)

## Sample Usage

### 1. Learn the sum of two numbers

```python
import numpy as np
from deep_neural_network import *

# Generate dataset
def generate_dataset(num_samples=3000):
  X = np.random.uniform(-10, 10, (num_samples, 2))
  y = np.sum(X, axis=1, keepdims=True)
  return X, y


X_train, y_train = generate_dataset()

X_train_min, X_train_max = X_train.min(), X_train.max()
y_train_min, y_train_max = y_train.min(), y_train.max()

# Normalize input data to range 0...1
X_train_normalized = (X_train - X_train_min) / (X_train_max - X_train_min)
y_train_normalized = (y_train - y_train_min) / (y_train_max - y_train_min)

# Create and train the neural network
dnn = DeepNeuralNetwork(
    layers=[
        Layer(2),  # input layer
        Layer(64, 'tanh'),  # hidden layer
        Layer(64, 'tanh'),  # hidden layer
        OutputLayer(1, 'sigmoid'),  # output layer
    ],
    loss='mse',
    optimization='adam',
    model_file='sum_model.json'
)

# Test the trained model
def test_model(num_tests=3):
  print('')
  for _ in range(num_tests):
    a, b = np.random.uniform(-10, 10, 2)
    # Normalize test input
    a_norm = (a - X_train_min) / (X_train_max - X_train_min)
    b_norm = (b - X_train_min) / (X_train_max - X_train_min)
    predicted_sum_norm = dnn.predict([a_norm, b_norm])[0][0]
    # Denormalize the prediction
    predicted_sum = predicted_sum_norm * \
        (y_train_max - y_train_min) + y_train_min
    actual_sum = a + b
    print(f"{a:.2f} + {b:.2f} = {predicted_sum:.2f} (Actual: {actual_sum:.2f})")



dnn.train(
  X_train_normalized,
  y_train_normalized,
  epochs=10_000,
  batch_size=32,
  initial_learning_rate=0.0001,
  periodic_callback=test_model,
)
```

### 2. Learn a classification using softmax

```python
import numpy as np
from deep_neural_network import *

# generate dataset
def generate_dataset(num_samples=1000):
  X = np.random.uniform(-50, 50, (num_samples, 1))
  y = np.zeros((num_samples, 3))  # 3 classes: left, middle, right

  for i, point in enumerate(X):
    if point < -3:
      y[i, 0] = 1  # Left
    elif point > 3:
      y[i, 2] = 1  # Right
    else:
      y[i, 1] = 1  # Middle

  return X, y


X_train, y_train = generate_dataset()

dnn = DeepNeuralNetwork(
  layers=[
    Layer(1),  # input layer
    Layer(32, 'tanh'),    # hidden layer
    Layer(32, 'tanh', batch_norm=True),    # hidden layer
    Layer(32, 'tanh'),    # hidden layer
    OutputLayer(3, 'softmax'),      # output layer
  ],
  loss='cross_entropy',
  optimization='adam',
  model_file='point_classification_model.json'
)


def rand_test():
  n = np.random.uniform(-3, 3)
  v = dnn.predict(n)
  correct = False
  if n < -3:
    correct = v[0] > 0.8
  elif n > 3:
    correct = v[2] > 0.8
  else:
    correct = v[1] > 0.8
  return n, correct[0]


dnn.train(
  X_train,
  y_train,
  epochs=900,
  initial_learning_rate=0.01,
  decay_rate=1e-6,
  generate_dataset_fn=generate_dataset,
  periodic_callback=lambda: print(rand_test())
)
```
