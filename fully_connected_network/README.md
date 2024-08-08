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

### 3. Word2Vec Implementation

Fetch `shakespeare.txt` text from [here](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt)

```python
import re
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from deep_neural_network import DeepNeuralNetwork, Layer, OutputLayer

# import nltk
# nltk.download('punkt') # need to uncomment at first


def split_corpus_into_sentences(corpus):
  corpus = re.sub(r'[^a-zA-Z.\s]', '', corpus)
  sentences = sent_tokenize(corpus)
  processed_sentences = [
      ' '.join([word.lower()
               for word in word_tokenize(sentence) if word.isalpha()])
      for sentence in sentences
  ]
  return processed_sentences


def prepare_training_data(sentences, window_size=10):
  training_data = []
  for sentence in sentences:
    words = sentence.lower().split()
    if len(words) < window_size:
      continue
    for i in range(len(words) - window_size + 1):
      context = words[i:i + window_size // 2] + \
        words[i + window_size // 2 + 1:i + window_size]
      target = words[i + window_size // 2]
      context = [word2idx[word] for word in context]
      target = [word2idx[target]]
      training_data.append((context, target))

  X, Y = list(zip(*training_data))

  X_one_hot = []
  for context in X:
    context_vector = [0] * vocab_size
    for word_idx in context:
      context_vector[word_idx] += 1
    X_one_hot.append(context_vector)

  Y_one_hot = []
  for target in Y:
    target_vector = [0] * vocab_size
    target_vector[target[0]] = 1
    Y_one_hot.append(target_vector)

  return X_one_hot, Y_one_hot


# Load Text corpus
shakespear_text = open("shakespeare.txt").read() # fetch from https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt
shakespear_sentences = split_corpus_into_sentences(shakespear_text)

words = set(word.lower()
            for sentence in shakespear_sentences for word in sentence.split())
word2idx = {word: idx for idx, word in enumerate(sorted(words))}
vocab_size = len(words)

X, Y = prepare_training_data(shakespear_sentences)

embedding_size = 5

model = DeepNeuralNetwork([
  Layer(vocab_size, activation='tanh'),
  Layer(embedding_size, activation='tanh'),
  OutputLayer(vocab_size, activation='softmax')
],
  loss='cross_entropy',
  optimization='adam',
  model_file='word2vec_embedding_model.json'
)

# Train the model
model.train(X, Y, epochs=700, initial_learning_rate=0.01, batch_size=32)


# Inference, Get similar words

def get_word_embedding(word):
  # get the embedding vector for a word
  if word in word2idx:
    word_idx = word2idx[word]
    one_hot = [0] * vocab_size
    one_hot[word_idx] = 1
    return model.layers[0].forward(np.array(one_hot).reshape(-1, 1)).flatten()
  else:
    return None


def similar_words(word):
  word_vec = get_word_embedding(word)
  if word_vec is None:
    return []

  similarities = []
  for word in words:
    embedding = get_word_embedding(word)
    if embedding is not None:
      similarity = np.dot(word_vec, embedding) / \
          (np.linalg.norm(word_vec) * np.linalg.norm(embedding))
      similarities.append((word, similarity))

  similarities.sort(key=lambda x: x[1], reverse=True)
  return [w for w, _ in similarities[:5]]


for i, word in enumerate(words):
  print(word, similar_words(word)[1:4])
  if i == 20:
    break
```
