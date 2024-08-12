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

### 3. Learn Word Embedding with Word2Vec algorithm (Negative Sampling)

Fetch `shakespeare.txt` text from [here](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt)

```python
import re
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from deep_neural_network import DeepNeuralNetwork, Layer, OutputLayer

# import nltk
# nltk.download('punkt') # need to uncomment at first

# ============================== Utility functions ==============================


def split_corpus_into_sentences(corpus):
  corpus = re.sub(r'[^a-zA-Z.\s]', '', corpus)
  sentences = sent_tokenize(corpus)
  processed_sentences = [
      ' '.join([word.lower()
               for word in word_tokenize(sentence) if word.isalpha()])
      for sentence in sentences
  ]
  return processed_sentences


def prepare_training_data(sentences, window_size=3, num_negative_samples=7):
  print('Preparing training data...')
  training_data = []
  wp = word_probabilities

  for sentence in sentences:
    words = sentence.lower().split()
    if len(words) < window_size:
      continue

    for i, target in enumerate(words):
      context_start = max(0, i - window_size)
      context_end = min(len(words), i + window_size + 1)
      context = words[context_start:i] + words[i + 1:context_end]

      for context_word in context:
        positive_pair = (word2idx[context_word], word2idx[target], 1)
        training_data.append(positive_pair)

        for _ in range(num_negative_samples):
          negative_target_idx = np.random.choice(vocab_size, p=wp)
          while negative_target_idx == word2idx[target]:
            negative_target_idx = np.random.choice(vocab_size, p=wp)

          negative_word = corpus_words[negative_target_idx]
          negative_pair = (word2idx[context_word], word2idx[negative_word], 0)
          training_data.append(negative_pair)

  X = np.zeros((len(training_data), vocab_size))
  Y = np.zeros(len(training_data))
  for i, (context_idx, target_idx, label) in enumerate(training_data):
    X[i, [context_idx, target_idx]] = 1
    Y[i] = label

  print("Dataset size:", len(X))
  print("Vocabulary size:", vocab_size)

  return X, Y.reshape(-1, 1)


# ============================== Prepare training data ==============================

# Load Text corpus
shakespear_text = open("shakespeare.txt").read() # fetch from https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt
sentences = split_corpus_into_sentences(shakespear_text)

word_frequency = Counter(word.lower() for sentence in sentences for word in sentence.split())
corpus_words = list(word_frequency.keys())
word2idx = {word: idx for idx, word in enumerate(sorted(corpus_words))}
vocab_size = len(corpus_words)

# Compute Modified Distribution for Negative Sampling
word_counts = np.array(list(word_frequency.values()))
word_counts_34 = np.power(word_counts, 3 / 4)
total_words_34 = np.sum(word_counts_34)
word_probabilities = word_counts_34 / total_words_34

X, Y = prepare_training_data(sentences)

# ============================== Setup model ==============================

embedding_size = 10

model = DeepNeuralNetwork([
  Layer(vocab_size, activation='tanh'),
  Layer(embedding_size, activation='tanh'),
  OutputLayer(1, activation='sigmoid')
],
  loss='binary_cross_entropy',
  optimization='adam',
  model_file='word2vec_negative_sampling_model.json'
)

print('\nTraining model...')
# Train the model
model.train(X, Y, epochs=150, initial_learning_rate=0.01, batch_size=32)

# ============================== Inference ==============================


def get_word_embedding(word):
  # get the embedding vector for a word
  if word in word2idx:
    one_hot = [0] * vocab_size
    one_hot[word2idx[word]] = 1
    return model.layers[0].forward(np.array(one_hot).reshape(-1, 1)).flatten()
  else:
    return None


def similar_words(word):
  # get the top 5 most similar words to a word
  word_vec = get_word_embedding(word)
  if word_vec is None:
    return []

  similarities = []
  for word in corpus_words:
    embedding = get_word_embedding(word)
    if embedding is not None:
      similarity = np.dot(word_vec, embedding) / \
          (np.linalg.norm(word_vec) * np.linalg.norm(embedding))
      similarities.append((word, similarity))

  similarities.sort(key=lambda x: x[1], reverse=True)
  return [w for w, _ in similarities[:10]]


print("\nSimilar words:")
for i in range(15):
  word = np.random.choice(corpus_words)
  print(word, '=>', similar_words(word)[1:9])
```
