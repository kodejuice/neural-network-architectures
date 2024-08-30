# Transformers

![Transformer architecture](https://upload.wikimedia.org/wikipedia/commons/3/34/Transformer%2C_full_architecture.png)

Implemented with [PyTorch](https://pytorch.org)

## Learn to invert a sequence of 1's and 0's

```python
import os
import torch
import torch.nn as nn
import numpy as np

from transformer import Transformer

model = Transformer(embed_dim=8,
                    src_vocab_size=5,
                    target_vocab_size=5,
                    num_decode_layers=3,
                    num_encode_layers=3,
                    src_seq_len=10,
                    trg_seq_len=9,
                    n_heads=4,
                    dropout=0.1,
                    pad_token=4
                    )

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

if os.path.exists('transformer.pt'):
  try:
    model.load_state_dict(torch.load('transformer.pt', weights_only=False))
  except:
    print('Failed to load model')


SOS_token = 2

def generate_random_data(n, batch_size=16):
  data = []
  length = 9

  SOS = np.array([SOS_token])

  # 1,1,1,1,1,1 -> 0,0,0,0,0
  for _ in range(n // 2):
    X = np.concatenate((SOS, np.ones(length)))
    y = np.concatenate((SOS, np.zeros(length)))
    data.append([X, y])

  # 0,0,0,0,0 -> 1,1,1,1,1,1
  for _ in range(n // 2):
    X = np.concatenate((SOS, np.zeros(length)))
    y = np.concatenate((SOS, np.ones(length)))
    data.append([X, y])

  batches = []
  for idx in range(0, len(data), batch_size):
    if idx + batch_size < len(data):
      batches.append(np.array(data[idx: idx + batch_size]).astype(np.int64))
  return batches


def train_loop(model, opt, loss_fn, dataloader):
  model.train()
  total_loss = 0

  for batch in dataloader:
    X, y = batch[:, 0], batch[:, 1]
    X, y = torch.tensor(X), torch.tensor(y)

    decode_input = y[:, :-1]
    target = y[:, 1:]

    pred = model(X, decode_input)

    pred_reshape = pred.reshape(-1, pred.shape[-1])
    loss = loss_fn(pred_reshape, target.flatten())

    opt.zero_grad()
    loss.backward()
    opt.step()

    total_loss += loss.detach().item()

  return total_loss / len(dataloader)


train_data = generate_random_data(5000)

epochs = 10
print("Training...")
for epoch in range(epochs):
  train_loss = train_loop(model, optimizer, loss_fn, train_data)
  print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}')
  torch.save(model.state_dict(), 'transformer.pt')


print("\nInference...")
x = torch.tensor(np.array([[2, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), dtype=torch.long)
print(f'In: {x}', 'Predict:', model.predict(x, SOS_token)[:, 1:])

x = torch.tensor(np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), dtype=torch.long)
print(f'In: {x}', 'Predict:', model.predict(x, SOS_token)[:, 1:])
```
