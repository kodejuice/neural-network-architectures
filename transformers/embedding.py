import torch
import math
import torch.nn as nn


class Embedding(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    """
    Initializes an Embedding module with the given vocabulary size and embedding dimension.
    Args:
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimension of the embedding vectors.
    """
    super(Embedding, self).__init__()
    self.embed = nn.Embedding(vocab_size, embed_dim)

  def forward(self, x):
    """
    Applies the embedding layer to the input tensor `x` and returns the output.
    Args:
        x (torch.Tensor): The input tensor of shape `(batch_size, sequence_length)`.
    Returns:
        torch.Tensor: The output tensor of shape `(batch_size, sequence_length, embed_dim)` after applying the embedding layer.
    """
    out = self.embed(x)
    return out


class PositionalEmbedding(nn.Module):
  def __init__(self, max_seq_len, embed_model_dim):
    """
    Initializes a PositionalEmbedding module that generates positional embeddings for a given maximum sequence length and embedding dimension.

    Args:
        max_seq_len (int): The maximum sequence length for which positional embeddings will be generated.
        embed_model_dim (int): The dimension of the embedding vectors.

    The positional embeddings are generated using sine and cosine functions, as described in the Transformer paper.
      The resulting tensor is registered as a non-trainable parameter of the module.
    """
    super(PositionalEmbedding, self).__init__()
    self.embed_dim = embed_model_dim

    position = torch.arange(0, max_seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.embed_dim, 2)
                         * -(math.log(10000.0) / self.embed_dim))
    pe = torch.zeros(max_seq_len, self.embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=False)

  def forward(self, x):
    # Scale embeddings
    x = x * math.sqrt(self.embed_dim)
    # Add positional encoding
    seq_len = x.size(1)
    x = x + self.pe[:, :seq_len]
    return x
