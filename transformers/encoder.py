import torch.nn as nn

from embedding import PositionalEmbedding, Embedding
from multi_head_attention import MultiHeadAttention


class EncodeLayer(nn.Module):
  def __init__(self, embed_dim, expansion_factor=4, n_heads=8, drop_prob=0.1):
    """
    Implements an encoder layer for a transformer model. The encoder layer consists of a multi-head attention mechanism
      followed by a feed-forward neural network, with layer normalization and dropout applied after each component.

    The attention mechanism takes in key, query, and value tensors, and computes the weighted sum of the value
      tensor based on the attention scores. The feed-forward neural network applies a simple feed-forward transformation
        to the output of the attention mechanism.

    The `EncodeLayer` class provides an implementation of this encoder layer, with configurable embedding dimension,
      expansion factor for the feed-forward layer, and number of attention heads.

    Args:
      embed_dim: The dimension of the input and output embeddings.
      expansion_factor: The expansion factor for the feed-forward neural network.
      n_heads: The number of attention heads.
      drop_prob: The dropout probability.
    """
    super(EncodeLayer, self).__init__()

    self.attention = MultiHeadAttention(embed_dim, n_heads)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.dropout1 = nn.Dropout(drop_prob)
    self.feed_forward = nn.Sequential(
        nn.Linear(embed_dim, expansion_factor * embed_dim),
        nn.ReLU(),
        # nn.GELU(),
        nn.Linear(expansion_factor * embed_dim, embed_dim)
    )
    self.dropout2 = nn.Dropout(drop_prob)
    self.norm2 = nn.LayerNorm(embed_dim)

  def forward(self, key, query, value, mask=None):
    attention_out = self.attention(key, query, value, mask)
    value = attention_out.expand_as(attention_out)
    attention_residual_out = attention_out + value
    x1 = self.dropout1(self.norm1(attention_residual_out))

    feed_fwd_residual_out = self.feed_forward(x1) + x1
    x2 = self.dropout2(self.norm2(feed_fwd_residual_out))
    return x2


class TransformerEncoder(nn.Module):
  def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, drop_prob=0.1):
    """
    Initializes a TransformerEncoder module, which is a stack of EncodeLayer modules.

    Args:
        seq_len (int): The maximum sequence length of the input.
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimension of the input and output embeddings.
        num_layers (int, optional): The number of EncodeLayer modules to stack. Defaults to 2.
        expansion_factor (int, optional): The expansion factor for the feed-forward neural network in each EncodeLayer. Defaults to 4.
        n_heads (int, optional): The number of attention heads in each EncodeLayer. Defaults to 8.
        drop_prob (float, optional): The dropout probability. Defaults to 0.1.
    """
    super(TransformerEncoder, self).__init__()

    self.embedding_layer = Embedding(vocab_size, embed_dim)
    self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

    self.layers = nn.ModuleList([
      EncodeLayer(embed_dim, expansion_factor, n_heads, drop_prob) for i in range(num_layers)
    ])

  def forward(self, x, mask=None):
    """
    Applies the TransformerEncoder module to the input tensor `x`.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
        mask (torch.Tensor, optional): A mask tensor of shape (batch_size, sequence_length) to apply to the attention layers.

    Returns:
        torch.Tensor: The output tensor of shape (batch_size, sequence_length, embed_dim).
    """
    embed_out = self.embedding_layer(x)
    out = self.positional_encoder(embed_out)
    for layer in self.layers:
      out = layer(out, out, out, mask)
    return out  # [batch_size, seq_lengh, embed_dim]
