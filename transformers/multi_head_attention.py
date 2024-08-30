import torch.nn as nn

from scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim=512, n_heads=8):
    """
    Implements a multi-head attention mechanism as described in the Transformer paper.

    The multi-head attention mechanism allows the model to jointly attend to information
    from different representation subspaces at different positions. It concatenates the
    outputs of multiple attention heads and projects the concatenated output to produce
    the final output.

    Args:
        embed_dim (int): The dimensionality of the input embeddings.
        n_heads (int): The number of attention heads to use.

    Attributes:
        attention (ScaleDotProductAttention): The attention mechanism used by each head.
        Q (nn.Linear): The linear layer that computes the query vectors.
        K (nn.Linear): The linear layer that computes the key vectors.
        V (nn.Linear): The linear layer that computes the value vectors.
        out (nn.Linear): The linear layer that projects the concatenated attention outputs.
    """
    super(MultiHeadAttention, self).__init__()

    self.embed_dim = embed_dim
    self.n_heads = n_heads
    self.single_head_dim = embed_dim // n_heads

    self.attention = ScaleDotProductAttention()
    self.Q = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
    self.K = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
    self.V = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
    self.out = nn.Linear(n_heads * self.single_head_dim, self.embed_dim)

  def forward(self, key, query, value, mask=None):
    batch_size = key.size(0)
    seq_length = key.size(1)
    seq_length_query = query.size(1)

    # split into n_heads
    # [batch_size, sequence_length, n_heads, single_head_dim]
    key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
    query = query.view(batch_size, seq_length_query,
                       self.n_heads, self.single_head_dim)
    value = value.view(batch_size, seq_length,
                       self.n_heads, self.single_head_dim)

    k = self.K(key)
    q = self.Q(query)
    v = self.V(value)

    q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
    k = k.transpose(1, 2)  # ...
    v = v.transpose(1, 2)  # ...

    # computes attention
    scores = self.attention(q, k, v, mask)

    # concatenating heads
    concat = scores.permute(0, 2, 1, 3).reshape(
      batch_size, seq_length_query, -1)
    output = self.out(concat)

    return output
