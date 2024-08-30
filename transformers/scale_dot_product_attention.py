import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleDotProductAttention(nn.Module):
  """
  Implements the scaled dot-product attention mechanism as described in the Transformer paper.

  The forward method takes in query, key, and value tensors, as well as an optional mask tensor, and returns the attention scores.

  Args:
      q (torch.Tensor): The query tensor of shape `(batch_size, n_heads, seq_length, single_head_dim)`.
      k (torch.Tensor): The key tensor of shape `(batch_size, n_heads, seq_length, single_head_dim)`.
      v (torch.Tensor): The value tensor of shape `(batch_size, n_heads, seq_length, single_head_dim)`.
      mask (torch.Tensor, optional): The mask tensor of shape `(batch_size, seq_length)`.

  Returns:
      torch.Tensor: The attention scores tensor of shape `(batch_size, n_heads, seq_length, single_head_dim)`.
  """

  def __init__(self):
    super(ScaleDotProductAttention, self).__init__()
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, q, k, v, mask=None):
    # q: [batch_size, n_heads, seq_length, single_head_dim]
    # k: [batch_size, n_heads, seq_length, single_head_dim]
    # v: [batch_size, n_heads, seq_length, single_head_dim]
    # mask: [batch_size, seq_length]
    batch_size, heads, seq_length, dim = k.size()

    # [batch_size, n_heads, single_head_dim, seq_ken]
    k_t = k.transpose(-1, -2)

    product = torch.matmul(q, k_t) / math.sqrt(dim)

    # to prevent giving attention to masked out posititons
    if mask is not None:
      product = product.masked_fill(mask == 0, float("-1e20"))

    # applying softmax
    scores = F.softmax(product, dim=-1)

    # mutiply with value matrix
    scores = torch.matmul(scores, v)

    return scores
