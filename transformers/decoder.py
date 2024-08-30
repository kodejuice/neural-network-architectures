import torch.nn as nn

from embedding import PositionalEmbedding
from encoder import EncodeLayer
from multi_head_attention import MultiHeadAttention


class DecodeBlock(nn.Module):
  def __init__(self, embed_dim, expansion_factor=4, n_heads=8, drop_prob=0.1):
    """
    Initializes a DecodeBlock module, which is a component of the Transformer Decoder.

    The DecodeBlock module contains a MultiHeadAttention layer, a LayerNorm layer, a Dropout layer, and an EncodeLayer module. These components are used to process the input sequence and produce the output sequence for the Transformer Decoder.

    Args:
        embed_dim (int): The dimensionality of the input and output embeddings.
        expansion_factor (int, optional): The expansion factor for the feed-forward network in the EncodeLayer module. Defaults to 4.
        n_heads (int, optional): The number of attention heads in the MultiHeadAttention layer. Defaults to 8.
        drop_prob (float, optional): The dropout probability for the Dropout layer. Defaults to 0.1.
    """
    super(DecodeBlock, self).__init__()
    self.attention = MultiHeadAttention(embed_dim, n_heads=n_heads)
    self.norm = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(drop_prob)
    self.encode_block = EncodeLayer(embed_dim, expansion_factor, n_heads)

  def forward(self, encode_output, x, mask):
    """
    Applies the DecodeBlock module to the input sequence and encoder output.

    Args:
        encode_output (torch.Tensor): The output of the encoder module.
        x (torch.Tensor): The input sequence to the decoder module.
        mask (torch.Tensor, optional): A mask tensor to be applied to the attention weights.

    Returns:
        torch.Tensor: The output of the DecodeBlock module.
    """
    attention = self.attention(x, x, x, mask=mask)
    query = self.dropout(self.norm(attention + x))
    key = encode_output
    value = encode_output
    out = self.encode_block(key, query, value)
    return out


class TransformerDecoder(nn.Module):
  def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8, drop_prob=0.1):
    """
    Initializes a TransformerDecoder module, which is the decoder component of a Transformer model.

    The TransformerDecoder module takes in a target vocabulary size, embedding dimension, sequence length, and optional parameters
      for the number of layers, expansion factor, number of attention heads, and dropout probability.
    It creates an Embedding layer for the target vocabulary, a PositionalEmbedding layer, and a series of DecodeBlock modules stacked
      in an nn.ModuleList.
    The output of the decoder is passed through a final linear layer to produce the logits.

    Args:
        target_vocab_size (int): The size of the target vocabulary.
        embed_dim (int): The dimensionality of the input and output embeddings.
        seq_len (int): The maximum sequence length for the input.
        num_layers (int, optional): The number of DecodeBlock layers in the decoder. Defaults to 2.
        expansion_factor (int, optional): The expansion factor for the feed-forward network in the EncodeLayer module. Defaults to 4.
        n_heads (int, optional): The number of attention heads in the MultiHeadAttention layer. Defaults to 8.
        drop_prob (float, optional): The dropout probability for the Dropout layer. Defaults to 0.1.
    """
    super(TransformerDecoder, self).__init__()
    self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
    self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

    self.layers = nn.ModuleList([
      DecodeBlock(embed_dim, expansion_factor=expansion_factor, n_heads=n_heads, drop_prob=drop_prob) for _ in range(num_layers)
    ])
    self.linear = nn.Linear(embed_dim, target_vocab_size)
    self.dropout = nn.Dropout(drop_prob)

  def forward(self, x, enc_out, mask=None):
    """
    Applies the TransformerDecoder module to the input sequence `x` and the encoder output `enc_out`, producing the output logits.

    Args:
        x (torch.Tensor): The input sequence to the decoder module.
        enc_out (torch.Tensor): The output of the encoder module.
        mask (torch.Tensor, optional): A mask tensor to be applied to the attention weights.

    Returns:
        torch.Tensor: The output logits of the TransformerDecoder module.
    """
    x = self.word_embedding(x)
    x = self.position_embedding(x)
    x = self.dropout(x)

    for layer in self.layers:
      x = layer(enc_out, x, mask)

    # raw logits, no softmax applied
    out = self.linear(x)

    return out
