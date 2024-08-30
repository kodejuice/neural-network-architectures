import torch
import torch.nn as nn

from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Transformer(nn.Module):
  def __init__(self, embed_dim, src_vocab_size, target_vocab_size, src_seq_len, trg_seq_len=None, pad_token=None, num_encode_layers=2, num_decode_layers=2, expansion_factor=4, n_heads=8, dropout=0.1):
    """
    Initializes a Transformer model with the specified parameters.

    Args:
        embed_dim (int): The size of the embedding dimension.
        src_vocab_size (int): The size of the source vocabulary.
        target_vocab_size (int): The size of the target vocabulary.
        src_seq_len (int): The maximum length of the source sequence.
        trg_seq_len (int, optional): The maximum length of the target sequence. If not provided, it defaults to the source sequence length.
        pad_token (int, optional): The integer value representing the padding token. Defaults to 0.
        num_encode_layers (int, optional): The number of encoder layers. Defaults to 2.
        num_decode_layers (int, optional): The number of decoder layers. Defaults to 2.
        expansion_factor (int, optional): The expansion factor for the feed-forward network. Defaults to 4.
        n_heads (int, optional): The number of attention heads. Defaults to 8.
        dropout (float, optional): The dropout rate for the feed-forward network. Defaults to 0.1.
    """
    super(Transformer, self).__init__()
    if trg_seq_len is None:
      trg_seq_len = src_seq_len
    self.trg_seq_len = trg_seq_len
    self.pad_token = pad_token
    self.target_vocab_size = target_vocab_size
    self.encoder = TransformerEncoder(src_seq_len, src_vocab_size, embed_dim,
                                      num_layers=num_encode_layers, expansion_factor=expansion_factor, n_heads=n_heads, drop_prob=dropout)
    self.decoder = TransformerDecoder(target_vocab_size, embed_dim, trg_seq_len,
                                      num_layers=num_decode_layers, expansion_factor=expansion_factor, n_heads=n_heads, drop_prob=dropout)

  def make_pad_mask(self, matrix: torch.tensor, pad_token) -> torch.tensor:
    """
    Creates a padding mask tensor that identifies the padding tokens in the input matrix.

    The mask is created by iterating over the rows of the input matrix, finding the first padding token in each row,
      and setting all elements after that token to 0.
    This ensures that the model will not attend to the padding tokens during the attention computation.

    Args:
        matrix (torch.Tensor): The input tensor to create the mask for.
        pad_token: The integer value representing the padding token.

    Returns:
        torch.Tensor: The padding mask tensor.

    For [1, 2, 3, PAD, PAD], the mask will be:
    [[1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]]
    """
    batch_size, seq_len = matrix.shape

    # if no pad token, return matrix of 1's
    if pad_token is None:
      return torch.ones(batch_size, seq_len, seq_len)

    masks = torch.tensor([])
    for i in range(batch_size):
      seq = matrix[i]
      pad_indices = torch.nonzero(seq == pad_token)
      if pad_indices.shape[0] == 0:
        first_pad_index = seq_len
      else:
        first_pad_index = pad_indices[0][0]
      mask = torch.ones(seq_len, seq_len)
      mask[:, first_pad_index:] = 0
      mask[first_pad_index:, :] = 0
      masks = torch.cat((masks, mask.unsqueeze(0)), dim=0)
    return masks

  def make_trg_mask(self, trg):
    """
    Creates a target mask tensor that ensures the model only attends to previous tokens in the sequence.
    The mask is created by taking the lower triangular matrix of ones and expanding it to match the batch size and sequence length
      of the target tensor.
    This ensures that each position in the target sequence can only attend to positions that come before it in the sequence.
    For example, if the target sequence is [1, 2, 3, 4], the mask will be:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]

    Args:
        trg (torch.Tensor): The target tensor to create the mask for.
    """
    batch_size, trg_len = trg.shape
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        batch_size, 1, trg_len, trg_len
    )
    return trg_mask

  def predict(self, src, SOS_TOKEN, EOS_TOKEN=None):
    """
    Generates a prediction sequence from the given source input `src` using the Transformer model.

    Args:
        src (torch.Tensor): The source input tensor.
        SOS_TOKEN (int): The integer value representing the start-of-sequence token.
        EOS_TOKEN (int, optional): The integer value representing the end-of-sequence token. If provided, the generated sequence will stop when this token is predicted.

    Returns:
        torch.Tensor: The predicted output sequence.
    """

    enc_out = self.encoder(src)

    trg_len = self.trg_seq_len
    trg_mask = self.make_trg_mask(torch.zeros((1, trg_len), dtype=torch.long))
    out_labels = torch.tensor([[SOS_TOKEN]], dtype=torch.long)
    for _ in range(trg_len):
      # pad out[] with pad_token to target length
      out = torch.nn.functional.pad(
        out_labels, (0, trg_len - out_labels.size(1)), value=self.pad_token)

      out_mask = trg_mask * self.make_pad_mask(out, self.pad_token)
      out = self.decoder(out, enc_out, out_mask)  # bs x seq_len x vocab_dim

      # taking the last token
      out = out[:, -1, :]
      out = out.argmax(-1).unsqueeze(0)
      # add to output
      out_labels = torch.cat([out_labels, out], dim=1)

      if EOS_TOKEN is not None:
        if out_labels[0, -1] == EOS_TOKEN:
          break

    return out_labels

  def forward(self, src, trg):
    """
    Performs a forward pass through the Transformer model.

    Args:
        src (torch.Tensor): The source input tensor.
        trg (torch.Tensor): The target input tensor.

    Returns:
        torch.Tensor: The output tensor from the Transformer model.
    """

    # Create target mask by combining sequence mask and padding mask
    # This ensures the model attends only to previous tokens and ignores padding
    trg_mask = self.make_trg_mask(trg)
    # trg_pad_mask = self.make_pad_mask(trg, self.pad_token)
    # trg_mask = trg_mask * trg_pad_mask

    # src pad mask ensures the model ignores padding
    # src_pad_mask = self.make_pad_mask(src, self.pad_token)
    # enc_out = self.encoder(src, src_pad_mask)

    enc_out = self.encoder(src)
    outputs = self.decoder(trg, enc_out, trg_mask)
    return outputs
