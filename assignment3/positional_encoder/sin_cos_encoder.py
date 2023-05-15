import math
import torch
from torch import nn, Tensor


class SinCosTextEncoder(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, init_range):
        super().__init__()
        
        # TODO [part 2a]
        # define the encoder

        ############# YOUR CODE HERE #############
        pass
        ##########################################

        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.d_model = d_model

    def forward(self, src: Tensor):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.encoder(src) * math.sqrt(self.d_model)


class SinCosPosEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()

        # TODO [part 2a]
        # define ``positional_encoding`` as described in the GoogleDoc instruction and register it 

        ############# YOUR CODE HERE #############
        pass
        ##########################################

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.positional_encoding[:x.size(0)]
        return self.dropout(x)
