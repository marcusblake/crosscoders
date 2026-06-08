import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

    def encode(self, x: torch.tensor) -> torch.tensor:
        """Encodes the input tensor into a hidden representation.

        Args:
            x (torch.tensor): A [batch_size, input_dim] tensor to be encoded.

        Returns:
            torch.tensor: A [batch_size, hidden_dim] encoding of the input tensor.
        """
        return self.encoder(x)

    def decode(self, x: torch.tensor) -> torch.tensor:
        """Decodes the hidden representation back into the input space.

        Args:
            x (torch.tensor): A [batch_size, hidden_dim] tensor to be decoded.

        Returns:
            torch.tensor: A [batch_size, input_dim] reconstruction of the input tensor.
        """
        return self.decoder(x)

    def forward(self, x: torch.tensor):
        h = self.encode(x)
        return self.decode(h)
