import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeAligner(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_types: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_types, latent_dim))

    def forward(self, x):
        z = F.normalize(self.encoder(x), dim=-1)
        logits = z @ F.normalize(self.prototypes, dim=-1).t() / 0.07
        return z, logits

    def loss(self, logits, pseudo_labels, mask):
        return F.cross_entropy(logits[mask], pseudo_labels[mask])
