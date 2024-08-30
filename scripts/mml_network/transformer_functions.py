#!/usr/bin/env python3
import torch
from torch import nn

import math

device = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, in_dim=2, n_embed=10, num_layers=1, n_head=1, dropout=0.01):
        super().__init__()
        self.model_type = "Transformer"

        self.src_mask = None
        self.embed = nn.Linear(in_dim, n_embed)
        self.pos_encoder = nn.Embedding(10, n_embed).to(device)
        # self.pos_encoder = PositionalEncoding(d_model=in_dim, max_len=10)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embed, nhead=n_head, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Linear(n_embed, in_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.view(-1, 10, 2)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.embed.to(src.device)(src)
        # src = self.pos_encoder.to(src.device)(src)
        pos_emb = self.pos_encoder(torch.arange(10, device=device))  # (T,C)
        src = src + pos_emb

        output = self.transformer_encoder.to(src.device)(src, self.src_mask)
        output = self.decoder.to(src.device)(output)
        output = output[:, -1, :]  # only return the last time step
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
