import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

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
        x = x + self.scale * self.pe[: x.size(0), :]
        return self.dropout(x)


@dataclass
class TransformerConfig:
    src_vocab_size: int = 13000
    tgt_vocab_size: int = 13000
    hidden_dim: int = 128
    encoder_layers: int = 1  
    decoder_layers: int = 1
    dropout: int = 0.15
    nheads: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.encoder_embd = nn.Embedding(config.src_vocab_size, self.config.hidden_dim)
        self.pos_enc_embd = PositionalEncoding(
            self.config.hidden_dim, self.config.dropout
        )

        self.decoder_embd = nn.Embedding(
            self.config.tgt_vocab_size, self.config.hidden_dim
        )
        self.pos_dec_embd = PositionalEncoding(
            self.config.hidden_dim, self.config.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            self.config.hidden_dim,
            self.config.nheads,
            self.config.hidden_dim * self.config.nheads,
            self.config.dropout,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, self.config.encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            self.config.hidden_dim,
            self.config.nheads,
            self.config.hidden_dim * self.config.nheads,
            self.config.dropout,
            activation="relu",
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, self.config.decoder_layers
        )

        self.fc = nn.Linear(self.config.hidden_dim, self.config.tgt_vocab_size)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz, sz1=None):
        if sz1 == None:
            mask = torch.triu(torch.ones(sz, sz), 1)
        else:
            mask = torch.triu(torch.ones(sz, sz1), 1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def make_enc_mask(self, src):
        return (src == 0).transpose(1, 0)

    def make_dec_mask(self, target):
        return (target == 0).transpose(1, 0)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(
                self.config.device
            )
        src_pad_mask = self.make_enc_mask(src)
        trg_pad_mask = self.make_dec_mask(trg)

        src = self.encoder_embd(src)
        src = self.pos_enc_embd(src)

        trg = self.decoder_embd(trg)
        trg = self.pos_dec_embd(trg)

        memory = self.transformer_encoder(src, None, src_pad_mask)
        output = self.transformer_decoder(
            trg, memory, self.trg_mask, None, trg_pad_mask, src_pad_mask
        )

        output = self.fc(output)

        return output
