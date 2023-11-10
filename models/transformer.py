
import copy
from collections import OrderedDict
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiSequential(nn.Sequential):
    """Sequential model with as many inputs as outputs."""

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def clones(module, n):
    """Produce n identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def seq_clones(module, seq_len):
    """Produce sequential model with seq_len identical layers."""

    return MultiSequential(
        OrderedDict(
            [('layer{}'.format(i), copy.deepcopy(module))
             for i in range(seq_len)]
        )
    )


class MultiHeadAttention(nn.Module):
    """Implement multi-head attention module.

    Args:
        model_dim: dimension of embedding.
        nnheads: number of attention nheads.
        mask: sets whether an input mask should be used.
    """

    def __init__(self, model_dim, nheads, p=0.1, mask=None):
        super().__init__()

        self.mask = mask
        self.nheads = nheads
        self.model_dim = model_dim

        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_out = nn.Linear(model_dim, model_dim)

        self.att = None

        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        """Compute multi-head attention forward pass.

        Args:
            query: tensor with shape (batch_size, sentence_len1, model_dim).
            key: tensor with shape (batch_size, sentence_len2, model_dim).
            value: tensor with shape (batch_size, sentence_len2, model_dim).

        Returns:
            tensor with shape (batch_size, sentence_len1, model_dim).
        """

        assert self.model_dim % self.nheads == 0

        key_dim = self.model_dim//self.nheads
        shape_q = query.shape[:2]+(self.nheads, key_dim)
        shape_k = key.shape[:2]+(self.nheads, key_dim)
        shape_v = value.shape[:2]+(self.nheads, key_dim)

        ret = self.attention(
            self.linear_q(query).reshape(shape_q),
            self.linear_k(key).reshape(shape_k),
            self.linear_v(value).reshape(shape_v)
        )
        ret = ret.reshape(ret.shape[:2] + (self.model_dim,))

        return self.dropout(self.linear_out(ret))

    def attention(self, query, key, value):
        """Compute scaled dot-product attention.

        Args:
            query: tensor with shape (batch_size, sentence_len1, nnheads, key_dim).
            key: tensor with shape (batch_size, sentence_len2, nnheads, key_dim).
            value: tensor with shape (batch_size, sentence_len2, nnheads, key_dim).

        Returns:
            tensor with shape (batch_size, sentence_len1, nnheads, key_dim).
        """

        score = torch.einsum('bqhd,bkhd->bhqk', query, key)
        if self.mask == 'triu':
            mask = torch.triu(
                torch.ones(score.shape, dtype=torch.bool), diagonal=1
            )
            score[mask] = -float('inf')

        if self.mask == 'diag':
            mask = torch.eye(
                n=score.shape[2], m=score.shape[3], dtype=torch.bool,
            )
            mask = mask.reshape(-1).repeat(
                (1, np.prod(score.shape[:2]))
            ).reshape(score.shape)

            score[mask] = -float('inf')

        self.att = F.softmax(score / np.sqrt(score.shape[-1]), dim=-1)
        ret = torch.einsum('bhqk,bkhd->bqhd', self.att, value)

        return ret


class Embedding(nn.Module):
    """Implement input and output embedding with tied weights."""

    def __init__(self, vocab_size, model_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.model_dim = model_dim

        self.encoder = nn.Embedding(vocab_size, model_dim)
        self.decoder = nn.Linear(model_dim, vocab_size, bias=False)

        self.decoder.weight = self.encoder.weight

    def forward(self, x, inverse=False):
        if inverse:
            return self.decoder(x)

        return self.encoder(x) * np.sqrt(self.model_dim)


class PositionalEncoder(nn.Module):
    """Implement the Positional Encoder function.

    Based on https://nlp.seas.harvard.edu/2018/04/03/attention.html.
    """

    def __init__(self, d_model, max_len=5000, p=0.1):
        super().__init__()

        # Compute the positional encodings once in log space.
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pe', pos_enc)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class DecoderLayer(nn.Module):
    """Implement decoder sub-layers."""

    def __init__(self, model_dim, hidden_dim, nnheads, p=0.1):
        super().__init__()
        self.mhatt_masked = MultiHeadAttention(
            model_dim, nnheads, p, mask='triu',
        )
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(p=p),
        )

        self.layer_norms = clones(nn.LayerNorm(model_dim), 2)

    def forward(self, inputs):
        inputs_normalized = self.layer_norms[0](inputs)
        att = self.mhatt_masked(inputs_normalized, inputs_normalized, inputs_normalized) + inputs_normalized

        att_normalized = self.layer_norms[1](att)
        out = self.ffn(att_normalized) + att_normalized

        return out


# class GPT2(nn.Module):
#     """Implement GPT-2 model.

#     Args:
#         vocab_size: number of unique tokens in vocabulary.
#         model_dim: dimension of embedding.
#         hidden_dim: size of hidden layer in feed forward sub-layers.
#         nnheads: number of attention nheads.
#         max_len: maximum sentence length used to pre-compute positional encoder.
#         depth: number of decoder sub-layers.
#     """

#     def __init__(
#             self,
#             vocab_size,
#             model_dim,
#             hidden_dim,
#             nnheads,
#             depth,
#             p=0.1,
#             max_len=5000,
#     ):
#         super().__init__()
#         self.embedding = Embedding(vocab_size, model_dim)
#         self.pos_enc = PositionalEncoder(model_dim, max_len, p)

#         self.decoder = seq_clones(
#             DecoderLayer(model_dim, hidden_dim, nnheads, p), depth,
#         )
#         self.layer_norm = nn.LayerNorm(model_dim)

#         self.apply(self._init_weights)
#         self.inputs_embedding = None

#     def forward(self, inputs):
#         self.inputs_embedding = self.embedding(inputs)
#         input_pe = self.pos_enc(self.inputs_embedding)

#         dec = self.decoder(input_pe)
#         dec_normalized = self.layer_norm(dec)

#         return self.embedding(dec_normalized, inverse=True)

#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 module.bias.data.fill_(0.1)

#         elif isinstance(module, nn.Embedding):
#             nn.init.xavier_uniform_(module.weight)