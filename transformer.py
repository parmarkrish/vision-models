import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, attn_mask=None):
        super().__init__()
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_mask = attn_mask
        self.num_heads = num_heads
        self.scaling_factor = math.sqrt(hidden_dim / num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        N, S, D = query.shape
        N, T, D = value.shape
        H = self.num_heads

        # (N, S, D) @ (D, D) -> (N, S, D) -> (N, S, H, D/H) -> (N, H, S, D/H)
        Q = self.query(query).view(N, S, H, D//H).transpose(1, 2)
        # (N, T, D) @ (D, D) -> (N, T, D) -> (N, T, H, D/H) -> (N, H, T, D/H)
        K = self.key(key).view(N, T, H, D//H).transpose(1, 2)
        # (N, T, D) @ (D, D) -> (N, T, D) -> (N, T, H, D/H) -> (N, H, T, D/H)
        V = self.value(value).view(N, T, H, D//H).transpose(1, 2)

        # (N, H, S, D/H) @ (N, H, D/H, T) -> (N, H, S, T)
        Y = torch.matmul(Q, K.transpose(2, 3)) / self.scaling_factor

        # We apply value -inf so that softmax for that element equals 0
        if self.attn_mask is not None:
            Y = Y.masked_fill(self.attn_mask==0, float("-inf"))

        # (N, H, S, T) @ (N, H, T, D/H) -> (N, H, S, D/H)
        Y = self.dropout(F.softmax(Y, dim=-1)) @ V
        # (N, S, H, D/H) -> (N, S, D) @ (D, D) -> (N, S, D)
        out = self.proj(Y.transpose(1, 2).reshape(N, S, D))
        return out

class MLP(nn.module):
    def __init__(self, hidden_dim, mlp_size, act_fn=nn.GELU(inplace=True)):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, mlp_size)
        self.w2 = nn.Linear(mlp_size, hidden_dim)
        self.act_fn = act_fn

    def forward(self, x):
        x = self.act_fn(x @ self.w1)
        out = x @ self.w2
        return out

class Block(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        attn = MultiHeadAttention(hidden_dim, num_heads)
        # for self attention, query, key & value are one and the same
        self.msa = lambda x: attn(x, x, x) 
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_size)

    def forward(self, x):
        x = self.msa(self.ln1(x)) + x
        out = self.mlp(self.ln2(x)) + x
        return out

class Transformer(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, mlp_size):
        self.layers = nn.Sequential(*[Block(hidden_dim, num_heads, mlp_size)
                                      for i in range(num_layers)])
        self.ln = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        x = self.layers(x)
        out = self.ln(x)
        return out