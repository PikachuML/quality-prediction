import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Blendmapping(nn.Module):
    def __init__(self, d_model, d_yc, d_y, N, heads, m, dropout):
        super().__init__()
        self.input = nn.Linear(m, d_model)
        self.encoder = MatEncoder(d_model, N, heads, dropout)
        self.Linear1 = nn.Linear(d_yc, d_model)
        self.output = nn.Linear(d_model, d_y)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, prop, yc):
        out = F.relu(self.input(src))
        out = self.encoder(out)
        out = torch.mul(out, prop.unsqueeze(2))
        out = torch.sum(out, dim=1)
        x = torch.sigmoid(self.Linear1(yc))
        out = torch.mul(out, x)
        out = self.dropout(out)
        out = self.output(out)
        return out


class MatEncoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(MatEncoderLayer(d_model, heads, dropout), N)
        self.d_model = d_model

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return x


class MatEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = Matricatt(heads, d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm_1(x)
        x = x + self.attn(x, x, x)
        x = self.norm_2(x)
        x = self.dropout_1(x)
        return x


class Matricatt(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.v_linear = nn.Linear(d_model, d_model)

        self.convkq = nn.Conv2d(1, self.d_k, 1)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        bs = q.size(0)

        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        ckq = self.convkq(k.unsqueeze(1))
        ckq = torch.sum(ckq, dim=2).view(bs, -1, self.h, self.d_k)
        ckq = ckq.transpose(1, 2)

        v = v.transpose(1, 2)

        scores = mat_attention(ckq, v, self.d_k, self.dropout) + v

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)
        output = torch.relu(output)
        return output


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def mat_attention(qk, v, d_k, dropout=None):
    scores = qk.transpose(-2, -1) / math.sqrt(d_k)

    if dropout is not None:
        scores = dropout(scores)

    scores = F.softmax(scores, dim=-2)
    output = torch.matmul(v, scores)
    return output
