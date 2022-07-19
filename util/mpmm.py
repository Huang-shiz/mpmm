import torch
from torch import nn
import math
import torch.nn.functional as F
from fastNLP.embeddings.embedding import TokenEmbedding, Embedding
from torch.nn import LayerNorm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class MPMM(nn.Module):
    def __init__(self, embed, hidden_size, num_classes, dropout):
        super(MPMM, self).__init__()
        if isinstance(embed, TokenEmbedding) or isinstance(embed, Embedding):
            self.embedding = embed
        else:
            self.embedding = Embedding(embed)
        self.p = 10
        self.p_dim = self.p * 4
        self.encoder = ARNN(self.embedding.embed_size, hidden_size, dropout=dropout)
        self.match = Multi_Match(hidden_size, dropout=dropout, p=self.p)
        self.fusion = Fusion(self.p_dim, hidden_size, dropout)
        self.pooling = Pooling()
        self.prediction = Prediction(hidden_size*4, hidden_size, num_classes, dropout)


    def forward(self, words1, words2):
        a_emb = self.embedding(words1)
        b_emb = self.embedding(words2)
        mask_a = torch.ne(a_emb, 0)[:,:,0].unsqueeze(2)
        mask_b = torch.ne(b_emb, 0)[:,:,0].unsqueeze(2)

        a_enc = self.encoder(a_emb, mask_a)
        b_enc = self.encoder(b_emb, mask_b)

        ia, ib, m_a, m_b = self.match(a_enc, b_enc, mask_a, mask_b)

        a_fus = self.fusion(a_enc, ia, m_a)
        b_fus = self.fusion(b_enc, ib, m_b)

        a_pool = self.pooling(a_fus)
        b_pool = self.pooling(b_fus)

        return self.prediction(a_pool, b_pool)


class ARNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(ARNN, self).__init__()
        self.hidden_size = hidden_size
        self.ffn1 = Linear(input_size, 50)
        self.ffn2 = Linear(input_size, 50)
        self.ffn3 = Linear(input_size, 50)
        self.ffn4 = nn.Sequential(
            Linear(50, 50, activations=True),
            nn.Dropout(dropout),
        )
        self.rnn = BiLSTM(input_size,hidden_size,dropout=dropout)
        self.ffn = nn.Sequential(
            Linear(hidden_size*2+50, hidden_size, activations=True),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        q = self.ffn1(x)
        k = self.ffn2(x)
        v = self.ffn3(x)
        mat_qk = torch.matmul(q, k.transpose(1, 2))/ math.sqrt(self.hidden_size)
        
        if mask is not None:
            mat_qk = mat_qk + mask * -1e9
        attn = F.softmax(mat_qk, dim=-1)
        mat_av = torch.matmul(attn, v)

        _golbal = self.ffn4(mat_av)
        _local, _ = self.rnn(x)

        loc_glo = torch.cat((_local, _golbal),dim=-1)
        outputs = self.ffn(loc_glo)

        return outputs

class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout):
        super(BiLSTM, self).__init__()
        self._encoder = nn.LSTM(input_size, hidden_size, dropout=dropout,
                                 num_layers=1,bias=True,batch_first=True,bidirectional=True)
        nn.init.normal_(self._encoder.weight_ih_l0, std=math.sqrt(1. / hidden_size))
        nn.init.zeros_(self._encoder.bias_ih_l0)
        nn.init.normal_(self._encoder.weight_hh_l0, std=math.sqrt(1. / hidden_size))
        nn.init.zeros_(self._encoder.bias_hh_l0)

        nn.init.normal_(self._encoder.weight_ih_l0_reverse, std=math.sqrt(1. / hidden_size))
        nn.init.zeros_(self._encoder.bias_ih_l0_reverse)
        nn.init.normal_(self._encoder.weight_hh_l0_reverse, std=math.sqrt(1. / hidden_size))
        nn.init.zeros_(self._encoder.bias_hh_l0_reverse)
    

    def forward(self, x):
        outputs = self._encoder(x)
        return outputs


class Multi_Match(nn.Module):
    def __init__(self, input_size, dropout, p):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(input_size)))
        self.w1 = nn.Parameter(torch.rand(p, input_size))
        self.w2 = nn.Parameter(torch.rand(p, input_size))
        self.w3 = nn.Parameter(torch.rand(p, input_size))
        self.w4 = nn.Parameter(torch.rand(p, input_size))
        self.dropout = nn.Dropout(dropout)

    def soft_attn(self, a, b, mask_a, mask_b):
        attn = torch.matmul(a, b.transpose(1, 2)) * self.temperature
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).bool()   # .byte()
        attn.masked_fill_(~mask, -1e7)
        attn_a = F.softmax(attn, dim=1)
        attn_b = F.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        return feature_a, feature_b

    def full_matching(self, a, b, w):
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
        s11 = w * torch.stack([a] * w.size(3), dim=3)
        seq_len1 = a.size(1)
        s12 = w * torch.stack([torch.stack([b[:,-1,:]] * seq_len1, dim=1)] * w.size(3), dim=3)
        cos_p1 = F.cosine_similarity(s11, s12, dim=2)

        s21 = w * torch.stack([b] * w.size(3), dim=3)
        seq_len2 = b.size(1)
        s22 = w * torch.stack([torch.stack([a[:,-1,:]] * seq_len2, dim=1)] * w.size(3), dim=3)
        cos_p2 = F.cosine_similarity(s21, s22, dim=2)

        return cos_p1, cos_p2

    def maxpooling_matching(self, a, b, w):
        w = w.unsqueeze(0).unsqueeze(2)
        s1, s2 = w * torch.stack([a] * w.size(1), dim=1), w * torch.stack([b] * w.size(1), dim=1)
        s1_norm = s1.norm(p=2, dim=3, keepdim=True)
        s2_norm = s2.norm(p=2, dim=3, keepdim=True)
        attn = torch.matmul(s1, s2.transpose(2, 3))
        norm = torch.matmul(s1_norm, s2_norm.transpose(2, 3))
        cos_p = torch.div(attn, norm+1e-8).permute(0, 2, 3, 1)
        cos_p1 = cos_p.max(dim=2)[0]
        cos_p2 = cos_p.max(dim=1)[0]
        return cos_p1, cos_p2

    def attentive_matching(self, a, b, w):
        s1_norm = a.norm(p=2, dim=2, keepdim=True)
        s2_norm = b.norm(p=2, dim=2, keepdim=True)
        attn = torch.matmul(a, b.transpose(1, 2))
        norm = torch.matmul(s1_norm, s2_norm.transpose(1, 2))
        alpha = torch.div(attn, norm + 1e-8)

        _s2 = b.unsqueeze(1) * alpha.unsqueeze(3)
        s2_mean = torch.div(_s2.sum(dim=2), alpha.sum(dim=2, keepdim=True))
        _s1 = a.unsqueeze(2) * alpha.unsqueeze(3)
        s1_mean = torch.div(_s1.sum(dim=1), alpha.sum(dim=1, keepdim=True).transpose(1, 2))

        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
        s1 = w * torch.stack([a] * w.size(3), dim=3)
        s2_mean = w * torch.stack([s2_mean] * w.size(3), dim=3)
        s2 = w * torch.stack([b] * w.size(3), dim=3)
        s1_mean = w * torch.stack([s1_mean] * w.size(3), dim=3)

        cos_p1 = F.cosine_similarity(s1, s2_mean, dim=2)
        cos_p2 = F.cosine_similarity(s2, s1_mean, dim=2)
        return cos_p1, cos_p2

    def max_attentive_matching(self, a, b, w):
        s1_norm = a.norm(p=2, dim=2, keepdim=True)
        s2_norm = b.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)
        attn = torch.matmul(a, b.permute(0, 2, 1))
        norm = s1_norm * s2_norm
        alpha = torch.div(attn, norm+1e-8)
        alpha_max2 = alpha.max(dim=2)[0].unsqueeze(2)
        alpha_max1 = alpha.max(dim=1)[0].unsqueeze(2)

        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
        s1 = w * torch.stack([a] * w.size(3), dim=3)
        s2_ma = w * torch.stack([alpha_max2] * w.size(3), dim=3)
        s2 = w * torch.stack([b] * w.size(3), dim=3)
        s1_ma = w * torch.stack([alpha_max1] * w.size(3), dim=3)

        cos_p1 = F.cosine_similarity(s1, s2_ma, dim=2)
        cos_p2 = F.cosine_similarity(s2, s1_ma, dim=2)
        return cos_p1, cos_p2

    def forward(self, a, b, mask_a, mask_b):
        ia, ib = self.soft_attn(a, b, mask_a, mask_b)
        fm_a, fm_b = self.full_matching(a, b, self.w1)
        mm_a, mm_b = self.maxpooling_matching(a, b, self.w2)
        am_a, am_b = self.attentive_matching(a, b, self.w3)
        mam_a, mam_b = self.max_attentive_matching(a, b, self.w4)
        m_a = torch.cat((fm_a, mm_a, am_a, mam_a), dim=-1)
        m_b = torch.cat((fm_b, mm_b, am_b, mam_b), dim=-1)
        m_a = self.dropout(m_a)
        m_b = self.dropout(m_b)

        return  ia, ib, m_a, m_b


class Fusion(nn.Module):
    def __init__(self, input_size, hidden_size,dropout):
        super().__init__()
        self.fusion1 = Linear(hidden_size * 2, hidden_size, activations=True) 
        self.fusion2 = Linear(hidden_size * 2, hidden_size, activations=True)
        self.fusion3 = BiLSTM(hidden_size * 2, hidden_size, dropout=dropout)
        self.fusion = BiLSTM(input_size, hidden_size, dropout=dropout)

    def forward(self, x, i, m):
        i1 = self.fusion1(torch.cat([x, i], dim=-1))
        i2 = self.fusion2(torch.cat([x, x - i], dim=-1))
        _,(i_last, _) = self.fusion3(torch.cat([i1, i2], dim=-1))
        _,(m_last, _) = self.fusion(m)
        return torch.cat((i_last,m_last), dim=-1)

class Pooling(nn.Module):
    def forward(self, x):
        hidden_size = x.size(2)
        return x.permute(1, 0, 2).contiguous().view(-1, hidden_size * 2)

class Prediction(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(dropout),
            Linear(input_size*2, hidden_size*2, activations=True),
            nn.Dropout(dropout),
            Linear(hidden_size*2, num_classes),
        )

    def forward(self, a, b):
        return self.dense(torch.cat([a, b], dim=-1))

class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class Linear(nn.Module):
    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
