import math
import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, id_cnt, emb_d):
        super().__init__()
        self.id_cnt = id_cnt
        self.emb_d = emb_d
        self.id_to_emb = nn.Embedding(id_cnt, emb_d)
        
    def forward(self, sentence):
        return self.id_to_emb(sentence)

class SinusoidalPositionalEncoding:
    def __init__(self, max_len, emb_d):
        self.max_len = max_len
        self.emb_d = emb_d

    def get_PE(self, length): # length를 안 받으면, length가 max_length보다 짧은 경우에도 계산됨
        emb_d = self.emb_d
        PE = torch.empty(length, self.emb_d)
        pos = torch.arange(0, length).unsqueeze(1)
        i = torch.arange(0, emb_d, 2)
        inst = pos / (10000**(i/emb_d))
        PE[:, 0::2] = torch.sin(inst)
        PE[:, 1::2] = torch.cos(inst)
        return PE


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        
        scores = torch.einsum("bhsd,bhtd->bhst", Q, K) / math.sqrt(d_k)
        
        if mask is not None:
            # maked_fill은 해당되는 위치의 값을 바꾸는 함수
            # mask 위치를 -1e9로 바꾼다는 뜻 (softmax에서 거의 0이 되는 값)
            scores = scores.masked_fill(mask==0, -1e9)

        weight = torch.softmax(scores, dim=-1)

        out = torch.einsum("bhst,bhtd->bhsd", weight, V)
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, head_size, emb_d, dropout=0.1):
        super().__init__()

        self.head_size = head_size
        self.emb_d = emb_d
        self.d_k = emb_d // head_size

        self.w_q = nn.Linear(emb_d, emb_d)
        self.w_k = nn.Linear(emb_d, emb_d)
        self.w_v = nn.Linear(emb_d, emb_d)
        self.w_o = nn.Linear(emb_d, emb_d)

        self.sdpa = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X, mask=None):
        B, S, D = X.shape

        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)

        Q = Q.view(B, S, self.head_size, self.d_k).transpose(1, 2)
        K = K.view(B, S, self.head_size, self.d_k).transpose(1, 2)
        V = V.view(B, S, self.head_size, self.d_k).transpose(1, 2)
        
        out = self.sdpa(Q, K, V, mask)

        out = out.transpose(1, 2).contiguous().view(B, S, D)

        out = self.w_o(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, emb_d, ffn_d):
        super().__init__()
        self.emb_d = emb_d
        self.ffn_d = ffn_d

        self.ffn = nn.Sequential(
            nn.Linear(emb_d, ffn_d),
            nn.ReLU(),
            nn.Linear(ffn_d, emb_d)
        )
    
    def forward(self, X):
        return self.ffn(X)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_d, head_size, ffn_d, dropout=0.1):
        super().__init__()
        self.emb_d = emb_d
        self.head_size = head_size
        self.droptout = dropout
        self.mha = MultiHeadSelfAttention(head_size, emb_d, dropout)
        self.ffn = FeedForward(emb_d, ffn_d)
        self.l_norm1 = nn.LayerNorm(emb_d)
        self.l_norm2 = nn.LayerNorm(emb_d)

    def forward(self, X, mask):
        mha_out = self.mha(X, mask)
        ffn_out = self.l_norm1(self.ffn(mha_out) + X)
        out = self.l_norm2(ffn_out + X)
        return out