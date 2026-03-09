import torch
import torch.nn.init as torch_init
import torch.nn as nn

from src.layers import *

# Add this helper function outside your class
def get_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0).cuda() # Shape: [1, seq_len, d_model]



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Batch, Time, Channels]
        # Add PE to the input
        x = x + self.pe[:, :x.size(1), :]
        return x
    

class XEncoder(nn.Module):
    def __init__(self, d_model, hid_dim, out_dim, n_heads, win_size, dropout, gamma, bias, model_mode, norm=None):
        super(XEncoder, self).__init__()
        self.n_heads = n_heads
        self.win_size = win_size
        self.self_attn = TemporalContext(d_model, hid_dim, hid_dim, n_heads, norm)
        self.linear1 = nn.Conv1d(d_model, d_model // 2, kernel_size=1)
        self.linear2 = nn.Conv1d(d_model // 2, out_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.loc_adj = DistanceAdj(gamma, bias)

        self.pos_encoder = PositionalEncoding(d_model)
        self.model = model_mode
 

    def forward(self, x, seq_len):
        if self.model != 4:
            x = self.pos_encoder(x)
        # x = x + pe # NOW the model knows exact frame order!
        adj = self.loc_adj(x.shape[0], x.shape[1])
        mask = self.get_mask(self.win_size, x.shape[1], seq_len)
        x = x + self.self_attn(x, mask, adj)
        x = self.norm(x).permute(0, 2, 1)
        x = self.dropout1(F.gelu(self.linear1(x)))
        x_e = self.dropout2(F.gelu(self.linear2(x)))

        return x_e, x

    def get_mask(self, window_size, temporal_scale, seq_len):
        if self.model !=4:
            m = torch.tril(torch.ones((temporal_scale, temporal_scale)))
            
            # Optional: If you STILL want to limit how far back it looks
            # (e.g., only remember the last 500 frames to save memory):
            # m = torch.triu(m, diagonal=-window_size)
            
            m = m.repeat(self.n_heads, len(seq_len), 1, 1).cuda()
            return m
        else:
            m = torch.zeros((temporal_scale, temporal_scale))
            w_len = window_size
            for j in range(temporal_scale):
                for k in range(w_len):
                    m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.

            m = m.repeat(self.n_heads, len(seq_len), 1, 1).cuda()

            return m
