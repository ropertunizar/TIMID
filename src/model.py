
import torch
from src.modules import *
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.1)


class TimidModel(nn.Module):
    def __init__(self, cfg, mode=1):
        super(TimidModel, self).__init__()
        self.mode = mode


        self.t = cfg.t_step
        self.self_attention = XEncoder(
            d_model=cfg.feat_dim,
            hid_dim=cfg.hid_dim,
            out_dim=cfg.out_dim,
            n_heads=cfg.head_num,
            win_size=cfg.win_size,
            dropout=cfg.dropout,
            gamma=cfg.gamma,
            bias=cfg.bias,
            norm=cfg.norm,
            model_mode = mode
        )
        # classifier now consumes features produced by cross-attention
        # embed_dim corresponds to the dimension of the projected tokens/values
        self.classifier = nn.Conv1d(cfg.out_dim, 1,self.t, padding=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.temp))
        

        self.apply(weight_init)


        # without temporal context
        self.linear = nn.Linear(cfg.feat_dim, cfg.out_dim)
        self.upsample = nn.ConvTranspose1d(
            in_channels=cfg.out_dim, 
            out_channels=cfg.out_dim, 
            kernel_size=3, 
            stride=1, 
            padding=0
        )
        self.activation = nn.ReLU()
        self.simple_layer = nn.Linear(cfg.feat_dim, 512)



        # cross attention configuration
        query_dim = 512           # dimensionality of token features
        embed_dim = query_dim   # attention embedding size
        feature_dim = cfg.out_dim # output channels of video encoder
        # Queries are constructed from video features
        self.q_proj = nn.Linear(feature_dim, embed_dim)
        # Keys and values are computed from token embeddings
        self.k_proj = nn.Linear(query_dim, embed_dim)
        self.v_proj = nn.Linear(query_dim, embed_dim)
        # project attention output into scalar score per query element
        self.out_proj = nn.Linear(embed_dim, 1)
        self.embed_dim = embed_dim

        self.temporal_substitute = nn.Linear(cfg.feat_dim, cfg.out_dim)
        
    def forward(self, x, seq_len, t_features):
        if self.mode == 1:

            x_e, x_v = self.self_attention(x, seq_len)
            x_e = x_e.type_as(x)
            x_e_seq = x_e.permute(0, 2, 1)  # 10xnx300

            # ensure tokens are on same device and have same dtype as video features
            t_features = t_features.to(x.device).type_as(x)

            Q = self.q_proj(x_e_seq)
            K = self.k_proj(t_features)
            V = self.v_proj(t_features)

            d_k = Q.size(-1)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5
            attn_weights = F.softmax(scores, dim=-1)

            context_vect = torch.matmul(attn_weights, V)  # [B, query_len, embed_dim]
            context_vect = F.layer_norm(context_vect + Q, context_vect.size()[1:])
            out = self.out_proj(context_vect)
            
            return out, context_vect
        elif self.mode == 2:
            x_e = self.temporal_substitute(x)
            x_e = x_e.type_as(x)
            x_e_seq = x_e  # 10xnx300

            # ensure tokens are on same device and have same dtype as video features
            t_features = t_features.to(x.device).type_as(x)

            Q = self.q_proj(x_e_seq)
            K = self.k_proj(t_features)
            V = self.v_proj(t_features)

            d_k = Q.size(-1)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5
            attn_weights = F.softmax(scores, dim=-1)

            
            context_vect = torch.matmul(attn_weights, V)  # [B, query_len, embed_dim]
            context_vect = F.layer_norm(context_vect + Q, context_vect.size()[1:])
            out = self.out_proj(context_vect)
            return out, context_vect
        elif self.mode == 3:
            x_e, x_v = self.self_attention(x, seq_len)
            logits = F.pad(x_e, (self.t - 1, 0))
            logits = self.classifier(logits)
            logits = logits.permute(0, 2, 1)
            return logits, x_v
        elif self.mode == 4:
            x_e, x_v = self.self_attention(x, seq_len)
            logits = F.pad(x_e, (self.t - 1, 0))
            logits = self.classifier(logits)
            logits = logits.permute(0, 2, 1)
            logits = torch.sigmoid(logits)
            return logits, x_v
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
