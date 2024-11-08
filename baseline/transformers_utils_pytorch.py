import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, hidden_dim, out_dim, depth):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        assert self.model_dim % self.num_heads == 0, "Model dimension must be divisible by number of heads"

        self.qkv_proj = nn.Linear(model_dim, model_dim * 3)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.model_dim)
        return self.out_proj(attn_output)

class SelfAttnTransformer(nn.Module):
    def __init__(self, n_layers, n_heads, head_dim, model_dim, widening_factor):
        super(SelfAttnTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                MultiHeadSelfAttention(model_dim, n_heads),
                nn.LayerNorm(model_dim),
                MLP(model_dim * widening_factor, model_dim, depth=1),
                nn.LayerNorm(model_dim)
            ]))

    def forward(self, x, mask=None):
        for attn, norm1, mlp, norm2 in self.layers:
            x = norm1(x + attn(x, mask))
            x = norm2(x + mlp(x))
        return x

# Creating data helpers

def build_matrices_from_data_shape(data_shape, mode, shot_num_min, return_shape_list=False):
    demo_num = data_shape['demo_cond_k'][0]
    demo_cond_len = data_shape['demo_cond_k'][1]
    demo_qoi_len = data_shape['demo_qoi_k'][1]
    quest_cond_len = data_shape['quest_cond_k'][1]
    quest_qoi_len = data_shape['quest_qoi_k'][1]

    cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = mu_pt.build_bool_sequence(demo_num, mode, shot_num_min)
    cond_len_list_raw = [demo_cond_len] * demo_num + [quest_cond_len]
    qoi_kv_len_list_raw = [demo_qoi_len] * demo_num + [quest_qoi_len]
    qoi_k_len_list_raw = [demo_qoi_len] * demo_num + [quest_qoi_len]

    cond_len_list = [i * j for i, j in zip(cond_bool_list, cond_len_list_raw)]
    qoi_kv_len_list = [i * j for i, j in zip(qoi_kv_bool_list, qoi_kv_len_list_raw)]
    qoi_k_len_list = [i * j for i, j in zip(qoi_k_bool_list, qoi_k_len_list_raw)]

    basic_mask = mu_pt.build_basic_mask(
        cond_len_list=cond_len_list, 
        qoi_kv_len_list=qoi_kv_len_list, 
        qoi_k_len_list=qoi_k_len_list
    )

    index_pos = mu_pt.build_index_integer(
        cond_len_list=cond_len_list, 
        qoi_kv_len_list=qoi_kv_len_list, 
        qoi_k_len_list=qoi_k_len_list
    )

    out_mask = mu_pt.build_out_mask(
        cond_len_list=cond_len_list, 
        qoi_kv_len_list=qoi_kv_len_list, 
        qoi_k_len_list=qoi_k_len_list, 
        num_range=(shot_num_min, demo_num + 1)
    )

    if return_shape_list:
        return basic_mask, index_pos, out_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list
    else:
        return basic_mask, index_pos, out_mask
