import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn
import jax
import haiku as hk
from flax.linen import MultiHeadDotProductAttention
import jax.tree_util as tree
import models_utils as mu
from einshape import jax_einshape as einshape

class MLP(nn.Module):
  hidden_dim: int
  out_dim: int
  depth: int

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i in range(self.depth):
      x = nn.Dense(features=self.hidden_dim)(x)
      x = nn.gelu(x)
    x = nn.Dense(features=self.out_dim)(x)
    return x


class SelfAttnTransformer(nn.Module):
  n_layers: int
  n_heads: int
  head_dim: int
  model_dim: int # the dimension of the input/output of the attention block and dense block
  widening_factor: int # widening factor for hidden_dim

  @nn.compact
  def __call__(self, inputs, mask = None):
    # self attention transformer
    '''
    inputs: [batch_sizes..., length, features]
    mask: [batch_sizes..., num_heads, query_length, key/value_length], 
          where query_length = key/value_length = length
    '''
    x = inputs # the size of x is kept the same throughout the transformer
    for _ in range(self.n_layers):
      attn_block = MultiHeadDotProductAttention(num_heads = self.n_heads, 
                                                qkv_features = self.head_dim * self.n_heads, 
                                                out_features = self.model_dim)
      x = attn_block(inputs_q = x, inputs_k = x, inputs_v = x, mask = mask) + x
      x = nn.LayerNorm()(x)
      dense_block = MLP(hidden_dim = self.model_dim * self.widening_factor, 
                        out_dim = self.model_dim, 
                        depth = 1)
      x = dense_block(x) + x
      x = nn.LayerNorm()(x)
    
    return x
