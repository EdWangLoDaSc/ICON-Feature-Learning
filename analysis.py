import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn
import jax
import haiku as hk
from flax.linen import MultiHeadDotProductAttention
import jax.tree_util as tree
import models_utils as mu
import data_utils as du
from einshape import jax_einshape as einshape
from collections import namedtuple

import optax
import random
from typing import Callable
import wandb
import os
from flax.training import checkpoints
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn
import jax
import haiku as hk
from flax.linen import MultiHeadDotProductAttention
import jax.tree_util as tree
import models_utils as mu
import data_utils as du
from einshape import jax_einshape as einshape
from collections import namedtuple

import optax
import random
from typing import Callable
import wandb
import os
from flax.training import checkpoints
import gc


num_demos=10
num_repeats_val=3
num_repeats_ood=1


raw_data_val_test_OOD_c_OOD_boundary_inrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/test_OOD_c_OOD_boundary_inrange_Poisson_PDE_data_test_oodc.npy",mmap_mode='r')
raw_data_val_test_OOD_c_OOD_boundary_outrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/test_OOD_c_OOD_boundary_outrange_Poisson_PDE_data_test_oodc.npy",mmap_mode='r')
raw_data_val_test_OOD_c_iid_boundary = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/test_OOD_c_iid_boundary_Poisson_PDE_data_test_oodc.npy",mmap_mode='r')
raw_data_val_train_iid_C_OOD_boundary_inrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/train_iid_C_OOD_boundary_inrange_Poisson_PDE_data_test_iidc.npy",mmap_mode='r')
raw_data_val_train_iid_C_OOD_boundary_outrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/train_iid_C_OOD_boundary_outrange_Poisson_PDE_data_test_iidc.npy",mmap_mode='r')
#raw_data_val = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/Poisson_PDE_data_valid_1k_-22_305.npy")



val_data_raw_data_val_test_OOD_c_OOD_boundary_inrange, val_combinations_raw_data_val_test_OOD_c_OOD_boundary_inrange = du.prepare_val_samples(raw_data_val_test_OOD_c_OOD_boundary_inrange, num_demos=num_demos, num_repeats=num_repeats_val)
val_data_raw_data_val_test_OOD_c_OOD_boundary_outrange, val_combinations_raw_data_val_test_OOD_c_OOD_boundary_outrange = du.prepare_val_samples(raw_data_val_test_OOD_c_OOD_boundary_outrange, num_demos=num_demos, num_repeats=num_repeats_ood)
val_data_raw_data_val_test_OOD_c_iid_boundary, val_combinations_raw_data_val_test_OOD_c_iid_boundary = du.prepare_val_samples(raw_data_val_test_OOD_c_iid_boundary, num_demos=num_demos, num_repeats=num_repeats_val)
val_data_raw_data_val_train_iid_C_OOD_boundary_inrange, val_combinations_raw_data_val_train_iid_C_OOD_boundary_inrange = du.prepare_val_samples(raw_data_val_train_iid_C_OOD_boundary_inrange, num_demos=num_demos, num_repeats=num_repeats_val)
val_data_raw_data_val_train_iid_C_OOD_boundary_outrange, val_combinations_raw_data_val_train_iid_C_OOD_boundary_outrange = du.prepare_val_samples(raw_data_val_train_iid_C_OOD_boundary_outrange, num_demos=num_demos, num_repeats=num_repeats_val)

n_shot = 10
import numpy as np
import jax.numpy as jnp
import jax
from flax import linen as nn
from flax.training import checkpoints
import haiku as hk
from flax.linen import MultiHeadDotProductAttention
import jax.tree_util as tree
from collections import namedtuple
from typing import Callable
import os

# 重新定义模型架构（需要与训练时完全相同）
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
    model_dim: int
    widening_factor: int
    
    @nn.compact
    def __call__(self, inputs, mask=None):
        x = inputs
        for _ in range(self.n_layers):
            attn_block = MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.head_dim * self.n_heads,
                out_features=self.model_dim,
            )
            x = attn_block(inputs_q=x, inputs_k=x, inputs_v=x, mask=mask) + x
            x = nn.LayerNorm()(x)
            dense_block = MLP(
                hidden_dim=self.model_dim * self.widening_factor,
                out_dim=self.model_dim,
                depth=1,
            )
            x = dense_block(x) + x
            x = nn.LayerNorm()(x)
        return x

def physics_aware_init(key, shape, dtype=jnp.float32):
    freq_components = jnp.linspace(0, 1, shape[-1])
    base_init = jax.random.normal(key, shape, dtype)
    weights = jnp.exp(-freq_components)
    return base_init * weights

class ICON_LM(nn.Module):
    config: dict
    basic_mask: jnp.ndarray
    index_pos: jnp.ndarray
    out_mask: jnp.ndarray

    def setup(self):
        self.pre_projection = nn.Dense(
            self.config['transformer']['model_dim'],
            kernel_init=physics_aware_init,
            name="pre_projection"
        )
        self.func_pos_embedding = nn.Embed(
            (self.config['demo_max_num']) * 3,
            self.config['transformer']['model_dim'],
            name="func_pos_embedding"
        )
        self.transformer = SelfAttnTransformer(
            **(self.config['transformer']),
            name='transformer'
        )
        self.post_projection = nn.Dense(
            self.config['out_dim'],
            name="post_projection"
        )

    def basic_forward(self, data, mode, index_pos, basic_mask):
        sequence = self.pre_projection(data)
        sequence = sequence + self.func_pos_embedding(index_pos)
        mask = jnp.broadcast_to(
            basic_mask[None, :, :],
            (self.config['transformer']['n_heads'], basic_mask.shape[0], basic_mask.shape[1])
        )
        sequence = self.transformer(sequence, mask=mask)
        sequence = self.post_projection(sequence)
        return sequence

    def __call__(self, data):
        sequence = self.basic_forward(data, 'train', self.index_pos, self.basic_mask)
        sequence = sequence[self.out_mask]
        return sequence

    def predict(self, data):
        data_shape = tree.tree_map(lambda x: x.shape, data)
        basic_mask, index_pos, out_mask = build_matrices_from_data_shape(
            data_shape, mode='test', shot_num_min=0
        )
        sequence = self.basic_forward(data, 'test', index_pos, basic_mask)
        sequence = sequence[-data.quest_qoi_mask.shape[-1]:, :]
        return sequence

def load_trained_model(ckpt_path, model_config, basic_mask, index_pos, out_mask):
    """加载训练好的模型"""
    # 初始化模型
    icon_lm_model = ICON_LM(
        model_config,
        basic_mask=basic_mask,
        index_pos=index_pos,
        out_mask=out_mask
    )
    
    # 创建初始参数以获取参数结构
    rng = jax.random.PRNGKey(0)
    subkey1, subkey2 = jax.random.split(rng)
    
    # 创建示例输入数据
    n_shot = model_config['demo_max_num'] - 1
    Data = namedtuple('Data', [
        'demo_cond_k', 'demo_cond_v', 'demo_cond_mask',
        'demo_qoi_k', 'demo_qoi_v', 'demo_qoi_mask',
        'quest_cond_k', 'quest_cond_v', 'quest_cond_mask',
        'quest_qoi_k', 'quest_qoi_mask'
    ])
    
    test_data = Data(
        demo_cond_k=np.ones((n_shot, 50, 1)),
        demo_cond_v=np.ones((n_shot, 50, 1)),
        demo_cond_mask=np.ones((n_shot, 50)).astype(bool),
        demo_qoi_k=np.ones((n_shot, 50, 1)),
        demo_qoi_v=np.ones((n_shot, 50, 1)),
        demo_qoi_mask=np.ones((n_shot, 50)).astype(bool),
        quest_cond_k=np.ones((1, 50, 1)),
        quest_cond_v=np.ones((1, 50, 1)),
        quest_cond_mask=np.ones((1, 50)).astype(bool),
        quest_qoi_k=np.ones((1, 50, 1)),
        quest_qoi_mask=np.ones((1, 50)).astype(bool)
    )
    
    # 初始化模型参数
    rngs = {'params': subkey1, 'dropout': subkey2}
    params = icon_lm_model.init(rngs, test_data)
    
    # 加载检查点
    restored_params = checkpoints.restore_checkpoint(ckpt_path, target=params)
    
    return icon_lm_model, restored_params

# 使用示例
if __name__ == "__main__":
    # 设置模型配置
    n_shot = 10
    model_config = {
        "demo_max_num": n_shot + 1,
        "index_mode": "learn",
        "transformer": {
            "n_layers": 3,
            "n_heads": 8,
            "head_dim": 256,
            "model_dim": 256,
            "widening_factor": 4,
        },
        "out_dim": 1
    }
    
    # 加载模型
    ckpt_path = "/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/initial_phrase/ckpt_layers3/best_model_70"
    
    Data = namedtuple('Data', ['demo_cond_k', 'demo_cond_v', 'demo_cond_mask', 
                          'demo_qoi_k', 'demo_qoi_v', 'demo_qoi_mask',
                          'quest_cond_k', 'quest_cond_v', 'quest_cond_mask',
                          'quest_qoi_k', 'quest_qoi_mask',])

    test_data = Data(demo_cond_k = np.ones((n_shot, 50, 1)),
                  demo_cond_v = np.ones((n_shot, 50, 1)),
                  demo_cond_mask = np.ones((n_shot, 50)).astype(bool),
                  demo_qoi_k = np.ones((n_shot, 50, 1)),
                  demo_qoi_v = np.ones((n_shot, 50, 1)),
                  demo_qoi_mask = np.ones((n_shot, 50)).astype(bool),
                  quest_cond_k = np.ones((1, 50, 1)),
                  quest_cond_v = np.ones((1, 50, 1)),
                  quest_cond_mask = np.ones((1, 50)).astype(bool),
                  quest_qoi_k = np.ones((1, 50, 1)),
                  quest_qoi_mask = np.ones((1, 50)).astype(bool)
                  )


    data_shape = tree.tree_map(lambda x: x.shape, test_data)

    basic_mask, index_pos, out_mask = mu.build_matrices_from_data_shape(data_shape,  mode = 'train', shot_num_min = 0)

    
    model, params = load_trained_model(
        ckpt_path,
        model_config,
        basic_mask,
        index_pos,
        out_mask
    )
    
    print("Model loaded successfully!")
    
    # 创建预测函数
    @jax.jit
    def predict_fn(params, data):
        return model.apply(params, data, method="predict")
    
    predictions = predict_fn(params, test_data)
    
    
    