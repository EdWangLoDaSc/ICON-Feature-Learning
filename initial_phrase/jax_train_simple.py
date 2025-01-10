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


n_shot = 8
class MLP(nn.Module):
  hidden_dim: int
  out_dim: int
  depth: int
  kernel_init: Callable = nn.initializers.lecun_normal()  # 添加kernel_init参数

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i in range(self.depth):
      x = nn.Dense(features=self.hidden_dim, kernel_init=self.kernel_init)(x)
      x = nn.gelu(x)
    x = nn.Dense(features=self.out_dim, kernel_init=self.kernel_init)(x)
    return x


class SelfAttnTransformer(nn.Module):
  n_layers: int
  n_heads: int
  head_dim: int
  model_dim: int # the dimension of the input/output of the attention block and dense block
  widening_factor: int # widening factor for hidden_dim
  kernel_init: Callable = nn.initializers.lecun_normal()  # 添加kernel_init参数
  
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
                                                out_features = self.model_dim,
                                                kernel_init = self.kernel_init  # 使用传入的初始化器
                                                )
      x = attn_block(inputs_q = x, inputs_k = x, inputs_v = x, mask = mask) + x
      x = nn.LayerNorm()(x)
      dense_block = MLP(hidden_dim = self.model_dim * self.widening_factor, 
                        out_dim = self.model_dim, 
                        depth = 1,
                        kernel_init = self.kernel_init  # 需要相应修改MLP类
                        )
      x = dense_block(x) + x
      x = nn.LayerNorm()(x)
    
    return x



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

basic_mask, index_pos, out_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list = \
    mu.build_matrices_from_data_shape(data_shape, 'train', 0, return_shape_list = True)



def physics_aware_init(key, shape, dtype=jnp.float32):
    """基于PDE特性的初始化"""
    # 使用傅里叶基函数启发的初始化
    freq_components = jnp.linspace(0, 1, shape[-1])
    base_init = jax.random.normal(key, shape, dtype)
    # 加入频率调制
    weights = jnp.exp(-freq_components) # 高频衰减
    return base_init * weights

class ICON_LM(nn.Module):

  config: dict
  # these matrices are constant, so we can pre-compute them.
  basic_mask: jnp.ndarray # [x, x] the mask designed for ICON-LM
  index_pos: jnp.ndarray # [x] the index for positional embedding
  out_mask: jnp.ndarray # [x]
  
  '''def setup(self):
    self.pre_projection = nn.Dense(self.config['transformer']['model_dim'], name="pre_projection")
    # trainable positional embedding
    self.func_pos_embedding = nn.Embed((self.config['demo_max_num']) * 3, self.config['transformer']['model_dim'], name="func_pos_embedding")
    self.transformer = SelfAttnTransformer(**(self.config['transformer']), name='transformer')
    self.post_projection = nn.Dense(self.config['out_dim'], name="post_projection")'''

  def setup(self):
    self.pre_projection = nn.Dense(self.config['transformer']['model_dim'],kernel_init=physics_aware_init,name="pre_projection")
    self.func_pos_embedding = nn.Embed((self.config['demo_max_num']) * 3, self.config['transformer']['model_dim'], name="func_pos_embedding")
    #self.transformer = SelfAttnTransformer(**(self.config['transformer']), name='transformer')
    self.transformer = SelfAttnTransformer(**(self.config['transformer']), 
                                         #kernel_init=physics_aware_init,  # Add this parameter
                                         name='transformer')
    
    #self.post_projection = nn.Dense(self.config['out_dim'], name="post_projection")

    self.post_projection = nn.Dense(self.config['out_dim'], 
                                  #kernel_init=physics_aware_init,
                                  name="post_projection")

  def basic_forward(self, data, mode, index_pos, basic_mask):

    demo_num = len(data.demo_cond_k)
    cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = mu.build_bool_sequence(demo_num, mode = mode, shot_num_min = 0)
    sequence = mu.build_data_sequence(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)
    mask = mu.build_data_mask(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)
    
    sequence = self.pre_projection(sequence)
    sequence = sequence + self.func_pos_embedding(index_pos)

    mask = einshape('i->ji', mask, j = sequence.shape[0])
    mask = mask * basic_mask
    mask = einshape("ij->mij", mask, m = self.config['transformer']['n_heads']) 
    sequence = self.transformer(sequence, mask = mask)
    sequence = self.post_projection(sequence)
    return sequence
  
  def __call__(self, data): 
    '''
    this is for standard forward, using pre-computed matrices
    used for training, predict all the QoIs based on previous examples and the current condition
    '''
    sequence = self.basic_forward(data, 'train', self.index_pos, self.basic_mask)
    sequence = sequence[self.out_mask] # [num * qoi_len, out_dim]
    return sequence

  def predict(self, data):
    '''
    this is for flexible data shape, will build basic mask, index, and out mask on the fly,
    used for testing, predict the last QoI, i.e. the question QoI
    '''
    data_shape = tree.tree_map(lambda x: x.shape, data)
    basic_mask, index_pos, out_mask = mu.build_matrices_from_data_shape(data_shape, mode = 'test', shot_num_min = 0)
    sequence = self.basic_forward(data, 'test', index_pos, basic_mask)
    sequence = sequence[-data.quest_qoi_mask.shape[-1]:,:] # [quest_qoi_len, out_dim]
    return sequence


model_config = {"demo_max_num": n_shot+1,
                "index_mode": "learn",
                "transformer": {"n_layers":3,
                                "n_heads":8, 
                                "head_dim":256, 
                                "model_dim":256, 
                                "widening_factor": 4, 
                              },
                "out_dim": 1
              }


basic_mask, index_pos, out_mask = mu.build_matrices_from_data_shape(data_shape,  mode = 'train', shot_num_min = 0)
rng_gen = hk.PRNGSequence(42) 
icon_lm_model = ICON_LM(model_config, basic_mask = basic_mask, index_pos = index_pos, out_mask = out_mask)
subkey1, subkey2 = jax.random.split(jax.random.PRNGKey(0))
rngs = {'params': subkey1, 'dropout': subkey2}
params = icon_lm_model.init(rngs, test_data)

@jax.jit
def icon_lm_forward_fn(params, rng_key, data):
  # rngs is useless here since the model doesn't have dropout or other random operations
  return icon_lm_model.apply(params, data, rngs = {'dropout':rng_key}) 

@jax.jit
def icon_lm_predict_fn(params, rng_key, data):
  return icon_lm_model.apply(params, data, rngs = {'dropout':rng_key}, method = "predict") 


file_name = "/home/wanghanyang.01/icon-feature_learning/scratch/data/data_possion_jax/Poisson_PDE_data_train.npy"
raw_data = np.load(file_name)

raw_data_val = np.load("/home/wanghanyang.01/icon-feature_learning/scratch/data/data_possion_jax/Poisson_PDE_data_valid_1k_-21_455.npy")

def calculate_poisson_metrics(pred_c, true_c, u, dx=0.01):
    """
    计算基于Poisson方程物理特性的评估指标
    pred_c: 预测的源项c [batch_size, seq_len, 1]
    true_c: 真实的源项c [batch_size, seq_len, 1]
    u: 输入的位移u [batch_size, seq_len, 1]
    """
    # 确保输入是正确的形状和类型
    dx = jnp.array(dx, dtype=jnp.float32)
    
    # 移除多余的维度并确保是2D数组
    pred_c = pred_c.reshape(-1)  # 展平为1D数组
    true_c = true_c.reshape(-1)
    
    # 1. 源项符号准确率
    sign_accuracy = jnp.mean(
        jnp.sign(pred_c) == jnp.sign(true_c)
    )
    
    pred_c_integral = jnp.sum(pred_c) * dx  # 简单的矩形积分
    true_c_integral = jnp.sum(true_c) * dx
    integral_error = jnp.abs(
        pred_c_integral - true_c_integral
    ) / (jnp.abs(true_c_integral) + 1e-8)
    
    integral_threshold = 0.05
    
    poisson_accuracy = jnp.mean(jnp.array([
        integral_error < integral_threshold,
        #sign_accuracy > 0.9
    ]))
    
    return {
        'sign_accuracy': sign_accuracy,
        'integral_error': integral_error,
        'poisson_accuracy': poisson_accuracy
    }

# 修改训练步骤
@jax.jit
def train_step(params, opt_state, batch, labels, rng):
    """Single training step with Poisson-specific metrics"""
    def loss_fn(params):
        preds = icon_lm_forward_fn_batch(params, rng, batch)
        preds = preds[:,-50:,:]  # 预测的c
        u = batch.quest_cond_k[:,-50:,:]  # 输入的u
        
        loss = jnp.mean((preds - labels.reshape(preds.shape)) ** 2)
        
        # 计算每个batch中所有样本的平均Poisson指标
        poisson_metrics = jax.vmap(calculate_poisson_metrics)(
            preds, 
            labels,
            u,
            jnp.array([0.01] * preds.shape[0])  # 为每个样本提供dx
        )
        avg_poisson_metrics = jax.tree_map(jnp.mean, poisson_metrics)
        
        return loss, avg_poisson_metrics

    (loss, poisson_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, poisson_metrics

icon_lm_forward_fn_batch = jax.jit(jax.vmap(icon_lm_forward_fn, in_axes = (None, None, 0)))
icon_lm_predict_fn_batch = jax.jit(jax.vmap(icon_lm_predict_fn, in_axes = (None, None, 0)))



def create_optimizer(learning_rate=1e-4, weight_decay=1e-5):
    """
    创建带有weight decay的优化器
    
    Args:
        learning_rate: 学习率
        weight_decay: weight decay系数
    """
    # 组合adam和weight decay
    optimizer = optax.chain(
        optax.adam(learning_rate=learning_rate),
        #optax.add_decayed_weights(weight_decay)
    )
    return optimizer

batch_size = 32
num_demos = n_shot
num_repeats = 10
learning_rate = 1e-4
weight_decay = 1e-5  # 可以尝试不同的值：1e-4, 1e-3, 1e-2

optimizer = create_optimizer(learning_rate, weight_decay)

def compute_weight_norm(params):
    """计算参数的L2范数"""
    return optax.global_norm(params)
    
opt_state = optimizer.init(params)
# 准备训练集和验证集
num_epochs = 1000

val_data, val_combinations = du.prepare_val_samples(raw_data_val, num_demos=num_demos, num_repeats=num_repeats)

# 初始化wandb
wandb.init(
    project="icon-learning",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_demos": num_demos,
        "num_repeats": num_repeats,
        "num_epochs": num_epochs,
        "model_config": model_config  
    },
    name="layer2_nhead8_headdim256_widening4_100%_5repeat_physics"
)


# 训练循环
rng = jax.random.PRNGKey(0)
best_metrics = {
    'val_loss': float('inf'),
    'val_sign_accuracy': 0.0,
    'val_integral_error': float('inf'),
    'val_poisson_accuracy': 0.0,
    'train_loss': float('inf'),
    'train_sign_accuracy': 0.0,
    'train_integral_error': float('inf'),
    'train_poisson_accuracy': 0.0
}

for epoch in range(num_epochs):
    # 训练阶段
    all_combinations = du.prepare_epoch_samples(raw_data, num_demos, num_repeats)
    # all_combinations = all_combinations*0.5

    # 随机打乱, 取一半
    all_combinations = all_combinations[:int(len(all_combinations))]
    np.random.shuffle(all_combinations)
    
    train_loss = 0.0
    num_train_batches = 0
    
    for batch_start in range(0, len(all_combinations), batch_size):
        batch_end = min(batch_start + batch_size, len(all_combinations))
        batch_combinations = all_combinations[batch_start:batch_end]
        
        rng, step_rng = jax.random.split(rng)
        batch_data, batch_label = du.get_batch_data(raw_data, batch_combinations)
        params, opt_state, loss, poisson_metrics = train_step(params, opt_state, batch_data, batch_label, step_rng)
        
        train_loss += loss
        num_train_batches += 1
        
        # 记录每个batch的loss
        wandb.log({
            "batch_train_loss": loss,
            "epoch": epoch,
            "batch": num_train_batches,
            "train_sign_accuracy": poisson_metrics['sign_accuracy'],
            "train_integral_error": poisson_metrics['integral_error'],
            "train_poisson_accuracy": poisson_metrics['poisson_accuracy']
        })
        
        if num_train_batches % 10 == 0:
            print(f"Epoch {epoch}, Batch {num_train_batches}, Train Loss: {loss:.4f}")
    
    avg_train_loss = train_loss / num_train_batches
    
    # 验证阶段
    val_metrics_sum = {
        'loss': 0.0,
        'sign_accuracy': 0.0,
        'integral_error': 0.0,
        'poisson_accuracy': 0.0
    }
    
    # 为每个shot数量创建指标字典
    val_metrics_sum_xshots = {
        i: {
            'loss': 0.0,
            'sign_accuracy': 0.0,
            'integral_error': 0.0,
            'poisson_accuracy': 0.0
        } for i in range(1, 9)  # 1-8 shots
    }
    
    num_val_batches = 0
    for batch_start in range(0, len(val_combinations), batch_size):
        batch_end = min(batch_start + batch_size, len(val_combinations))
        batch_combinations = val_combinations[batch_start:batch_end]
        
        rng, step_rng = jax.random.split(rng)
        batch_data, batch_label = du.get_batch_data(val_data, batch_combinations)
        
        # 验证原始数据
        val_preds = icon_lm_forward_fn_batch(params, step_rng, batch_data)
        val_preds = val_preds[:,-50:,:]
        val_loss = jnp.mean((val_preds - batch_label.reshape(val_preds.shape)) ** 2)
        
        # 计算原始验证集的物理指标
        val_poisson_metrics = jax.vmap(calculate_poisson_metrics)(
            val_preds,
            batch_label,
            batch_data.quest_cond_k[:,-50:,:],
            jnp.array([0.01] * val_preds.shape[0])
        )
        
        # 累加原始batch的指标
        val_metrics_sum['loss'] += val_loss
        for key in val_poisson_metrics:
            val_metrics_sum[key] += jnp.mean(val_poisson_metrics[key])
        
        # 对每个shot数量进行验证
        for shot in range(1, n_shot+1):
            batch_data_xshot, batch_label_xshot = du.get_batch_data_xshot(val_data, batch_combinations, xshot=shot)
            
            # 验证xshot数据
            val_preds_xshot = icon_lm_forward_fn_batch(params, step_rng, batch_data_xshot)
            val_preds_xshot = val_preds_xshot[:,-50:,:]
            val_loss_xshot = jnp.mean((val_preds_xshot - batch_label_xshot.reshape(val_preds_xshot.shape)) ** 2)
            
            # 计算xshot验证集的物理指标
            val_poisson_metrics_xshot = jax.vmap(calculate_poisson_metrics)(
                val_preds_xshot,
                batch_label_xshot,
                batch_data_xshot.quest_cond_k[:,-50:,:],
                jnp.array([0.01] * val_preds_xshot.shape[0])
            )
            
            # 累加xshot batch的指标
            val_metrics_sum_xshots[shot]['loss'] += val_loss_xshot
            for key in val_poisson_metrics_xshot:
                val_metrics_sum_xshots[shot][key] += jnp.mean(val_poisson_metrics_xshot[key])
        
        num_val_batches += 1
    
    # 计算平均指标
    avg_val_metrics = {k: v / num_val_batches for k, v in val_metrics_sum.items()}
    avg_val_metrics_xshots = {
        shot: {k: v / num_val_batches for k, v in metrics.items()}
        for shot, metrics in val_metrics_sum_xshots.items()
    }
    
    # 转换为可记录的格式
    train_metrics = {
        'loss': float(loss),
        'sign_accuracy': float(poisson_metrics['sign_accuracy']),
        'integral_error': float(poisson_metrics['integral_error']),
        'poisson_accuracy': float(poisson_metrics['poisson_accuracy'])
    }
    
    val_metrics = {
        'loss': float(avg_val_metrics['loss']),
        'sign_accuracy': float(avg_val_metrics['sign_accuracy']),
        'integral_error': float(avg_val_metrics['integral_error']),
        'poisson_accuracy': float(avg_val_metrics['poisson_accuracy'])
    }
    
    # 更新best指标
    if val_metrics['loss'] < best_metrics['val_loss']:
        best_metrics['val_loss'] = val_metrics['loss']
    if val_metrics['sign_accuracy'] > best_metrics['val_sign_accuracy']:
        best_metrics['val_sign_accuracy'] = val_metrics['sign_accuracy']
    if val_metrics['integral_error'] < best_metrics['val_integral_error']:
        best_metrics['val_integral_error'] = val_metrics['integral_error']
    if val_metrics['poisson_accuracy'] > best_metrics['val_poisson_accuracy']:
        best_metrics['val_poisson_accuracy'] = val_metrics['poisson_accuracy']
    
    if train_metrics['loss'] < best_metrics['train_loss']:
        best_metrics['train_loss'] = train_metrics['loss']
    if train_metrics['sign_accuracy'] > best_metrics['train_sign_accuracy']:
        best_metrics['train_sign_accuracy'] = train_metrics['sign_accuracy']
    if train_metrics['integral_error'] < best_metrics['train_integral_error']:
        best_metrics['train_integral_error'] = train_metrics['integral_error']
    if train_metrics['poisson_accuracy'] > best_metrics['train_poisson_accuracy']:
        best_metrics['train_poisson_accuracy'] = train_metrics['poisson_accuracy']
    
    # wandb记录
    wandb_log_dict = {
        "train/epoch": epoch,
        "train/batch": num_train_batches,
    
        "metrics/train_loss": train_metrics['loss'],
        "metrics/train_poisson_accuracy": train_metrics['poisson_accuracy'],
        
        "metrics/val_loss": val_metrics['loss'],
        "metrics/val_poisson_accuracy": val_metrics['poisson_accuracy'],
        
        **{f"metrics/val_loss_{i}shot": avg_val_metrics_xshots[i]['loss'] for i in range(1, 9)},
        **{f"metrics/val_poisson_accuracy_{i}shot": avg_val_metrics_xshots[i]['poisson_accuracy'] for i in range(1, 9)},
        
        # Best指标
        "best/val_loss": best_metrics['val_loss'],
        "best/val_poisson_accuracy": best_metrics['val_poisson_accuracy'],
    }
    
    # 检查数值有效性
    for k, v in wandb_log_dict.items():
        if not isinstance(v, (int, float)):
            print(f"Warning: {k} is not a number: {v}, type: {type(v)}")
        if isinstance(v, float) and (jnp.isnan(v) or jnp.isinf(v)):
            print(f"Warning: {k} is nan or inf: {v}")
    
    # 记录到wandb
    wandb.log(wandb_log_dict)
    
    # 打印当前epoch的主要指标
    print(f"\nEpoch {epoch}")
    print(f"Train Loss: {train_metrics['loss']:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f}")
    print("\nPoisson Accuracy:")
    print(f"Train: {train_metrics['poisson_accuracy']:.4f}")
    print(f"Val: {val_metrics['poisson_accuracy']:.4f}")
    print("\nX-shot Val Poisson Accuracy:")
    for i in range(1, 9):
        print(f"{i}-shot: {avg_val_metrics_xshots[i]['poisson_accuracy']:.4f}", end='  ')
    print("\n" + "-" * 50)

    # 在训练循环中
    weight_norm = compute_weight_norm(params)

    wandb.log({
        "weight_norm": float(weight_norm),
        "val_loss": val_metrics['loss'],
        "val_sign_accuracy": val_metrics['sign_accuracy'],
        "val_integral_error": val_metrics['integral_error'],
        "val_poisson_accuracy": val_metrics['poisson_accuracy'],
        
        # 计算weight_norm的相对变化
        "weight_norm_change": float(weight_norm - prev_weight_norm) if epoch > 0 else 0.0
    })

    prev_weight_norm = weight_norm
wandb.finish()