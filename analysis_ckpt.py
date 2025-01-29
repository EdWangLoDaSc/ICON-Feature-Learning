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
import json
from datetime import datetime
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
n_shot = 10

# Add this helper function near the top of the file, after the imports
def convert_to_json_serializable(obj):
    """Convert JAX arrays and other non-serializable objects to Python types"""
    if isinstance(obj, (jnp.ndarray, jax.Array)):
        return float(obj)  # Convert single values to float
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(x) for x in obj]
    return obj

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
                                                out_features = self.model_dim,
                                                )
      x = attn_block(inputs_q = x, inputs_k = x, inputs_v = x, mask = mask) + x
      x = nn.LayerNorm()(x)
      dense_block = MLP(hidden_dim = self.model_dim * self.widening_factor, 
                        out_dim = self.model_dim, 
                        depth = 1,
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
  basic_mask: jnp.ndarray 
  index_pos: jnp.ndarray 
  out_mask: jnp.ndarray


  def setup(self):
    self.pre_projection = nn.Dense(self.config['transformer']['model_dim'],kernel_init=physics_aware_init,name="pre_projection")
    self.func_pos_embedding = nn.Embed((self.config['demo_max_num']) * 3, self.config['transformer']['model_dim'], name="func_pos_embedding")
    #self.transformer = SelfAttnTransformer(**(self.config['transformer']), name='transformer')
    self.transformer = SelfAttnTransformer(**(self.config['transformer']), 
                                         name='transformer')

    self.post_projection = nn.Dense(self.config['out_dim'], 
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
    sequence = sequence[self.out_mask]
    return sequence

  def predict(self, data):

    data_shape = tree.tree_map(lambda x: x.shape, data)
    basic_mask, index_pos, out_mask = mu.build_matrices_from_data_shape(data_shape, mode = 'test', shot_num_min = 0)
    sequence = self.basic_forward(data, 'test', index_pos, basic_mask)
    sequence = sequence[-data.quest_qoi_mask.shape[-1]:,:] # [quest_qoi_len, out_dim]
    return sequence


model_config = {"demo_max_num": n_shot+1,
                "index_mode": "learn",
                "transformer": {"n_layers":2,
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


file_name = "/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/train_iid_Poisson_PDE_data_train.npy"
raw_data = np.load(file_name,mmap_mode='r')


raw_data_val_test_OOD_c_OOD_boundary_inrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/test_OOD_c_OOD_boundary_inrange_Poisson_PDE_data_test_oodc.npy",mmap_mode='r')
raw_data_val_test_OOD_c_OOD_boundary_outrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/test_OOD_c_OOD_boundary_outrange_Poisson_PDE_data_test_oodc.npy",mmap_mode='r')
raw_data_val_test_OOD_c_iid_boundary = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/test_OOD_c_iid_boundary_Poisson_PDE_data_test_oodc.npy",mmap_mode='r')
raw_data_val_train_iid_C_OOD_boundary_inrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/train_iid_C_OOD_boundary_inrange_Poisson_PDE_data_test_iidc.npy",mmap_mode='r')
raw_data_val_train_iid_C_OOD_boundary_outrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/train_iid_C_OOD_boundary_outrange_Poisson_PDE_data_test_iidc.npy",mmap_mode='r')
#raw_data_val = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/Poisson_PDE_data_valid_1k_-22_305.npy")



def calculate_poisson_metrics(pred_c, true_c, u, dx=0.01):

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
    
    integral_threshold = 0.1
    
    poisson_accuracy = jnp.mean(jnp.array([
        integral_error < integral_threshold,
        #sign_accuracy > 0.9
    ]))
    
    return {
        'sign_accuracy': sign_accuracy,
        'integral_error': integral_error,
        'poisson_accuracy': poisson_accuracy
    }


icon_lm_forward_fn_batch = jax.jit(jax.vmap(icon_lm_forward_fn, in_axes = (None, None, 0)))
icon_lm_predict_fn_batch = jax.jit(jax.vmap(icon_lm_predict_fn, in_axes = (None, None, 0)))



def create_optimizer(learning_rate=1e-4, weight_decay=1e-5):

    # 组合adam和weight decay
    optimizer = optax.chain(
        optax.adam(learning_rate=learning_rate),
        optax.add_decayed_weights(weight_decay)
    )
    return optimizer

batch_size = 32
num_demos = n_shot


# 准备训练集和验证集
num_epochs = 1000

num_repeats_val=1
num_repeats_ood=1
val_data_raw_data_val_test_OOD_c_OOD_boundary_inrange, val_combinations_raw_data_val_test_OOD_c_OOD_boundary_inrange = du.prepare_val_samples(raw_data_val_test_OOD_c_OOD_boundary_inrange, num_demos=num_demos, num_repeats=num_repeats_val)
val_data_raw_data_val_test_OOD_c_OOD_boundary_outrange, val_combinations_raw_data_val_test_OOD_c_OOD_boundary_outrange = du.prepare_val_samples(raw_data_val_test_OOD_c_OOD_boundary_outrange, num_demos=num_demos, num_repeats=num_repeats_ood)
val_data_raw_data_val_test_OOD_c_iid_boundary, val_combinations_raw_data_val_test_OOD_c_iid_boundary = du.prepare_val_samples(raw_data_val_test_OOD_c_iid_boundary, num_demos=num_demos, num_repeats=num_repeats_val)
val_data_raw_data_val_train_iid_C_OOD_boundary_inrange, val_combinations_raw_data_val_train_iid_C_OOD_boundary_inrange = du.prepare_val_samples(raw_data_val_train_iid_C_OOD_boundary_inrange, num_demos=num_demos, num_repeats=num_repeats_val)
val_data_raw_data_val_train_iid_C_OOD_boundary_outrange, val_combinations_raw_data_val_train_iid_C_OOD_boundary_outrange = du.prepare_val_samples(raw_data_val_train_iid_C_OOD_boundary_outrange, num_demos=num_demos, num_repeats=num_repeats_val)




# 定义验证数据集
validation_datasets = {
    'OOD_c_OOD_boundary_inrange': {
        'data': val_data_raw_data_val_test_OOD_c_OOD_boundary_inrange,
        'combinations': val_combinations_raw_data_val_test_OOD_c_OOD_boundary_inrange,
        'description': 'OOD c with OOD boundary (in range)'
    },
    'OOD_c_OOD_boundary_outrange': {
        'data': val_data_raw_data_val_test_OOD_c_OOD_boundary_outrange,
        'combinations': val_combinations_raw_data_val_test_OOD_c_OOD_boundary_outrange,
        'description': 'OOD c with OOD boundary (out of range)'
    },
    'OOD_c_iid_boundary': {
        'data': val_data_raw_data_val_test_OOD_c_iid_boundary,
        'combinations': val_combinations_raw_data_val_test_OOD_c_iid_boundary,
        'description': 'OOD c with IID boundary'
    },
    'iid_C_OOD_boundary_inrange': {
        'data': val_data_raw_data_val_train_iid_C_OOD_boundary_inrange,
        'combinations': val_combinations_raw_data_val_train_iid_C_OOD_boundary_inrange,
        'description': 'IID c with OOD boundary (in range)'
    },
    'iid_C_OOD_boundary_outrange': {
        'data': val_data_raw_data_val_train_iid_C_OOD_boundary_outrange,
        'combinations': val_combinations_raw_data_val_train_iid_C_OOD_boundary_outrange,
        'description': 'IID c with OOD boundary (out of range)'
    }
}
'''validation_datasets = {
    'OOD_c_OOD_boundary_inrange': {
        'data': val_data_raw_data_val_test_OOD_c_OOD_boundary_inrange,
        'combinations': val_combinations_raw_data_val_test_OOD_c_OOD_boundary_inrange,
        'description': 'OOD c with OOD boundary (in range)'
    },
}
'''
    
@jax.jit
def validate_batch(params, batch_data, batch_label, step_rng):
    """JIT编译的批量验证函数"""
    val_preds = icon_lm_forward_fn_batch(params, step_rng, batch_data)
    val_preds = val_preds[:,-50:,:]
    print(val_preds.shape)
    print(batch_label.shape)
    individual_losses = jnp.mean((val_preds - batch_label.reshape(val_preds.shape)) ** 2, axis=(1,2))

    val_loss = jnp.mean((val_preds - batch_label.reshape(val_preds.shape)) ** 2)
    
    val_poisson_metrics = jax.vmap(calculate_poisson_metrics)(
        val_preds,
        batch_label,
        batch_data.quest_cond_k[:,-50:,:],
        jnp.array([0.01] * val_preds.shape[0])
    )
    val_preds = val_preds.reshape(val_preds.shape[0],50)
    batch_label = batch_label.reshape(val_preds.shape)
    return val_loss, val_poisson_metrics,batch_label,val_preds


@jax.jit
def validate_xshot_batch(params, batch_data, batch_label, step_rng):
    """JIT编译的x-shot批量验证函数"""
    val_preds = icon_lm_forward_fn_batch(params, step_rng, batch_data)
    val_preds = val_preds[:,-50:,:]
    val_loss = jnp.mean((val_preds - batch_label.reshape(val_preds.shape)) ** 2)
    
    val_poisson_metrics = jax.vmap(calculate_poisson_metrics)(
        val_preds,
        batch_label,
        batch_data.quest_cond_k[:,-50:,:],
        jnp.array([0.01] * val_preds.shape[0])
    )
    val_preds = val_preds.reshape(val_preds.shape[0],50)
    batch_label = batch_label.reshape(val_preds.shape)
    return val_loss, val_poisson_metrics,batch_label,val_preds

import pandas as pd
import os
def save_metrics_to_csv(dataset_name, all_individual_losses, all_poisson_metrics, save_dir="/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/initial_phrase/ckpt_layers3/res_folder"):
    """
    将损失和Poisson指标保存到CSV文件
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 处理individual losses
    all_individual_losses = np.concatenate(all_individual_losses, axis=0)
    losses_df = pd.DataFrame({'individual_loss': all_individual_losses})
    losses_csv_path = os.path.join(save_dir, f'{dataset_name}_individual_losses.csv')
    losses_df.to_csv(losses_csv_path, index=False)
    print(f"Individual losses saved to {losses_csv_path}")
    
    # 处理poisson metrics
    # 将所有batch的poisson metrics合并
    combined_metrics = {
        'sign_accuracy': [],
        'integral_error': [],
        'poisson_accuracy': []
    }
    
    for batch_metrics in all_poisson_metrics:
        for key in combined_metrics:
            combined_metrics[key].extend(batch_metrics[key].tolist())
    
    # 创建DataFrame并保存
    poisson_df = pd.DataFrame(combined_metrics)
    poisson_csv_path = os.path.join(save_dir, f'{dataset_name}_poisson_metrics.csv')
    poisson_df.to_csv(poisson_csv_path, index=False)
    print(f"Poisson metrics saved to {poisson_csv_path}")


def save_metrics_to_csv(dataset_name, all_individual_losses, all_poisson_metrics, save_dir="/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/initial_phrase/ckpt_layers3/res_folder"):
    """
    将损失和Poisson指标保存到CSV文件
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 处理individual losses
    all_individual_losses = np.concatenate(all_individual_losses, axis=0)
    losses_df = pd.DataFrame({'individual_loss': all_individual_losses})
    losses_csv_path = os.path.join(save_dir, f'{dataset_name}_individual_losses.csv')
    losses_df.to_csv(losses_csv_path, index=False)
    print(f"Individual losses saved to {losses_csv_path}")
    
    # 处理poisson metrics
    # 将所有batch的poisson metrics合并
    combined_metrics = {
        'sign_accuracy': [],
        'integral_error': [],
        'poisson_accuracy': []
    }
    
    for batch_metrics in all_poisson_metrics:
        for key in combined_metrics:
            combined_metrics[key].extend(batch_metrics[key].tolist())
    
    # 创建DataFrame并保存
    poisson_df = pd.DataFrame(combined_metrics)
    poisson_csv_path = os.path.join(save_dir, f'{dataset_name}_poisson_metrics.csv')
    poisson_df.to_csv(poisson_csv_path, index=False)
    print(f"Poisson metrics saved to {poisson_csv_path}")

def save_xshot_results_to_csv(all_val_metrics_xshots, save_dir="/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/initial_phrase/ckpt_layers2/xshot_results"):
    """
    保存每个验证集的x-shot结果到CSV文件
    """
    import os
    import pandas as pd
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 对每个数据集分别保存结果
    for dataset_name, xshot_metrics in all_val_metrics_xshots.items():
        # 创建数据字典
        data_dict = {
            'n_shot': [],
            'loss': [],
            'sign_accuracy': [],
            'integral_error': [],
            'poisson_accuracy': []
        }
        
        # 收集数据
        for shot, metrics in xshot_metrics.items():
            data_dict['n_shot'].append(shot)
            for metric_name, value in metrics.items():
                data_dict[metric_name].append(value)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data_dict)
        csv_path = os.path.join(save_dir, f'{dataset_name}_xshot_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved x-shot results for {dataset_name} to {csv_path}")



def validate_dataset(dataset, params, rng, batch_size, n_shot):
    """优化后的数据集验证函数"""
    from tqdm import tqdm
    
    metrics_sum = {
        'loss': 0.0,
        'sign_accuracy': 0.0,
        'integral_error': 0.0,
        'poisson_accuracy': 0.0
    }
    
    metrics_sum_xshots = {
        i: {
            'loss': 0.0,
            'sign_accuracy': 0.0,
            'integral_error': 0.0,
            'poisson_accuracy': 0.0
        } for i in range(1, n_shot+1)
    }
    
    # 预处理所有batch
    total_samples = len(dataset['combinations'])
    num_batches = (total_samples + batch_size - 1) // batch_size

    all_preds = []
    all_labels = []
    all_individual_losses = []
    all_poisson_metrics = []
    
    with tqdm(total=num_batches, desc="Validating batches") as pbar:
        
        all_preds = []
        all_labels = []
        all_1shot_preds = []
        all_nshot_labels = []
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_samples)
            batch_combinations = dataset['combinations'][batch_start:batch_end]
            
            rng, step_rng = jax.random.split(rng)
            
            # 批量处理主验证
            batch_data, batch_label = du.get_batch_data(dataset['data'], batch_combinations)
            val_loss, val_poisson_metrics, batch_label, val_preds = validate_batch(params, batch_data, batch_label, step_rng)
            
            # 使用列表的append
            all_preds.append(val_preds)
            print(len(all_preds))
            all_labels.append(batch_label)
            print(len(all_labels))
            # 更新指标
            metrics_sum['loss'] += val_loss
            for key in val_poisson_metrics:
                metrics_sum[key] += jnp.mean(val_poisson_metrics[key])

            # 批量处理x-shot验证
            xshot_data = {}
            xshot_labels = {}
            for shot in range(1, n_shot + 1):
                xshot_data[shot], xshot_labels[shot] = du.get_batch_data_xshot(
                    dataset['data'], 
                    batch_combinations, 
                    xshot=shot
                )

            
            # 初始化存储预测结果的字典（使用列表而不是numpy数组）
            shot_predictions = {f'shot_{i}': [] for i in range(1, n_shot + 1)}
            all_nshot_labels = []  # 使用列表

            try:
                for shot in range(1, n_shot + 1):
                    val_loss_xshot, val_metrics_xshot, xshot_label, xshot_pred = validate_xshot_batch(
                        params, 
                        xshot_data[shot], 
                        xshot_labels[shot], 
                        step_rng
                    )
                    
                    # 使用列表的append
                    all_nshot_labels.append(xshot_label)
                    shot_predictions[f'shot_{shot}'].append(xshot_pred)
                    
                    # 更新metrics统计
                    metrics_sum_xshots[shot]['loss'] += val_loss_xshot
                    for key in val_metrics_xshot:
                        metrics_sum_xshots[shot][key] += jnp.mean(val_metrics_xshot[key])
                
                pbar.update(1)

                print("Finished processing all batches")
                print(f"Final length of all_preds: {len(all_preds)}")

                # 在最后再转换为numpy数组
                base_path = "/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/initial_phrase/ckpt_layers2/res_folder/shot"
                dataset_path = f"{base_path}/{dataset['description']}"
                os.makedirs(dataset_path, exist_ok=True)

                # 保存预测结果
                for shot in range(1, n_shot + 1):
                    shot_key = f'shot_{shot}'
                    if shot_predictions[shot_key]:  # 确保列表不为空
                        preds = np.concatenate(shot_predictions[shot_key], axis=0)
                        save_path = f"{dataset_path}/{shot}_preds.npy"
                        np.save(save_path, preds)

                # 保存标签
                if all_nshot_labels:  # 确保列表不为空
                    all_nshot_labels = np.concatenate(all_nshot_labels, axis=0)
                    np.save(f"{base_path}/{dataset['description']}/labels.npy", all_nshot_labels)

            except Exception as e:
                print(f"Error validating {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()

        print("Finished processing all batches")
        print(f"Final length of all_preds: {len(all_preds)}")

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            print(f"Final shapes - all_preds: {all_preds.shape}, all_labels: {all_labels.shape}")
            
            np.save(f"/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/initial_phrase/ckpt_layers2/res_folder/{dataset['description']}_preds.npy", all_preds)
            np.save(f"/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/initial_phrase/ckpt_layers2/res_folder/{dataset['description']}_labels.npy", all_labels)
        else:
            print("Warning: No predictions were collected!")

    # 计算平均指标
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
    avg_metrics_xshots = {
        shot: {k: v / num_batches for k, v in metrics.items()}
        for shot, metrics in metrics_sum_xshots.items()
    }
    
    return avg_metrics, avg_metrics_xshots

# 加载预训练模型
ckpt_path = "/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/initial_phrase/ckpt_layers2/best_model_420"
restored_params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)

# 设置随机种子以保持一致性
rng = jax.random.PRNGKey(42)

print("Starting validation...")
all_val_metrics = {}
all_val_metrics_xshots = {}

for dataset_name, dataset in validation_datasets.items():
    print(f"\nValidating on {dataset['description']}...")
    
    try:
        avg_metrics, avg_metrics_xshots = validate_dataset(
            dataset, 
            restored_params, 
            rng, 
            batch_size, 
            n_shot
        )
        
        all_val_metrics[dataset_name] = avg_metrics
        all_val_metrics_xshots[dataset_name] = avg_metrics_xshots
        
        # 记录x-shot指标
        print("\nX-shot Metrics:")
        metrics_names = ['loss', 'poisson_accuracy', 'sign_accuracy', 'integral_error']
        for dataset_name, xshot_metrics in all_val_metrics_xshots.items():
            print(f"\n{dataset_name} x-shot metrics:")
            for metric in metrics_names:
                print(f"\n{metric.replace('_', ' ').title()}:")
                for shot, shot_metrics in xshot_metrics.items():
                    print(f"{shot}-shot: {shot_metrics[metric]:.4f}", end='  ')
                print()
        
        # 打印结果
        print(f"\nResults for {dataset['description']}:")
        print(f"Loss: {avg_metrics['loss']:.4f}")
        print(f"Sign Accuracy: {avg_metrics['sign_accuracy']:.4f}")
        print(f"Integral Error: {avg_metrics['integral_error']:.4f}")
        print(f"Poisson Accuracy: {avg_metrics['poisson_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error validating {dataset_name}: {e}")
        continue

# 打印所有数据集的汇总结果
print("\n=== Final Summary ===")
print("\nFull Dataset Metrics:")
for dataset_name, metrics in all_val_metrics.items():
    print(f"\n{dataset_name} metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# 打印x-shot结果汇总
print("\n=== X-shot Summary ===")
for dataset_name, xshot_metrics in all_val_metrics_xshots.items():
    print(f"\n{dataset_name}:")
    for shot in range(1, n_shot+1):
        print(f"\n{shot}-shot results:")
        for metric_name, value in xshot_metrics[shot].items():
            print(f"{metric_name}: {value:.4f}")
save_xshot_results_to_csv(all_val_metrics_xshots)