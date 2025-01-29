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
import metric_utils as metricu

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
n_shot = 12
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


file_name = "/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/train_iid_Poisson_PDE_data_train.npy"
raw_data = np.load(file_name,mmap_mode='r')


raw_data_val_test_OOD_c_OOD_boundary_inrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/test_OOD_c_OOD_boundary_inrange_Poisson_PDE_data_test_oodc.npy",mmap_mode='r')
raw_data_val_test_OOD_c_OOD_boundary_outrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/test_OOD_c_OOD_boundary_outrange_Poisson_PDE_data_test_oodc.npy",mmap_mode='r')
raw_data_val_test_OOD_c_iid_boundary = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/test_OOD_c_iid_boundary_Poisson_PDE_data_test_oodc.npy",mmap_mode='r')
raw_data_val_train_iid_C_OOD_boundary_inrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/train_iid_C_OOD_boundary_inrange_Poisson_PDE_data_test_iidc.npy",mmap_mode='r')
raw_data_val_train_iid_C_OOD_boundary_outrange = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/train_iid_C_OOD_boundary_outrange_Poisson_PDE_data_test_iidc.npy",mmap_mode='r')
#raw_data_val = np.load("/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/Poisson_PDE_data_valid_1k_-22_305.npy")


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
        poisson_metrics = jax.vmap(metricu.calculate_poisson_metrics)(
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

    # 组合adam和weight decay
    optimizer = optax.chain(
        optax.adam(learning_rate=learning_rate),
        optax.add_decayed_weights(weight_decay)
    )
    return optimizer


def create_optimizer(learning_rate=1e-4, weight_decay=1e-2, num_epochs=1000, warmup_epochs=50):
    """
    创建带有warmup和cosine decay的AdamW optimizer
    """
    # 创建学习率调度
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_epochs
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=num_epochs - warmup_epochs
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs]
    )
    
    # 创建optimizer
    optimizer = optax.chain(
        optax.adamw(
            learning_rate=schedule_fn,
            weight_decay=weight_decay,
            b1=0.9,
            b2=0.999,
            eps=1e-8
        )
    )
    return optimizer

batch_size = 32
num_demos = n_shot
num_repeats = 10
learning_rate = 1e-4
weight_decay = 1e-3
optimizer = create_optimizer(learning_rate, weight_decay)


    
opt_state = optimizer.init(params)
# 准备训练集和验证集
num_epochs = 1000

num_repeats_val=1
num_repeats_ood=1
val_data_raw_data_val_test_OOD_c_OOD_boundary_inrange, val_combinations_raw_data_val_test_OOD_c_OOD_boundary_inrange = du.prepare_val_samples(raw_data_val_test_OOD_c_OOD_boundary_inrange, num_demos=num_demos, num_repeats=num_repeats_val)
val_data_raw_data_val_test_OOD_c_OOD_boundary_outrange, val_combinations_raw_data_val_test_OOD_c_OOD_boundary_outrange = du.prepare_val_samples(raw_data_val_test_OOD_c_OOD_boundary_outrange, num_demos=num_demos, num_repeats=num_repeats_ood)
val_data_raw_data_val_test_OOD_c_iid_boundary, val_combinations_raw_data_val_test_OOD_c_iid_boundary = du.prepare_val_samples(raw_data_val_test_OOD_c_iid_boundary, num_demos=num_demos, num_repeats=num_repeats_val)
val_data_raw_data_val_train_iid_C_OOD_boundary_inrange, val_combinations_raw_data_val_train_iid_C_OOD_boundary_inrange = du.prepare_val_samples(raw_data_val_train_iid_C_OOD_boundary_inrange, num_demos=num_demos, num_repeats=num_repeats_val)
val_data_raw_data_val_train_iid_C_OOD_boundary_outrange, val_combinations_raw_data_val_train_iid_C_OOD_boundary_outrange = du.prepare_val_samples(raw_data_val_train_iid_C_OOD_boundary_outrange, num_demos=num_demos, num_repeats=num_repeats_val)



# 初始化wandb
wandb.init(
    project="icon-learning-0123",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_demos": num_demos,
        "num_repeats": num_repeats,
        "num_epochs": num_epochs,
        "model_config": model_config  
    },
    name="layer2_nhead8_12shot_adamw"
)


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


    
@jax.jit
def validate_batch(params, batch_data, batch_label, step_rng):
    """JIT编译的批量验证函数"""
    val_preds = icon_lm_forward_fn_batch(params, step_rng, batch_data)
    val_preds = val_preds[:,-50:,:]
    val_loss = jnp.mean((val_preds - batch_label.reshape(val_preds.shape)) ** 2)
    
    val_poisson_metrics = jax.vmap(metricu.calculate_poisson_metrics)(
        val_preds,
        batch_label,
        batch_data.quest_cond_k[:,-50:,:],
        jnp.array([0.01] * val_preds.shape[0])
    )
    
    return val_loss, val_poisson_metrics

@jax.jit
def validate_xshot_batch(params, batch_data, batch_label, step_rng):
    """JIT编译的x-shot批量验证函数"""
    val_preds = icon_lm_forward_fn_batch(params, step_rng, batch_data)
    val_preds = val_preds[:,-50:,:]
    val_loss = jnp.mean((val_preds - batch_label.reshape(val_preds.shape)) ** 2)
    
    val_poisson_metrics = jax.vmap(metricu.calculate_poisson_metrics)(
        val_preds,
        batch_label,
        batch_data.quest_cond_k[:,-50:,:],
        jnp.array([0.01] * val_preds.shape[0])
    )
    
    return val_loss, val_poisson_metrics

def validate_dataset(dataset, params, rng, batch_size, n_shot):
    """优化后的数据集验证函数"""
    from tqdm import tqdm  # 添加进度条
    
    metrics_sum = {
        'loss': 0.0,

        'poisson_accuracy': 0.0
    }
    
    metrics_sum_xshots = {
        i: {
            'loss': 0.0,
            'poisson_accuracy': 0.0
        } for i in range(1, n_shot+1)
    }
    
    # 预处理所有batch
    total_samples = len(dataset['combinations'])
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    # 使用tqdm创建进度条
    with tqdm(total=num_batches, desc="Validating batches") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_samples)
            batch_combinations = dataset['combinations'][batch_start:batch_end]
            
            rng, step_rng = jax.random.split(rng)
            
            # 批量处理主验证
            batch_data, batch_label = du.get_batch_data(dataset['data'], batch_combinations)
            val_loss, val_poisson_metrics = validate_batch(params, batch_data, batch_label, step_rng)
            
            # 更新指标
            metrics_sum['loss'] += val_loss
            for key in val_poisson_metrics:
                metrics_sum[key] += jnp.mean(val_poisson_metrics[key])
            
            # 批量处理x-shot验证
            xshot_data = {}
            xshot_labels = {}
            for shot in range(1, n_shot+1):
                xshot_data[shot], xshot_labels[shot] = du.get_batch_data_xshot(
                    dataset['data'], 
                    batch_combinations, 
                    xshot=shot
                )
            
            # 并行处理所有x-shot验证
            for shot in range(1, n_shot+1):
                val_loss_xshot, val_metrics_xshot = validate_xshot_batch(
                    params, 
                    xshot_data[shot], 
                    xshot_labels[shot], 
                    step_rng
                )
                
                metrics_sum_xshots[shot]['loss'] += val_loss_xshot
                for key in val_metrics_xshot:
                    metrics_sum_xshots[shot][key] += jnp.mean(val_metrics_xshot[key])
            
            pbar.update(1)
    
    # 计算平均指标
    avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
    avg_metrics_xshots = {
        shot: {k: v / num_batches for k, v in metrics.items()}
        for shot, metrics in metrics_sum_xshots.items()
    }
    
    return avg_metrics, avg_metrics_xshots

ckpt_dir = "/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/initial_phrase/ckpt/ckpt_layers3_adamw_12shot"
os.makedirs(ckpt_dir, exist_ok=True)
best_val_loss = float('inf')
# 设置全局随机种子
import random
random.seed(42)
np.random.seed(42)

# 修改训练循环中的随机性控制
for epoch in range(num_epochs):
    # 使用确定性的随机数生成
    rng = jax.random.PRNGKey(epoch+1)  # 每个epoch使用不同但确定的key
    
    # 准备训练样本时使用确定性的随机打乱
    all_combinations = du.prepare_epoch_samples(raw_data, num_demos, num_repeats)
    all_combinations = all_combinations[:int(len(all_combinations))]
    
    # 使用numpy的RandomState来确保shuffle的确定性
    rng_shuffle = np.random.RandomState(epoch)
    rng_shuffle.shuffle(all_combinations)
    
    train_loss = 0.0
    num_train_batches = 0
    
    for batch_start in range(0, len(all_combinations), batch_size):
        # 为每个batch生成确定性的随机key
        batch_rng = jax.random.fold_in(rng, batch_start)
        batch_end = min(batch_start + batch_size, len(all_combinations))
        batch_combinations = all_combinations[batch_start:batch_end]
        batch_data, batch_label = du.get_batch_data(raw_data, batch_combinations)
        params, opt_state, loss, poisson_metrics = train_step(params, opt_state, batch_data, batch_label, batch_rng)
        
        train_loss += loss
        num_train_batches += 1
        
        # 记录每个batch的loss
        wandb.log({
            "batch_train_loss": loss,
            "epoch": epoch,
            "batch": num_train_batches,
            "train_poisson_accuracy": poisson_metrics['poisson_accuracy']
        })
        
        if num_train_batches % 10 == 0:
            print(f"Epoch {epoch}, Batch {num_train_batches}, Train Loss: {loss:.4f}")
    
    avg_train_loss = train_loss / num_train_batches
    
    # 每5个epoch进行一次验证
    if epoch % 5 == 0:
        print(f"\nEpoch {epoch} Validation:")
        all_val_metrics = {}
        all_val_metrics_xshots = {}
        
        for dataset_name, dataset in validation_datasets.items():
            print(f"\nValidating on {dataset['description']}...")
            
            try:
                avg_metrics, avg_metrics_xshots = validate_dataset(
                    dataset, 
                    params, 
                    rng, 
                    batch_size, 
                    n_shot
                )
                
                all_val_metrics[dataset_name] = avg_metrics
                all_val_metrics_xshots[dataset_name] = avg_metrics_xshots
                
                # 记录基础验证指标
                wandb.log({
                    f"val/{dataset_name}/loss": float(avg_metrics['loss']),
                    f"val/{dataset_name}/poisson_accuracy": float(avg_metrics['poisson_accuracy']),
                    "epoch": epoch
                })
                
                # 记录x-shot指标
                print("\nX-shot Metrics:")
                metrics_names = ['loss', 'poisson_accuracy']
                for dataset_name, xshot_metrics in all_val_metrics_xshots.items():
                    print(f"\n{dataset_name} x-shot metrics:")
                    for metric in metrics_names:
                        
                        # 打印输出
                        print(f"\n{metric.replace('_', ' ').title()}:")
                        for shot, shot_metrics in xshot_metrics.items():
                            print(f"{shot}-shot: {shot_metrics[metric]:.4f}", end='  ')
                        print()
                
                # 打印结果
                print(f"\nResults for {dataset['description']}:")
                print(f"Loss: {avg_metrics['loss']:.4f}")
                print(f"Poisson Accuracy: {avg_metrics['poisson_accuracy']:.4f}")
                
            except Exception as e:
                print(f"Error validating {dataset_name}: {e}")
                continue
        
        # 计算所有验证集的平均损失
        current_val_loss = np.mean([metrics['loss'] for metrics in all_val_metrics.values()])
        
        # 保存最佳模型
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir,
                target=params,
                step=epoch,
                prefix='best_model_',
                overwrite=True
            )
            print(f"\nNew best model saved! Val Loss: {best_val_loss:.4f}")


        # wandb记录也相应更新
        wandb_log_dict = {
            **{f"metrics/val_loss_{i}shot": avg_metrics_xshots[i]['loss'] for i in range(1, n_shot+1)},
            **{f"metrics/val_poisson_accuracy_{i}shot": avg_metrics_xshots[i]['poisson_accuracy'] for i in range(1, n_shot+1)},
        }
        wandb.log(wandb_log_dict)
        
        # 打印当前epoch的主要指标
        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {current_val_loss:.4f} (Best: {best_val_loss:.4f})")
        
        print("\nFull Dataset Metrics:")
        for dataset_name, metrics in all_val_metrics.items():
            print(f"\n{dataset_name} metrics:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
    
    # 每10个epoch保存一次常规检查点
    if epoch % 10 == 0:
        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=params,
            step=epoch,
            prefix='checkpoint_',
            keep=3
        )
    
    # 清理内存
    del all_combinations
    del batch_data
    del batch_label
    import gc
    gc.collect()

# 训练结束时保存最终模型
checkpoints.save_checkpoint(
    ckpt_dir=ckpt_dir,
    target=params,
    step=num_epochs-1,
    prefix='final_model_',
    overwrite=True
)

wandb.finish()