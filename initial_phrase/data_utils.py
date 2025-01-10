import random

#import jax.treeutil as tree
import numpy as np
import jax.numpy as jnp
import jax
#import Data
from collections import namedtuple
from jax import tree_util as tree
Data = namedtuple('Data', ['demo_cond_k', 'demo_cond_v', 'demo_cond_mask', 
                          'demo_qoi_k', 'demo_qoi_v', 'demo_qoi_mask',
                          'quest_cond_k', 'quest_cond_v', 'quest_cond_mask',
                          'quest_qoi_k', 'quest_qoi_mask',])


def prepare_val_samples(val_data, num_demos=12, num_repeats=20):
    """为验证集准备采样组合"""
    val_combinations = []
    for i in range(len(val_data)):
        for _ in range(num_repeats):
            available_indices = list(range(val_data.shape[1]-1))
            demo_indices = random.sample(available_indices, num_demos)
            remaining_indices = list(set(available_indices) - set(demo_indices))
            quest_idx = random.choice(remaining_indices)
            val_combinations.append((i, demo_indices, quest_idx))
    
    return val_data, val_combinations



def prepare_val_sample_xshots(val_data, num_demos=12,xshots=10, num_repeats=20):
    """为验证集准备采样组合"""
    val_combinations = []
    for i in range(len(val_data)):
        for _ in range(num_repeats):
            available_indices = list(range(val_data.shape[1]-1))
            demo_indices = random.sample(available_indices, num_demos)
            remaining_indices = list(set(available_indices) - set(demo_indices))
            quest_idx = random.choice(remaining_indices)
            val_combinations.append((i, demo_indices, quest_idx))
    
    return val_data, val_combinations

def prepare_epoch_samples(raw_data, num_demos=12, num_repeats=3):
    """
    为整个epoch预先准备好所有采样组合
    返回: 所有样本的(demo_indices, quest_idx)列表，每个样本重复num_repeats次
    """
    total_samples = len(raw_data)
    all_combinations = []
    
    # 为每个样本生成采样组合
    for i in range(total_samples):
        for _ in range(num_repeats):
            # 随机选择demo和quest的索引
            available_indices = list(range(raw_data.shape[1]-1))  # 除去最后一个用作标签
            demo_indices = random.sample(available_indices, num_demos)
            remaining_indices = list(set(available_indices) - set(demo_indices))
            quest_idx = random.choice(remaining_indices)
            
            all_combinations.append((i, demo_indices, quest_idx))
    
    return all_combinations



def get_batch_data(raw_data, batch_combinations):
    """根据采样组合获取批次数据"""
    batch_data_list = []
    batch_label_list = []
    
    for sample_idx, demo_indices, quest_idx in batch_combinations:
        # 构建单个样本的数据
        sample_data = Data(
            demo_cond_k = raw_data[sample_idx][demo_indices,0,1:-1:2,0:1],
            demo_cond_v = raw_data[sample_idx][demo_indices,0,1:-1:2,1:2],
            demo_cond_mask = np.ones((len(demo_indices), 50)).astype(bool),
            demo_qoi_k = raw_data[sample_idx][demo_indices,1,1:-1:2,0:1],
            demo_qoi_v = raw_data[sample_idx][demo_indices,1,1:-1:2,1:2],
            demo_qoi_mask = np.ones((len(demo_indices), 50)).astype(bool),
            quest_cond_k = raw_data[sample_idx][quest_idx:quest_idx+1,0,1:-1:2,0:1],
            quest_cond_v = raw_data[sample_idx][quest_idx:quest_idx+1,0,1:-1:2,1:2],
            quest_cond_mask = np.ones((1, 50)).astype(bool),
            quest_qoi_k = raw_data[sample_idx][quest_idx:quest_idx+1,1,1:-1:2,0:1],
            quest_qoi_mask = np.ones((1, 50)).astype(bool)
        )
        batch_data_list.append(sample_data)
        batch_label_list.append(raw_data[sample_idx][quest_idx:quest_idx+1,1,1:-1:2,1:2])
    
    # 将batch中的所有样本堆叠起来
    batch_data = tree.tree_map(lambda *x: np.stack(x), *batch_data_list)
    batch_label = np.stack(batch_label_list)
    
    return batch_data, batch_label

def get_batch_data_xshot(raw_data, batch_combinations, xshot=None):
    """根据采样组合获取批次数据，支持xshot设置"""
    batch_data_list = []
    batch_label_list = []
    
    for sample_idx, demo_indices, quest_idx in batch_combinations:
        if xshot is not None:
            # 获取实际使用的demo索引（最后xshot个）
            active_indices = demo_indices[-xshot:]
            inactive_indices = demo_indices[:-xshot]
            
            # 构建数据，inactive部分用零填充
            demo_cond_k = np.concatenate([
                np.zeros_like(raw_data[sample_idx][inactive_indices,0,1:-1:2,0:1]),
                raw_data[sample_idx][active_indices,0,1:-1:2,0:1]
            ], axis=0)
            
            demo_cond_v = np.concatenate([
                np.zeros_like(raw_data[sample_idx][inactive_indices,0,1:-1:2,1:2]),
                raw_data[sample_idx][active_indices,0,1:-1:2,1:2]
            ], axis=0)
            
            demo_qoi_k = np.concatenate([
                np.zeros_like(raw_data[sample_idx][inactive_indices,1,1:-1:2,0:1]),
                raw_data[sample_idx][active_indices,1,1:-1:2,0:1]
            ], axis=0)
            
            demo_qoi_v = np.concatenate([
                np.zeros_like(raw_data[sample_idx][inactive_indices,1,1:-1:2,1:2]),
                raw_data[sample_idx][active_indices,1,1:-1:2,1:2]
            ], axis=0)
            
            # 创建mask，inactive部分设为False
            demo_mask = np.concatenate([
                np.zeros((len(inactive_indices), 50)),  # 非活跃部分mask为False
                np.ones((len(active_indices), 50))      # 活跃部分mask为True
            ], axis=0).astype(bool)
        else:
            # 原有的处理逻辑
            demo_cond_k = raw_data[sample_idx][demo_indices,0,1:-1:2,0:1]
            demo_cond_v = raw_data[sample_idx][demo_indices,0,1:-1:2,1:2]
            demo_qoi_k = raw_data[sample_idx][demo_indices,1,1:-1:2,0:1]
            demo_qoi_v = raw_data[sample_idx][demo_indices,1,1:-1:2,1:2]
            demo_mask = np.ones((len(demo_indices), 50)).astype(bool)
        
        sample_data = Data(
            demo_cond_k = demo_cond_k,
            demo_cond_v = demo_cond_v,
            demo_cond_mask = demo_mask,         
            demo_qoi_k = demo_qoi_k,
            demo_qoi_v = demo_qoi_v,
            demo_qoi_mask = demo_mask,           
            quest_cond_k = raw_data[sample_idx][quest_idx:quest_idx+1,0,1:-1:2,0:1],
            quest_cond_v = raw_data[sample_idx][quest_idx:quest_idx+1,0,1:-1:2,1:2],
            quest_cond_mask = np.ones((1, 50)).astype(bool),
            quest_qoi_k = raw_data[sample_idx][quest_idx:quest_idx+1,1,1:-1:2,0:1],
            quest_qoi_mask = np.ones((1, 50)).astype(bool)
        )
        batch_data_list.append(sample_data)
        batch_label_list.append(raw_data[sample_idx][quest_idx:quest_idx+1,1,1:-1:2,1:2])
    
    batch_data = tree.tree_map(lambda *x: np.stack(x), *batch_data_list)
    batch_label = np.stack(batch_label_list)
    
    return batch_data, batch_label