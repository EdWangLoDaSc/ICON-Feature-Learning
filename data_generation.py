# Converted from notebook: data_generation.ipynb

# Cell
from einshape import jax_einshape as einshape
from jax import lax
import jax
from functools import partial
import sys
sys.path.append('../')
import haiku as hk
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

from functools import partial
from jax import lax
import os

# Cell
import os
import sys
sys.path.append('/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/')

os.makedirs('/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/', exist_ok=True)

"""
# Poisson equation
"""

"""
## Utils -- k有2个value，只适用于forward problem+inverse problem
"""

# Cell
@partial(jax.jit, static_argnames=('num','kernel'))
def generate_gaussian_process(key, ts, num, kernel, k_sigma, k_l):
  '''
  ts: 1D array (length,)
  out: Gaussian process samples, 2D array (num, length)
  '''
  length = len(ts)
  mean = jnp.zeros((num,length))
  # cov = rbf_kernel(ts[:, None], ts[:, None], sigma=k_sigma, l=k_l)
  cov = kernel(ts, ts, sigma=k_sigma, l=k_l)
  cov = einshape('ii->nii', cov, n = num)
  out = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(num,), method='svd')
  return out


def rbf_kernel_jax(x1, x2, sigma, l): # Define the covariance function
    """
    Radial basis function kernel, only support 1D x1 and x2
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    sq_norm = (xx1-xx2)**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)


def tridiagonal_solve(dl, d, du, b):
  """Pure JAX implementation of `tridiagonal_solve`."""
  prepend_zero = lambda x: jnp.append(jnp.zeros([1], dtype=x.dtype), x[:-1])
  fwd1 = lambda tu_, x: x[1] / (x[0] - x[2] * tu_)
  fwd2 = lambda b_, x: (x[0] - x[3] * b_) / (x[1] - x[3] * x[2])
  bwd1 = lambda x_, x: x[0] - x[1] * x_
  double = lambda f, args: (f(*args), f(*args))

  # Forward pass.
  _, tu_ = lax.scan(lambda tu_, x: double(fwd1, (tu_, x)),
                    du[0] / d[0],
                    (d, du, dl),
                    unroll=32)

  _, b_ = lax.scan(lambda b_, x: double(fwd2, (b_, x)),
                  b[0] / d[0],
                  (b, d, prepend_zero(tu_), dl),
                  unroll=32)

  # Backsubstitution.
  _, x_ = lax.scan(lambda x_, x: double(bwd1, (x_, x)),
                  b_[-1],
                  (b_[::-1], tu_[::-1]),
                  unroll=32)

  return x_[::-1]


@partial(jax.jit, static_argnames=("N"))
def solve_poisson(L, N, u_left, u_right, c): # Poisson equation solver
    '''
    du/dxx = c over domain [0,L]
    c: spatially varying function, size N-1,
    u_left, u_right: boundary conditions.
    the output is the full solution, (N+1) grid point values.
    '''
    dx = L / N
    # x = jnp.linspace(0, L, N+1)

    # finite difference matrix
    du = jnp.array([1.0] * (N-2) + [0.0])
    dl =  jnp.array([0.0] + [1.0] * (N-2))
    d = - 2.0 * jnp.ones((N-1,))

    b = c*dx*dx
    b = b.at[0].add(-u_left)
    b = b.at[-1].add(-u_right)

    out_u = tridiagonal_solve(dl, d, du, b)
    u = jnp.pad(out_u, (1, 1), mode='constant', constant_values=(u_left, u_right))
    return u

solve_poisson_batch = jax.jit(jax.vmap(solve_poisson, in_axes=(None, None, None, None, 0)), static_argnums=(1,))





def generate_pde_poisson(seed, eqns, length, dx, num, file_name):
  '''
  du/dxx = c(x) over domain [0,L]
  c(x) : spatially varying coefficient, size N-1,
          we use GP to sample c(x)
  u_left, u_right: boundary conditions.
  the output is the full solution, (N+1) grid point values.
  '''
  N = length
  L = length * dx
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  all_xs = []; all_cs = []; all_us = []

  """
  This is a 5D array. [eqns, num, 2, N+1, 2]
  First dimension: number of operators (or equations) 
  Second dimension: number of (u, c) pairs chosen for each operator
  Third dimension: index 0 indicates u; index 1 indicates c
  Fourth dimension: number of positions where x,u,c values are saved for each particular equation
  Fifth dimension: index 0 indicates x; index 1 indicates u or c value at the specified position
  """

  index = -1
  for i, (coeff_ul, coeff_ur) in enumerate(zip(coeffs_ul, coeffs_ur)):
    xs = jnp.linspace(0.0, L, N+1)# (N+1,)
    cs = generate_gaussian_process(next(rng), xs, num, kernel = rbf_kernel_jax, k_sigma = 2.0, k_l = 0.5) # (num, N+1)
    us = solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
    all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
    all_cs.append(einshape("ij->ijk", cs, k = 1)) # (num, N+1, 1)
    all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)

  all_xs = np.array(all_xs) # (eqns, num, N+1, 1)
  all_cs = np.array(all_cs) # (eqns, num, N+1, 1)
  all_us = np.array(all_us) # (eqns, num, N+1, 1)

  all_u = np.concatenate([all_xs, all_us], axis = -1) # (eqns, num, N+1, 2)
  all_c = np.concatenate([all_xs, all_cs], axis = -1) # (eqns, num, N+1, 2)
  data = np.concatenate([all_u[:,:,None,:,:], all_c[:,:,None,:,:]], axis = 2) # (eqns, num, 2, N+1, 2)
  file_path = f'./{file_name}.npy'
  np.save(file_path, data)
  print("Data saved successfully")
  


def generate_pde_poisson(seed, eqns, length, dx, num, file_name, ood_type='all'):
  '''
  du/dxx = c(x) over domain [0,L]
  c(x) : spatially varying coefficient, size N-1,
          we use GP to sample c(x)
  u_left, u_right: boundary conditions.
  the output is the full solution, (N+1) grid point values.
  '''
  N = length
  L = length * dx
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  if ood_type == 'boundary' or ood_type == 'all':
        coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.5, maxval = 1.5)  # 扩大范围
        coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.5, maxval = 1.5)
  else:
        coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
        coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    
  all_xs = []; all_cs = []; all_us = []

  """
  This is a 5D array. [eqns, num, 2, N+1, 2]
  First dimension: number of operators (or equations) 
  Second dimension: number of (u, c) pairs chosen for each operator
  Third dimension: index 0 indicates u; index 1 indicates c
  Fourth dimension: number of positions where x,u,c values are saved for each particular equation
  Fifth dimension: index 0 indicates x; index 1 indicates u or c value at the specified position
  """

  index = -1
  for i, (coeff_ul, coeff_ur) in enumerate(zip(coeffs_ul, coeffs_ur)):
    '''xs = jnp.linspace(0.0, L, N+1)# (N+1,)
    cs = generate_gaussian_process(next(rng), xs, num, kernel = rbf_kernel_jax, k_sigma = 2.0, k_l = 0.5) # (num, N+1)
    us = solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
    all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
    all_cs.append(einshape("ij->ijk", cs, k = 1)) # (num, N+1, 1)
    all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)'''
    xs = jnp.linspace(0.0, L, N+1)# (N+1,)
    if ood_type == 'gp' or ood_type == 'all':
        k_sigma = 4.0  # 增大方差
        k_l = 0.5    # 减小长度尺度，使函数变化更剧烈
    else:
        k_sigma = 2.0
        k_l = 0.5
    cs = generate_gaussian_process(next(rng), xs, num, rbf_kernel_jax, k_sigma, k_l) # (num, N+1)
    us = solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
    all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
    
    # 为c添加标识符1
    cs_with_id = np.zeros((cs.shape[0], cs.shape[1], 2))  # (num, N+1, 2)
    cs_with_id[:,:,0] = cs
    cs_with_id[:,:,1] = 1  # 标识这是c
    all_cs.append(cs_with_id)
    
    # 为u添加标识符0
    us_with_id = np.zeros((us.shape[0], us.shape[1], 2))  # (num, N+1, 2)
    us_with_id[:,:,0] = us
    us_with_id[:,:,1] = 0  # 标识这是u
    all_us.append(us_with_id)

  all_xs = np.array(all_xs) # (eqns, num, N+1, 1)
  all_cs = np.array(all_cs) # (eqns, num, N+1, 1)
  all_us = np.array(all_us) # (eqns, num, N+1, 1)

  all_u = np.concatenate([all_xs, all_us], axis = -1) # (eqns, num, N+1, 2)
  all_c = np.concatenate([all_xs, all_cs], axis = -1) # (eqns, num, N+1, 2)
  data = np.concatenate([all_u[:,:,None,:,:], all_c[:,:,None,:,:]], axis = 2) # (eqns, num, 2, N+1, 2)
  root_dir = '/home/wanghanyang.01/ICON-Feature-Learning/data_generation/data'
  
  file_path = f'{root_dir}/data_possion_jax/{file_name}.npy'
  np.save(file_path, data)
  print("Data saved successfully")


"""
## Utils -- k有1个value，只适用于forward problem
"""

# Cell
@partial(jax.jit, static_argnames=('num','kernel'))
def generate_gaussian_process(key, ts, num, kernel, k_sigma, k_l):
  '''
  ts: 1D array (length,)
  out: Gaussian process samples, 2D array (num, length)
  '''
  length = len(ts)
  mean = jnp.zeros((num,length))
  # cov = rbf_kernel(ts[:, None], ts[:, None], sigma=k_sigma, l=k_l)
  cov = kernel(ts, ts, sigma=k_sigma, l=k_l)
  cov = einshape('ii->nii', cov, n = num)
  out = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(num,), method='svd')
  return out


def rbf_kernel_jax(x1, x2, sigma, l): # Define the covariance function
    """
    Radial basis function kernel, only support 1D x1 and x2
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    sq_norm = (xx1-xx2)**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)


def tridiagonal_solve(dl, d, du, b):
  """Pure JAX implementation of `tridiagonal_solve`."""
  prepend_zero = lambda x: jnp.append(jnp.zeros([1], dtype=x.dtype), x[:-1])
  fwd1 = lambda tu_, x: x[1] / (x[0] - x[2] * tu_)
  fwd2 = lambda b_, x: (x[0] - x[3] * b_) / (x[1] - x[3] * x[2])
  bwd1 = lambda x_, x: x[0] - x[1] * x_
  double = lambda f, args: (f(*args), f(*args))

  # Forward pass.
  _, tu_ = lax.scan(lambda tu_, x: double(fwd1, (tu_, x)),
                    du[0] / d[0],
                    (d, du, dl),
                    unroll=32)

  _, b_ = lax.scan(lambda b_, x: double(fwd2, (b_, x)),
                  b[0] / d[0],
                  (b, d, prepend_zero(tu_), dl),
                  unroll=32)

  # Backsubstitution.
  _, x_ = lax.scan(lambda x_, x: double(bwd1, (x_, x)),
                  b_[-1],
                  (b_[::-1], tu_[::-1]),
                  unroll=32)

  return x_[::-1]


@partial(jax.jit, static_argnames=("N"))
def solve_poisson(L, N, u_left, u_right, c): # Poisson equation solver
    '''
    du/dxx = c over domain [0,L]
    c: spatially varying function, size N-1,
    u_left, u_right: boundary conditions.
    the output is the full solution, (N+1) grid point values.
    '''
    dx = L / N
    # x = jnp.linspace(0, L, N+1)

    # finite difference matrix
    du = jnp.array([1.0] * (N-2) + [0.0])
    dl =  jnp.array([0.0] + [1.0] * (N-2))
    d = - 2.0 * jnp.ones((N-1,))

    b = c*dx*dx
    b = b.at[0].add(-u_left)
    b = b.at[-1].add(-u_right)

    out_u = tridiagonal_solve(dl, d, du, b)
    u = jnp.pad(out_u, (1, 1), mode='constant', constant_values=(u_left, u_right))
    return u

solve_poisson_batch = jax.jit(jax.vmap(solve_poisson, in_axes=(None, None, None, None, 0)), static_argnums=(1,))


# Cell



def generate_out_of_range_values(key, eqns):
    """生成边界条件,确保每对(ul,ur)中至少有一个在[-1,1]范围外"""
    # 为每个方程生成两个边界值
    coeffs = jax.random.uniform(key, (eqns, 2), minval=-2.0, maxval=2.0)
    
    # 检查每对值是否都在[-1,1]内
    is_inside = (coeffs >= -1.0) & (coeffs <= 1.0)
    both_inside = jnp.all(is_inside, axis=1)
    
    # 随机选择要重新生成的是第一个还是第二个值
    which_to_regenerate = jax.random.randint(key, (eqns,), 0, 2)
    
    # 生成新的外部值 [-2,-1]∪[1,2]
    new_values = jax.random.uniform(key, (eqns,), minval=1.0, maxval=2.0)
    new_values = new_values * jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(eqns,))
    
    # 根据随机选择更新值
    update_first = both_inside & (which_to_regenerate == 0)
    update_second = both_inside & (which_to_regenerate == 1)
    
    coeffs = coeffs.at[update_first, 0].set(new_values[update_first])
    coeffs = coeffs.at[update_second, 1].set(new_values[update_second])
    
    return coeffs[:, 0], coeffs[:, 1]  # 返回ul和ur


# Cell
df = np.load(f'/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/{dataset_type_i}_{train_file_name}.npy')

print(df.shape)

"""
# 改进下
"""

# Cell
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def generate_pde_poisson(seed, num_c_per_k, num_boundary_pairs, length, dx, file_name, dataset_type='train_iid'):
    """
    生成Poisson方程数据集，支持多个k_sigma配置
    
    参数:
    - num_c_per_k: 每个k_sigma配置下生成的c(x)数量
    - num_boundary_pairs: 每个c对应的边界条件对数量
    - dataset_type: 数据集类型 ['train_iid', 'test_ood']
    """
    N = length
    L = length * dx
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    rng_ood = hk.PRNGSequence(jax.random.PRNGKey(seed+1))
    
    all_xs = []; all_cs = []; all_us = []
    
    # 定义不同的k_sigma配置
    if dataset_type == 'train_iid' or dataset_type == 'train_iid_C_OOD_boundary_inrange' or dataset_type == 'train_iid_C_OOD_boundary_outrange':
        # 训练集使用m个k_sigma配置
        k_configs = [
            {'k_sigma': 1.5, 'k_l': 0.5},
            {'k_sigma': 2.0, 'k_l': 0.5},
            {'k_sigma': 2.5, 'k_l': 0.5},
            {'k_sigma': 3.0, 'k_l': 0.5},
            {'k_sigma': 3.5, 'k_l': 0.5},
            {'k_sigma': 4.0, 'k_l': 0.5},
            {'k_sigma': 4.5, 'k_l': 0.5},
            {'k_sigma': 5.0, 'k_l': 0.5},
        ]
    elif dataset_type == 'test_OOD_c_iid_boundary' or dataset_type == 'test_OOD_c_OOD_boundary_inrange' or dataset_type == 'test_OOD_c_OOD_boundary_outrange':
        # 测试集使用n个不同的k_sigma配置
        k_configs = [
            {'k_sigma': 1.75, 'k_l': 0.5},
            {'k_sigma': 2.25, 'k_l': 0.5},
            {'k_sigma': 2.75, 'k_l': 0.5},
            {'k_sigma': 3.25, 'k_l': 0.5},
            {'k_sigma': 3.75, 'k_l': 0.5},
            {'k_sigma': 4.25, 'k_l': 0.5},
            {'k_sigma': 4.75, 'k_l': 0.5},
            {'k_sigma': 5.25, 'k_l': 0.5},
        ]
    
    # 生成空间网格
    xs = jnp.linspace(0.0, L, N+1)
    
    # 为每个k_sigma配置生成数据
    from tqdm import tqdm
    
    all_xs_list = []
    all_cs_list = []
    all_us_list = []
    for k_config in k_configs:
        if dataset_type == 'train_iid' or dataset_type == 'test_OOD_c_iid_boundary':
            coeffs_ul = jax.random.uniform(next(rng), (num_c_per_k,), minval=-1.0, maxval=1.0)
            coeffs_ur = jax.random.uniform(next(rng), (num_c_per_k,), minval=-1.0, maxval=1.0)
        elif dataset_type in ['train_iid_C_OOD_boundary_inrange', 'test_OOD_c_OOD_boundary_inrange']:
            coeffs_ul = jax.random.uniform(next(rng_ood), (num_c_per_k,), minval=-1.0, maxval=1.0)
            coeffs_ur = jax.random.uniform(next(rng_ood), (num_c_per_k,), minval=-1.0, maxval=1.0)
        elif dataset_type in ['train_iid_C_OOD_boundary_outrange', 'test_OOD_c_OOD_boundary_outrange']:

            coeffs_ul, coeffs_ur = generate_out_of_range_values(next(rng_ood), num_c_per_k)
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        
        for i,(coeff_ul, coeff_ur) in enumerate(zip(coeffs_ul, coeffs_ur)):
            xs = jnp.linspace(0.0, L, N+1)
            # 生成c(x)样本
            cs = generate_gaussian_process(
                next(rng), 
                xs, 
                num=num_boundary_pairs,
                kernel=rbf_kernel_jax,
                k_sigma=k_config['k_sigma'],
                k_l=k_config['k_l']
            )  # shape: (num_c_per_k, N+1)
            

            us = solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
            all_xs.append(einshape("i->jik", xs, j = num_boundary_pairs, k = 1)) # (num, N+1, 1)
    
            cs_with_id = np.zeros((cs.shape[0], cs.shape[1], 1))  # (num, N+1, 2)
            cs_with_id[:,:,0] = cs
            all_cs.append(cs_with_id)
        
            us_with_id = np.zeros((us.shape[0], us.shape[1], 1))  # (num, N+1, 2)
            us_with_id[:,:,0] = us
            all_us.append(us_with_id)
            
            
        #all_xs_list.append(all_xs)
        #all_cs_list.append(all_cs)
        #all_us_list.append(all_us)
           # 保存边界条件
            
        
    np.save(f'/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/ur_ul/ul_{dataset_type}.npy', coeffs_ul)
    np.save(f'/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/ur_ul/ur_{dataset_type}.npy', coeffs_ur)
        
        
        

    # 转换为numpy数组并合并
    total_samples = len(k_configs) * num_c_per_k * num_boundary_pairs
    all_xs = np.array(all_xs)  # (total_samples, 1, N+1, 1)
    all_cs = np.array(all_cs)  # (total_samples, 1, N+1, 1)
    all_us = np.array(all_us)  # (total_samples, 1, N+1, 1)
    
    print(all_xs.shape)
    print(all_cs.shape)
    print(all_us.shape)


    all_u = np.concatenate([all_xs, all_us], axis=-1).reshape(num_c_per_k*len(k_configs), num_boundary_pairs, 1, N+1, 2)
    all_c = np.concatenate([all_xs, all_cs], axis=-1).reshape(num_c_per_k*len(k_configs), num_boundary_pairs, 1, N+1, 2)
    data = np.concatenate([all_u, all_c], axis=2)  # (num_c_per_k*len(k_configs), num_boundary_pairs, 2, N+1, 3)

    # 保存数据
    file_path = f'/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/{dataset_type}_{file_name}.npy'
    np.save(file_path, data)
    print(f"Data saved successfully with {len(k_configs)} k_sigma configurations")
    print(data.shape)
    return data


test_seed = 1
test_eqns = 50  
test_xs = 100
test_dx = 0.01
test_num_ucpair = 100
test_file_name = "Poisson_PDE_data_test_oodc"
for dataset_type_i in ['test_OOD_c_OOD_boundary_inrange','test_OOD_c_OOD_boundary_outrange','test_OOD_c_iid_boundary']:
    print(f"Generating {dataset_type_i} data...")
    train_data = generate_pde_poisson(
        seed=test_seed,
        num_c_per_k=test_eqns,
        num_boundary_pairs=test_num_ucpair,  # 添加缺失的参数
        length=test_xs,
        dx=test_dx,
        file_name=test_file_name,
        dataset_type=dataset_type_i
    )
    

train_seed = 1
train_eqns = 125  
train_xs = 100
train_dx = 0.01
train_num_ucpair = 100
train_file_name = "Poisson_PDE_data_train"
dataset_type='train_iid'



for dataset_type_i in ['train_iid']:
# 生成训练集
    print(f"Generating {dataset_type_i} data...")
    train_data = generate_pde_poisson(
        seed=train_seed,
        num_c_per_k=train_eqns,
        num_boundary_pairs=train_num_ucpair,  # 添加缺失的参数
        length=train_xs,
        dx=train_dx,
        file_name=train_file_name,
        dataset_type=dataset_type_i
    )



test_seed = 1
test_eqns = 125  
test_xs = 100
test_dx = 0.01
test_num_ucpair = 100
test_file_name = "Poisson_PDE_data_test_iidc"
for dataset_type_i in ['train_iid_C_OOD_boundary_inrange','train_iid_C_OOD_boundary_outrange']:
    print(f"Generating {dataset_type_i} data...")
    train_data = generate_pde_poisson(
        seed=test_seed,
        num_c_per_k=test_eqns,
        num_boundary_pairs=test_num_ucpair,  # 添加缺失的参数
        length=test_xs,
        dx=test_dx,
        file_name=test_file_name,
        dataset_type=dataset_type_i
    )
    



# Cell

df = np.load(f'/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/test_OOD_c_OOD_boundary_outrange_Poisson_PDE_data_test_oodc.npy')

print(df.shape)
from matplotlib import pyplot as plt
for i in range(2):
  plt.figure()
  for j in range(5):
    plt.plot(df[i,j,0,:,0], df[i,j,0,:,1], label = "u", linestyle = "--")
    plt.plot(df[i,j,1,:,0], df[i,j,1,:,1], label = "c", linestyle = "-")
  plt.show()

# Cell
from matplotlib import pyplot as plt
for i in range(2):
  plt.figure()
  for j in range(5):
    plt.plot(df[i,j,0,:,0], df[i,j,0,:,1], label = "u", linestyle = "--")
    plt.plot(df[i,j,1,:,0], df[i,j,1,:,1], label = "c", linestyle = "-")
  plt.show()

# Cell
def generate_pde_poisson(seed, eqns, length, dx, num, file_name, ood_type='all'):
  '''
  du/dxx = c(x) over domain [0,L]
  c(x) : spatially varying coefficient, size N-1,
          we use GP to sample c(x)
  u_left, u_right: boundary conditions.
  the output is the full solution, (N+1) grid point values.
  '''
  N = length
  L = length * dx
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  if ood_type == 'boundary' or ood_type == 'all':
        coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -2, maxval = 2)  # 扩大范围 +-1.5
        coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -2, maxval = 2)  # +-2
        
        # To-do:if coeffs_ul or coeffs_ur is -1~1, reset
  else:
        coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
        coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  
  # 
  all_xs = []; all_cs = []; all_us = []
  from tqdm import tqdm

  """
  This is a 5D array. [eqns, num, 2, N+1, 2]
  First dimension: number of operators (or equations) 
  Second dimension: number of (u, c) pairs chosen for each operator
  Third dimension: index 0 indicates u; index 1 indicates c
  Fourth dimension: number of positions where x,u,c values are saved for each particular equation
  Fifth dimension: index 0 indicates x; index 1 indicates u or c value at the specified position
  """
  index = -1
  for i, (coeff_ul, coeff_ur) in tqdm(enumerate(zip(coeffs_ul, coeffs_ur))):
    '''xs = jnp.linspace(0.0, L, N+1)# (N+1,)
    cs = generate_gaussian_process(next(rng), xs, num, kernel = rbf_kernel_jax, k_sigma = 2.0, k_l = 0.5) # (num, N+1)
    us = solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
    all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
    all_cs.append(einshape("ij->ijk", cs, k = 1)) # (num, N+1, 1)
    all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)'''
    xs = jnp.linspace(0.0, L, N+1)# (N+1,)
    if ood_type == 'gp' or ood_type == 'all':
        k_sigma = 3.0  # 增大方差 3.0
        k_l = 0.5   # 减小长度尺度，使函数变化更剧烈 0.5
    else:
        k_sigma = 2.0
        k_l = 0.5
    cs = generate_gaussian_process(next(rng), xs, num, rbf_kernel_jax, k_sigma, k_l) # (num, N+1)
    us = solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
    all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
    
    # 为c添加标识符1
    cs_with_id = np.zeros((cs.shape[0], cs.shape[1], 1))  # (num, N+1, 2)
    cs_with_id[:,:,0] = cs
    all_cs.append(cs_with_id)
    
    # 为u添加标识符0
    us_with_id = np.zeros((us.shape[0], us.shape[1], 1))  # (num, N+1, 2)
    us_with_id[:,:,0] = us
    all_us.append(us_with_id)

  all_xs = np.array(all_xs) # (eqns, num, N+1, 1)
  all_cs = np.array(all_cs) # (eqns, num, N+1, 1)
  all_us = np.array(all_us) # (eqns, num, N+1, 1)

  all_u = np.concatenate([all_xs, all_us], axis = -1) # (eqns, num, N+1, 2)
  all_c = np.concatenate([all_xs, all_cs], axis = -1) # (eqns, num, N+1, 2)
  data = np.concatenate([all_u[:,:,None,:,:], all_c[:,:,None,:,:]], axis = 2) # (eqns, num, 2, N+1, 2)

  file_path = f'/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/{file_name}.npy'
  np.save(file_path, data)
  print("Data saved successfully")


# Cell
def generate_pde_poisson(seed, eqns, length, dx, num, file_name, ood_type='all'):
  '''
  du/dxx = c(x) over domain [0,L]
  c(x) : spatially varying coefficient, size N-1,
          we use GP to sample c(x)
  u_left, u_right: boundary conditions.
  the output is the full solution, (N+1) grid point values.
  '''
  N = length
  L = length * dx
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  if ood_type == 'boundary' or ood_type == 'all':
        coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -2, maxval = 2)  # 扩大范围 +-1.5
        coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -2, maxval = 2)  # +-2
        
        # To-do:if coeffs_ul or coeffs_ur is -1~1, reset
  else:
        coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
        coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  
  # 
  all_xs = []; all_cs = []; all_us = []
  from tqdm import tqdm

  """
  This is a 5D array. [eqns, num, 2, N+1, 2]
  First dimension: number of operators (or equations) 
  Second dimension: number of (u, c) pairs chosen for each operator
  Third dimension: index 0 indicates u; index 1 indicates c
  Fourth dimension: number of positions where x,u,c values are saved for each particular equation
  Fifth dimension: index 0 indicates x; index 1 indicates u or c value at the specified position
  """
  index = -1
  for i, (coeff_ul, coeff_ur) in tqdm(enumerate(zip(coeffs_ul, coeffs_ur))):
    '''xs = jnp.linspace(0.0, L, N+1)# (N+1,)
    cs = generate_gaussian_process(next(rng), xs, num, kernel = rbf_kernel_jax, k_sigma = 2.0, k_l = 0.5) # (num, N+1)
    us = solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
    all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
    all_cs.append(einshape("ij->ijk", cs, k = 1)) # (num, N+1, 1)
    all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)'''
    xs = jnp.linspace(0.0, L, N+1)# (N+1,)
    if ood_type == 'gp' or ood_type == 'all':
        k_sigma = 3.0  # 增大方差 3.0
        k_l = 0.5   # 减小长度尺度，使函数变化更剧烈 0.5
    else:
        k_sigma = 2.0
        k_l = 0.5
    cs = generate_gaussian_process(next(rng), xs, num, rbf_kernel_jax, k_sigma, k_l) # (num, N+1)
    us = solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
    all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
    
    # 为c添加标识符1
    cs_with_id = np.zeros((cs.shape[0], cs.shape[1], 1))  # (num, N+1, 2)
    cs_with_id[:,:,0] = cs
    all_cs.append(cs_with_id)
    
    # 为u添加标识符0
    us_with_id = np.zeros((us.shape[0], us.shape[1], 1))  # (num, N+1, 2)
    us_with_id[:,:,0] = us
    all_us.append(us_with_id)

  all_xs = np.array(all_xs) # (eqns, num, N+1, 1)
  all_cs = np.array(all_cs) # (eqns, num, N+1, 1)
  all_us = np.array(all_us) # (eqns, num, N+1, 1)

  all_u = np.concatenate([all_xs, all_us], axis = -1) # (eqns, num, N+1, 2)
  all_c = np.concatenate([all_xs, all_cs], axis = -1) # (eqns, num, N+1, 2)
  data = np.concatenate([all_u[:,:,None,:,:], all_c[:,:,None,:,:]], axis = 2) # (eqns, num, 2, N+1, 2)

  file_path = f'/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/{file_name}.npy'
  np.save(file_path, data)
  print("Data saved successfully")

# Cell
def generate_pde_poisson(seed, eqns, length, dx, num, file_name, ood_type='all'):
  '''
  du/dxx = c(x) over domain [0,L]
  c(x) : spatially varying coefficient, size N-1,
          we use GP to sample c(x)
  u_left, u_right: boundary conditions.
  the output is the full solution, (N+1) grid point values.
  '''
  N = length
  L = length * dx
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  
  rng_ood = hk.PRNGSequence(jax.random.PRNGKey(seed+1))
  
  
  if ood_type == 'boundary' or ood_type == 'all':
        coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -2, maxval = 2)  # 扩大范围 +-1.5
        coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -2, maxval = 2)  # +-2
        
        # To-do:if coeffs_ul or coeffs_ur is -1~1, reset
  else:
        coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
        coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  

  all_xs = []; all_cs = []; all_us = []
  from tqdm import tqdm

  """
  This is a 5D array. [eqns, num, 2, N+1, 2]
  First dimension: number of operators (or equations) 
  Second dimension: number of (u, c) pairs chosen for each operator
  Third dimension: index 0 indicates u; index 1 indicates c
  Fourth dimension: number of positions where x,u,c values are saved for each particular equation
  Fifth dimension: index 0 indicates x; index 1 indicates u or c value at the specified position
  """
  index = -1
  for i, (coeff_ul, coeff_ur) in tqdm(enumerate(zip(coeffs_ul, coeffs_ur))):
    '''xs = jnp.linspace(0.0, L, N+1)# (N+1,)
    cs = generate_gaussian_process(next(rng), xs, num, kernel = rbf_kernel_jax, k_sigma = 2.0, k_l = 0.5) # (num, N+1)
    us = solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
    all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
    all_cs.append(einshape("ij->ijk", cs, k = 1)) # (num, N+1, 1)
    all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)'''
    xs = jnp.linspace(0.0, L, N+1)# (N+1,)
    if ood_type == 'gp' or ood_type == 'all':
        k_sigma = 3.0  # 增大方差 3.0
        k_l = 0.5   # 减小长度尺度，使函数变化更剧烈 0.5
    else:
        k_sigma = 2.0
        k_l = 0.5
    cs = generate_gaussian_process(next(rng), xs, num, rbf_kernel_jax, k_sigma, k_l) # (num, N+1)
    print(coeff_ul)
    print(coeff_ur)
    print(cs[:,1:-1])
    print(cs[:,1:-1].shape)
    import pdb
    pdb.set_trace()
    us = solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
    all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
    
    # 为c添加标识符1
    cs_with_id = np.zeros((cs.shape[0], cs.shape[1], 1))  # (num, N+1, 2)
    cs_with_id[:,:,0] = cs
    all_cs.append(cs_with_id)
    
    # 为u添加标识符0
    us_with_id = np.zeros((us.shape[0], us.shape[1], 1))  # (num, N+1, 2)
    us_with_id[:,:,0] = us
    all_us.append(us_with_id)

  all_xs = np.array(all_xs) # (eqns, num, N+1, 1)
  all_cs = np.array(all_cs) # (eqns, num, N+1, 1)
  all_us = np.array(all_us) # (eqns, num, N+1, 1)

  all_u = np.concatenate([all_xs, all_us], axis = -1) # (eqns, num, N+1, 2)
  all_c = np.concatenate([all_xs, all_cs], axis = -1) # (eqns, num, N+1, 2)
  data = np.concatenate([all_u[:,:,None,:,:], all_c[:,:,None,:,:]], axis = 2) # (eqns, num, 2, N+1, 2)

  file_path = f'/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/{file_name}.npy'
  np.save(file_path, data)
  print("Data saved successfully")

"""
## 生成模块
"""

# Cell
valid_seed = 2  # 使用不同的随机种子
valid_eqns = 100  # 200个方程用于验证
valid_xs = 100  # 保持与训练集相同的空间离散化
valid_dx = 0.01
valid_num_ucpair = 100
valid_file_name = "Poisson_PDE_data_valid_1k_-22_305" # 1维，-2～1，3var，0.45

# 生成验证集
print("\nGenerating validation data...")
valid_data = generate_pde_poisson(
    seed=valid_seed,
    eqns=valid_eqns,
    length=valid_xs,
    dx=valid_dx,
    num=valid_num_ucpair,
    file_name=valid_file_name,
    ood_type='all'
)

# Cell
# 训练集参数
train_seed = 1
train_eqns = 2000  
train_xs = 100
train_dx = 0.01
train_num_ucpair = 100
train_file_name = "Poisson_PDE_data_train_-11_205"

# 生成训练集
print("Generating training data...")
train_data = generate_pde_poisson(
    seed=train_seed,
    eqns=train_eqns,
    length=train_xs,
    dx=train_dx,
    num=train_num_ucpair,
    file_name=train_file_name,
    ood_type='train'
)


# Cell
import numpy as np
data = np.load(f'/jumbo/yaoqingyang/hanyang/ICON-Feature-Learning/data_generation/data/data_possion_jax/Poisson_PDE_data_train_-11_205.npy')
print(data.shape)

# Cell
from matplotlib import pyplot as plt
for i in range(2):
  plt.figure()
  for j in range(5):
    plt.plot(df[i,j,0,:,0], df[i,j,0,:,1], label = "u", linestyle = "--")
    plt.plot(df[i,j,1,:,0], df[i,j,1,:,1], label = "c", linestyle = "-")
  plt.show()

