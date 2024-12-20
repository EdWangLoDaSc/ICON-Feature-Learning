def print_model_architecture(config):
    """
    Print a beautiful visualization of the ICON-LM model architecture
    """
    print("\n" + "="*80)
    print(" "*30 + "ICON-LM Model Architecture")
    print("="*80 + "\n")

    # Model Overview
    print("│ Model Overview:")
    print("├─ Max Demo Number:", config["demo_max_num"])
    print("├─ Output Dimension:", config["out_dim"])
    print("│")

    # Transformer Details
    print("│ Transformer Configuration:")
    print("├─ Number of Layers:", config["transformer"]["n_layers"])
    print("├─ Number of Attention Heads:", config["transformer"]["n_heads"])
    print("├─ Head Dimension:", config["transformer"]["head_dim"])
    print("├─ Model Dimension:", config["transformer"]["model_dim"])
    print("├─ Widening Factor:", config["transformer"]["widening_factor"])
    print("│")

    # Layer Structure
    print("│ Layer Structure:")
    print("├─ Input")
    print("├─ Pre-projection Layer (Dense)")
    print("│   └─ Output Dim:", config["transformer"]["model_dim"])
    print("│")
    print("├─ Positional Embedding")
    print("│   └─ Embedding Dim:", config["transformer"]["model_dim"])
    print("│")
    print("├─ Transformer Blocks (×{})".format(config["transformer"]["n_layers"]))
    print("│   ├─ Self Attention")
    print("│   │   ├─ Heads:", config["transformer"]["n_heads"])
    print("│   │   └─ Head Dim:", config["transformer"]["head_dim"])
    print("│   │")
    print("│   ├─ Layer Norm")
    print("│   │")
    print("│   ├─ MLP Block")
    print("│   │   ├─ Hidden Dim:", config["transformer"]["model_dim"] * config["transformer"]["widening_factor"])
    print("│   │   └─ Output Dim:", config["transformer"]["model_dim"])
    print("│   │")
    print("│   └─ Layer Norm")
    print("│")
    print("└─ Post-projection Layer (Dense)")
    print("    └─ Output Dim:", config["out_dim"])
    print("\n" + "="*80 + "\n")


def visualize_layer_outputs(intermediates, layer_idx=None):
    """Visualize the outputs of different layers.
  
    Args:
        intermediates: List of dictionaries containing layer outputs
        layer_idx: Optional specific layer to visualize. If None, shows all layers.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
  
    def plot_attention_pattern(layer_data, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(layer_data.mean(axis=-1), cmap='viridis')
        plt.title(title)
        plt.show()
  
    layers_to_plot = [layer_idx] if layer_idx is not None else range(len(intermediates) // 3)
  
    for layer in layers_to_plot:
        base_idx = layer * 3
    
        # Plot pre-attention activations
        plot_attention_pattern(
            intermediates[base_idx]["pre_attn"],
            f"Layer {layer} - Pre-Attention Activations"
        )
    
        # Plot post-attention activations
        plot_attention_pattern(
            intermediates[base_idx + 1]["post_attn"],
            f"Layer {layer} - Post-Attention Activations"
        )
    
        # Plot post-MLP activations
        plot_attention_pattern(
            intermediates[base_idx + 2]["post_mlp"],
            f"Layer {layer} - Post-MLP Activations"
        )

def analyze_model_behavior(model, params, test_batch):
    """Analyze model behavior using logit lens.
    
    Args:
        model: ICON_LM model instance
        params: Model parameters
        test_batch: Input data batch
    """
    rng = jax.random.PRNGKey(0)
    outputs, intermediates = model.apply(
        params, 
        test_batch, 
        rngs={'dropout': rng}, 
        method="analyze_intermediates"
    )
  
    visualize_layer_outputs(intermediates)
  
    for i in range(len(intermediates) // 3):
        base_idx = i * 3
        print(f"\nLayer {i} Statistics:")
        print(f"Pre-Attention  - Mean: {intermediates[base_idx]['pre_attn'].mean():.4f}, "
              f"Std: {intermediates[base_idx]['pre_attn'].std():.4f}")
        print(f"Post-Attention - Mean: {intermediates[base_idx+1]['post_attn'].mean():.4f}, "
              f"Std: {intermediates[base_idx+1]['post_attn'].std():.4f}")
        print(f"Post-MLP      - Mean: {intermediates[base_idx+2]['post_mlp'].mean():.4f}, "
              f"Std: {intermediates[base_idx+2]['post_mlp'].std():.4f}")
        


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_attention_weights(attention_weights, layer_idx, head_idx=None):
    """
    可视化注意力权重
    
    Args:
        attention_weights: shape [n_heads, seq_len, seq_len] 的注意力权重
        layer_idx: 层索引
        head_idx: 注意力头索引，如果为None则显示所有头的平均值
    """
    if head_idx is not None:
        weights = attention_weights[head_idx]
        title = f"Layer {layer_idx}, Head {head_idx} Attention Pattern"
    else:
        weights = attention_weights.mean(axis=0)
        title = f"Layer {layer_idx}, Average Attention Pattern"

    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, cmap='viridis')
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()

def plot_attention_head_summary(attention_weights, layer_idx):
    """
    为每个注意力头生成注意力模式摘要
    
    Args:
        attention_weights: shape [n_heads, seq_len, seq_len] 的注意力权重
        layer_idx: 层索引
    """
    n_heads = attention_weights.shape[0]
    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()

    for head_idx in range(n_heads):
        ax = axes[head_idx]
        sns.heatmap(attention_weights[head_idx], cmap='viridis', ax=ax)
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

    # 隐藏多余的子图
    for idx in range(n_heads, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Layer {layer_idx} - All Attention Heads', y=1.02)
    plt.tight_layout()
    plt.show()

def analyze_attention_patterns(attention_weights, layer_idx):
    """
    分析注意力模式的特征
    
    Args:
        attention_weights: shape [n_heads, seq_len, seq_len] 的注意力权重
        layer_idx: 层索引
    """
    n_heads = attention_weights.shape[0]
    
    # 计算每个头的统计信息
    for head_idx in range(n_heads):
        head_weights = attention_weights[head_idx]
        
        # 计算注意力熵（衡量注意力分散程度）
        entropy = -np.sum(head_weights * np.log(head_weights + 1e-10), axis=-1).mean()
        
        # 计算对角线注意力（自注意力）强度
        diagonal_attention = np.diagonal(head_weights).mean()
        
        # 计算最大注意力值及其位置
        max_attention = head_weights.max()
        max_pos = np.unravel_index(head_weights.argmax(), head_weights.shape)
        
        print(f"\nLayer {layer_idx}, Head {head_idx} Analysis:")
        print(f"Attention Entropy: {entropy:.4f}")
        print(f"Average Self-Attention: {diagonal_attention:.4f}")
        print(f"Max Attention: {max_attention:.4f} at position {max_pos}")

def visualize_attention_distance(attention_weights, layer_idx):
    """
    可视化注意力距离分布
    
    Args:
        attention_weights: shape [n_heads, seq_len, seq_len] 的注意力权重
        layer_idx: 层索引
    """
    n_heads = attention_weights.shape[0]
    seq_len = attention_weights.shape[1]
    
    # 创建距离矩阵
    positions = np.arange(seq_len)
    distances = np.abs(positions[:, None] - positions[None, :])
    
    plt.figure(figsize=(12, 6))
    
    for head_idx in range(n_heads):
        head_weights = attention_weights[head_idx]
        
        # 计算每个距离的平均注意力权重
        distance_weights = [head_weights[distances == d].mean() for d in range(seq_len)]
        plt.plot(distance_weights, label=f'Head {head_idx}')
    
    plt.title(f'Layer {layer_idx} - Attention Weight vs Token Distance')
    plt.xlabel('Token Distance')
    plt.ylabel('Average Attention Weight')
    plt.legend()
    plt.show()

def analyze_model_attention(model, params, test_batch):
    """
    完整的注意力分析流程
    
    Args:
        model: ICON_LM 模型实例
        params: 模型参数
        test_batch: 输入数据批次
    """
    rng = jax.random.PRNGKey(0)
    _, attention_data = model.apply(
        params, 
        test_batch, 
        rngs={'dropout': rng}, 
        method="get_attention_weights"
    )
    
    for layer_idx, layer_attention in enumerate(attention_data):
        print(f"\n=== Analyzing Layer {layer_idx} ===")
        
        # 1. 显示所有注意力头的热力图
        plot_attention_head_summary(layer_attention, layer_idx)
        
        # 2. 显示平均注意力模式
        plot_attention_weights(layer_attention, layer_idx)
        
        # 3. 分析注意力模式
        analyze_attention_patterns(layer_attention, layer_idx)
        
        # 4. 可视化注意力距离分布
        visualize_attention_distance(layer_attention, layer_idx)

def get_head_importance(attention_weights):
    """
    计算每个注意力头的重要性分数
    
    Args:
        attention_weights: shape [n_heads, seq_len, seq_len] 的注意力权重
    Returns:
        头重要性分数
    """
    # 使用注意力熵作为重要性的度量
    entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10), axis=(-2, -1))
    return entropy