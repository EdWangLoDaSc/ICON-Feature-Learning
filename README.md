# ICON-Feature-Learning

## 项目简介

Stage 1. Liuyang ICON-LM for PDE解耦，用最简单的形式进行PDE ICOL，基于https://arxiv.org/pdf/2301.05217，做一下机制解释，虽然可能有点撤，但是看看同样的思路能不能通过HookPoint+Fourier解决ODE解释
## 环境设置

`env.yaml` 文件用于配置基于 JAX 的 CPU 版本的环境。要创建和激活该环境，请按照以下步骤操作：

### 创建 Conda 环境

1. **创建环境**
   ```bash
   conda env create -f env.yaml
   ```

2. **激活环境**
   ```bash
   conda activate <env_name>
   ```
   请将 `<env_name>` 替换为 `env.yaml` 文件中定义的环境名称。

## 目录结构

```
project-root/
├── baseline/                # 基线实现和实验代码
│   └── mechanism_doc/       # 机制解释文档
├── env.yaml                # 环境配置文件
├── src/                    # 源代码文件夹
└── README.md               # 项目文档
```

## 使用说明

在本项目中，我们对 ICON 模型的结构进行了解耦和改进，以便更好地理解和训练 Transformer 架构。请确保在运行项目前已正确安装并激活相应的 Conda 环境。

## 训练和评估

详细的训练和评估脚本请参考 `baseline` 文件夹中的内容，其中包含了 ICON 模型的 Transformer 版本的实现、训练和评估步骤。





