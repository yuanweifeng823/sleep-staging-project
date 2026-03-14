# Sleep Staging Project: Multi-modal Automatic Sleep Staging

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目概述

这是一个多模态自动睡眠分期项目，使用 Sleep-EDFx 数据集开发高精度睡眠分期模型。项目实现了五种不同的方法进行比较和评估。

### 🎯 项目目标
- 开发准确的自动睡眠分期系统
- 比较不同深度学习架构在睡眠分期任务上的性能
- 探索多模态融合方法
- 提供可复现的基准结果

### 👥 团队成员与方法

| 成员 | 方法 | 描述 |
|------|------|------|
| **A** | 1D-CNN | 原始EEG波形的1D卷积神经网络 |
| **B** | 2D-CNN | STFT频谱图的2D卷积神经网络 |
| **C** | Transformer | 时序建模的Transformer架构 |
| **D** | 多模态融合 | EEG + EOG的多模态融合 |
| **E** | 特征工程基线 | 基于PSD特征的传统机器学习方法 |

## 🚀 快速开始

### 1. 环境准备

#### 安装依赖
```bash
# 克隆项目
git clone <repository-url> 暂时还没放到github上 忽略
cd sleep-staging-project

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 系统要求
- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速) （自行检查适合自己电脑的CUDA版本）

### 2. 数据准备

#### 下载数据集
数据集下载极慢，可以先去官网看下下载速度如何
```bash
python scripts/download_data.py
```
这将下载 Sleep-EDFx 数据集到 `data/raw/` 目录。

#### 数据预处理
```bash
python scripts/preprocess_all.py
```
这将处理所有受试者数据并保存到 `data/processed/` 目录。

### 3. 配置管理

#### 创建默认配置
```bash
python scripts/manage_config.py create
```
这将为所有团队成员创建默认的训练配置文件。

#### 查看配置
```bash
# 查看特定成员的配置
python scripts/manage_config.py show --config experiments/memberA_1dcnn/train_config.yaml
```

#### 自定义配置
```bash
# 创建自定义配置
python scripts/manage_config.py custom --member A --output my_config.yaml
```

### 4. 模型训练

#### 训练单个模型
```bash
# 进入模型目录
cd src/models/memberA_1dcnn

# 开始训练
python train.py
```

#### 训练参数说明
- `--config`: 指定配置文件路径 (默认使用 `experiments/memberA_1dcnn/train_config.yaml`)
- `--resume`: 从检查点恢复训练

### 5. 模型评估

#### 评估单个模型
```bash
python scripts/evaluate_all.py --member A
```

#### 评估所有模型
```bash
python scripts/evaluate_all.py
```

## 📁 项目结构

```
📦 sleep-staging-project
├── 📄 README.md                    # 项目说明文档
├── 📄 requirements.txt             # Python依赖
├── 📄 .gitignore                   # Git忽略文件
│
├── 📁 data/                        # 数据目录
│   ├── 📁 raw/                     # 原始EDF文件
│   ├── 📁 processed/               # 预处理后的数据
│   └── 📁 external/                # 外部资源
│
├── 📁 src/                         # 源代码
│   ├── 📄 __init__.py
│   │
│   ├── 📁 data/                    # 数据处理模块
│   │   ├── 📄 loader.py            # 数据加载
│   │   ├── 📄 preprocess.py        # 数据预处理
│   │   ├── 📄 dataset.py           # PyTorch数据集
│   │   └── 📄 split.py             # 数据分割
│   │
│   ├── 📁 features/                # 特征提取模块
│   │   ├── 📄 stft.py              # STFT变换
│   │   └── 📄 psd.py               # PSD特征
│   │
│   ├── 📁 models/                  # 模型实现
│   │   ├── 📄 base.py              # 基类
│   │   │
│   │   ├── 📁 memberA_1dcnn/       # 成员A: 1D-CNN
│   │   │   ├── 📄 model.py          # 网络结构定义
│   │   │   ├── 📄 train.py          # 训练脚本（示例）
│   │   │   └── 📄 train_config.yaml # 默认配置
│   │   │
│   │   ├── 📁 memberB_2dcnn/       # 成员B: 2D-CNN
│   │   │   ├── 📄 model.py
│   │   │   ├── 📄 train.py
│   │   │   └── 📄 train_config.yaml
│   │   │
│   │   ├── 📁 memberC_transformer/ # 成员C: Transformer
│   │   │   ├── 📄 model.py
│   │   │   ├── 📄 train.py
│   │   │   └── 📄 train_config.yaml
│   │   │
│   │   ├── 📁 memberD_multimodal/  # 成员D: 多模态
│   │   │   ├── 📄 model.py
│   │   │   ├── 📄 train.py
│   │   │   └── 📄 train_config.yaml
│   │   │
│   │   └── 📁 memberE_baseline/    # 成员E: 基线
│   │       ├── 📄 model.py
│   │       ├── 📄 train.py
│   │       └── 📄 train_config.yaml
│   │
│   ├── 📁 training/                # 训练模块
│   │   ├── 📄 trainer.py           # 训练器基类
│   │   ├── 📄 metrics.py           # 评估指标
│   │   └── 📄 losses.py            # 损失函数
│   │
│   ├── 📁 utils/                   # 工具模块
│   │   ├── 📄 config.py            # 配置管理
│   │   ├── 📄 logger.py            # 日志工具
│   │   ├── 📄 paths.py             # 路径管理
│   │   └── 📄 helpers.py           # 辅助函数
│   │
│   └── 📁 validation/              # 验证模块
│       └── 📄 validator.py         # 验证器
│
├── 📁 experiments/                 # 实验配置
│   ├── 📁 memberA_1dcnn/           # 成员A配置文件夹
│   │   └── 📄 train_config.yaml     # 默认训练配置
│   ├── 📁 memberB_2dcnn/           # 成员B配置文件夹
│   │   └── 📄 train_config.yaml
│   ├── 📁 memberC_transformer/     # 成员C配置文件夹
│   │   └── 📄 train_config.yaml
│   ├── 📁 memberD_multimodal/      # 成员D配置文件夹
│   │   └── 📄 train_config.yaml
│   └── 📁 memberE_baseline/        # 成员E配置文件夹
│       └── 📄 train_config.yaml
│
├── 📁 results/                     # 实验结果
│   ├── 📁 memberA_1dcnn/           # 成员A输出目录
│   │   ├── 📄 final_model.pth
│   │   ├── 📄 predictions.npy
│   │   ├── 📄 confusion_matrix.png
│   │   └── 📄 results.json
│   ├── 📁 memberB_2dcnn/
│   ├── 📁 memberC_transformer/
│   ├── 📁 memberD_multimodal/
│   ├── 📁 memberE_baseline/
│   └── 📄 comparison.json          # 模型比较结果
│
└── 📁 scripts/                     # 脚本工具
    ├── 📄 download_data.py         # 数据下载
    ├── 📄 preprocess_all.py        # 批量预处理
    ├── 📄 evaluate_all.py          # 批量评估
    └── 📄 manage_config.py         # 配置管理
```

## ⚙️ 配置系统

### 配置结构
每个实验配置包含三个主要部分：

#### 模型配置 (model)
```yaml
model:
  n_classes: 5          # 睡眠阶段数量
  input_channels: 1     # 输入通道数
  hidden_dims: [64, 128, 256]  # 隐藏层维度
  dropout: 0.5          # Dropout率
  activation: "relu"    # 激活函数
```

#### 训练配置 (training)
```yaml
training:
  epochs: 50            # 训练轮数
  batch_size: 32        # 批次大小
  learning_rate: 0.001  # 学习率
  weight_decay: 0.0001  # 权重衰减
  scheduler: "cosine"   # 学习率调度器
  grad_clip: 1.0        # 梯度裁剪
  patience: 10          # 早停耐心值
```

#### 数据配置 (data)
```yaml
data:
  sampling_rate: 100    # 采样率
  epoch_length: 30      # 历元长度(秒)
  overlap: 0.0          # 重叠比例
  test_fold: 0          # 测试折
  n_folds: 5            # 总折数
  modalities: ["eeg"]   # 数据模态
```

## 📊 结果说明

### 评估指标
- **准确率 (Accuracy)**: 整体分类准确率
- **宏平均F1 (Macro F1)**: 各类别F1分数的平均值
- **各阶段F1**: W、N1、N2、N3、REM五个睡眠阶段的F1分数

### 输出文件
- `results/memberX_model/predictions.npy`: 模型预测结果
- `results/memberX_model/confusion_matrix.png`: 混淆矩阵可视化
- `results/memberX_model/results.json`: 详细评估结果
- `results/comparison.json`: 所有模型比较结果

## 🔧 故障排除

### 常见问题

#### 1. 依赖安装失败
```bash
# 更新pip
pip install --upgrade pip

# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. 数据下载失败
- 检查网络连接
- 确认有足够的磁盘空间 (>8GB)
- 手动下载数据并放置到 `data/raw/` 目录

#### 3. 内存不足
- 减小 `batch_size` 在配置文件中
- 使用数据子集进行测试: `--subset 0.1`

#### 4. CUDA相关错误
- 检查CUDA版本兼容性
- 设置 `device: "cpu"` 在配置文件中强制使用CPU（尽量不要这么干）

### 日志调试
```bash
# 查看详细日志
export LOG_LEVEL=DEBUG
python scripts/your_script.py
```