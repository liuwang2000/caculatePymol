# Linux 安装和使用指南

本文档指导您如何在Linux系统上安装和使用特征选择优化工具。

## 系统要求

- Linux操作系统(Ubuntu, CentOS, Debian等)
- Conda包管理工具
- Git(可选，用于克隆代码)

## 安装步骤

### 1. 安装Conda

如果您尚未安装Conda，可以按照以下步骤安装Miniconda:

```bash
# 下载Miniconda安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 执行安装脚本
bash miniconda.sh

# 按照提示完成安装
# 安装完成后，关闭并重新打开终端，或者执行以下命令使环境变量生效
source ~/.bashrc
```

### 2. 获取代码

如果您有仓库的Git访问权限：

```bash
git clone <仓库URL>
cd <仓库目录>
```

或者，从压缩包解压：

```bash
unzip feature_optimizer.zip
cd feature_optimizer
```

### 3. 创建Conda环境

使用提供的`environment.yml`文件创建Conda环境：

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate feature_optimizer
```

### 4. 设置执行权限

为Shell脚本设置执行权限：

```bash
chmod +x run_optimization.sh
```

## 使用方法

### 运行交互式优化器

```bash
./run_optimization.sh
```

按照脚本提示选择优化模式并提供所需参数。

### 直接运行特定优化

如果您想直接运行特定优化而不使用交互式脚本，可以使用以下命令：

#### 基本特征选择优化

```bash
python feature_selection_optimizer.py --data <数据文件路径> --output ./models
```

#### 高级特征选择优化

```bash
python advanced_feature_optimizer.py --data <数据文件路径> --output ./models_advanced --interactions --polynomials
```

#### 同时运行两种优化方法

```bash
python run_feature_optimization.py --data <数据文件路径> --mode both --interactions --polynomials
```

## 常见问题

### 1. 依赖项安装失败

如果通过pip安装依赖项失败，请确保使用Conda环境：

```bash
conda env create -f environment.yml
conda activate feature_optimizer
```

### 2. 中文显示问题

如果图表中的中文显示为方块或乱码，可能需要安装中文字体：

```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# CentOS/RHEL
sudo yum install wqy-microhei-fonts wqy-zenhei-fonts
```

然后在Python代码中设置：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
```

### 3. 图形界面问题

如果在无图形界面的服务器上运行，可能会出现图表保存失败的问题。可以通过设置以下环境变量解决：

```bash
export MPLBACKEND=Agg
```

或者在Python代码开头添加：

```python
import matplotlib
matplotlib.use('Agg')
```

## 资源消耗

高级特征选择优化可能需要较大的计算资源，特别是在大数据集上启用特征交互和多项式特征时。建议在具有充足内存和CPU核心的机器上运行。 