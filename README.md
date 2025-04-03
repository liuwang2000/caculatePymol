# 🧬 PymolThermoPredict

<div align="center">

![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![RandomForest](https://img.shields.io/badge/算法-随机森林/堆叠模型-green.svg)
![PDB](https://img.shields.io/badge/数据源-PDB结构-yellow.svg)
![License](https://img.shields.io/badge/许可证-MIT-brightgreen.svg)

</div>

## 💡 项目简介

本系统利用**机器学习算法**分析蛋白质PDB结构文件的物理化学特征，准确预测蛋白质的**最适工作温度**。该技术对于生物技术研究、工业酶设计以及生物催化剂开发具有重要指导意义。

通过综合分析二级结构比例、氨基酸组成、表面特性等多维特征，结合先进的模型堆叠技术，本系统能够精准捕捉蛋白质结构与热稳定性的内在联系。

## ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🤖 **多模型集成** | 集成随机森林、梯度提升、神经网络等多种算法，通过模型堆叠提升预测准确性 |
| 🔬 **全面特征分析** | 自动从PDB文件提取30+种结构特征，包括疏水核心、二硫键、盐桥等关键稳定性因素 |
| 📊 **数据可视化** | 生成特征重要性、预测效果、残差分析等多种直观图表 |
| 🧪 **交叉验证** | 采用K折交叉验证评估模型泛化能力，提供可靠性指标 |
| 🛠️ **高级特征工程** | 支持非线性变换、特征交互、特征选择等高级特征工程技术 |
| 📈 **性能优化** | 自动网格搜索最佳超参数，持续提升预测精度 |

## 🚀 快速开始

### 安装依赖

```bash
# 安装核心依赖包
pip install pandas numpy scikit-learn matplotlib joblib biopython pymol

# 或使用requirements文件一键安装
pip install -r requirements.txt
```

### 基本使用流程

1. **特征提取**：分析PDB文件并提取特征
   ```bash
   python analyze_pdb.py ./your_pdb_directory
   ```

2. **数据合并**：将特征数据与已知温度信息合并
   ```bash
   python merge_data.py
   ```

3. **训练模型**：使用预处理数据训练预测模型
   ```bash
   python train_rf_model.py
   ```

4. **预测温度**：对新的PDB文件进行温度预测
   ```bash
   python train_rf_model.py --predict ./new_pdb_directory
   ```

## 🔧 高级参数设置

训练模型时，可以通过以下参数优化模型性能：

```bash
# 启用所有高级特性进行训练
python train_rf_model.py --advanced_models --interactions --nonlinear --feature_selection

# 使用自定义数据源
python train_rf_model.py --data ./trainData/your_custom_data.csv --output ./custom_models
```

| 参数 | 说明 |
|------|------|
| `--advanced_models` | 启用神经网络和模型堆叠技术 |
| `--interactions` | 创建特征交互项以捕捉复杂关系 |
| `--nonlinear` | 应用非线性特征变换增强表达能力 |
| `--feature_selection` | 启用特征选择以减少过拟合 |
| `--data` | 指定训练数据文件路径 |
| `--output` | 设置模型输出目录 |

## 📊 结果解读

### 模型评估指标

- **RMSE (均方根误差)**：预测温度的平均偏差，越小越好
- **MAE (平均绝对误差)**：预测温度的平均绝对偏差
- **R²**：决定系数，反映模型解释数据变异的能力，越接近1越好
- **相对误差**：预测误差占实际温度的百分比

### 输出文件

- **temperature_predictor.joblib**：训练好的模型包
- **feature_importance.png**：特征重要性可视化
- **predictions_vs_actual.png**：预测值与实际值对比图
- **residuals.png**：残差分析图
- **model_comparison.png**：不同模型性能对比
- **prediction_results.csv**：预测结果汇总表

## 📁 项目结构

```
PymolThermoPredict/
├── train_rf_model.py     # 主程序：模型训练与预测
├── analyze_pdb.py        # PDB结构特征提取程序
├── merge_data.py         # 数据预处理与合并工具
├── models/               # 存放训练好的模型和可视化结果
├── output/               # 特征提取结果输出目录
├── trainData/            # 训练数据集存放目录
└── README.md             # 项目说明文档
```

## 🔍 应用场景

- **工业酶开发**：预测新设计酶的最佳工作温度
- **蛋白质工程**：辅助热稳定性改造设计
- **生物催化剂筛选**：快速评估候选蛋白的适用温度范围
- **结构生物学研究**：探索结构特征与热稳定性关系

## ⚠️ 注意事项

- 确保PDB文件结构完整，无严重缺失
- 模型预测精度受训练数据质量和覆盖范围影响
- 建议使用3Å以下分辨率的结构获取更可靠结果
- 对于极端温度环境的蛋白（<0°C或>100°C），可能需要特殊调整

## 🔗 参考信息

- 默认训练数据包含二级结构比例、表面极性、疏水核心等关键特征
- 模型已针对常见温度范围（5°C-80°C）的蛋白质进行优化
- 预测结果可作为实验设计的重要参考，但建议结合实验验证

---

*开发者: BIO221 | 版本: 1.2.0 | 最后更新: 2025-04-15*

# 特征选择优化工具

这是一个用于提高模型R²值的特征选择优化工具集。通过系统地尝试不同的特征组合和特征选择方法，找出最佳特征集来提高模型性能。

## 功能特点

- **基本特征选择优化**：通过穷举不同组合找出最佳特征子集
- **高级特征选择优化**：使用多种特征选择算法并结合特征工程
- **特征交互**：自动创建有意义的特征交互项
- **多项式特征**：生成多项式特征以捕捉非线性关系
- **模型比较**：自动比较随机森林和梯度提升模型
- **可视化**：生成特征重要性、预测结果和特征相关性矩阵的可视化图表

## 依赖项

- Python 3.6+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- tqdm

安装依赖项：

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib tqdm
```

## 文件结构

- `feature_selection_optimizer.py`：基本特征选择优化脚本
- `advanced_feature_optimizer.py`：高级特征选择和特征工程脚本
- `run_feature_optimization.py`：运行两种优化方式的便捷脚本
- `README.md`：使用说明文档

## 使用方法

### 基本特征选择优化

```bash
python feature_selection_optimizer.py --data 训练数据.csv --output ./models --min_features 3 --max_features 15 --cv 5
```

参数说明：
- `--data`：训练数据文件路径（CSV格式）
- `--output`：输出目录，用于保存模型和结果
- `--min_features`：最小特征数量
- `--max_features`：最大特征数量
- `--cv`：交叉验证折数
- `--top_k`：返回的顶部结果数量
- `--test_size`：测试集比例
- `--random_state`：随机种子

### 高级特征选择优化

```bash
python advanced_feature_optimizer.py --data 训练数据.csv --output ./models_advanced --interactions --polynomials --min_features 5 --max_features 20 --cv 5
```

参数说明：
- `--data`：训练数据文件路径（CSV格式）
- `--output`：输出目录，用于保存模型和结果
- `--interactions`：创建特征交互项
- `--polynomials`：创建多项式特征
- `--degree`：多项式特征次数
- `--min_features`：最小特征数量
- `--max_features`：最大特征数量
- `--cv`：交叉验证折数
- `--test_size`：测试集比例
- `--random_state`：随机种子

### 一键运行两种优化方法

```bash
python run_feature_optimization.py --data 训练数据.csv --mode both --interactions --polynomials --min_features 5 --max_features 15 --cv 5
```

参数说明：
- `--data`：训练数据文件路径（CSV格式）
- `--mode`：运行模式（basic/advanced/both）
- `--interactions`：在高级模式中创建特征交互项
- `--polynomials`：在高级模式中创建多项式特征
- `--min_features`：最小特征数量
- `--max_features`：最大特征数量
- `--cv`：交叉验证折数

## 数据格式要求

输入CSV文件应包含：
1. 特征列
2. 目标变量列 `optimal_temperature`

脚本会自动排除以下列：
- `pdb_id`
- `sequence`
- `optimal_temperature`（作为目标变量）
- `hydrogen_bonds`
- `hydrophobic_contacts`
- `salt_bridges`
- `hydrophobic_sasa`

## 输出结果

脚本运行后会生成以下输出：
1. 训练好的最佳模型（.pkl文件）
2. 特征选择结果摘要（.csv文件）
3. 预测结果可视化
4. 特征重要性图
5. 特征相关性矩阵（高级模式）

## 示例

```bash
# 使用默认参数运行基本特征选择
python feature_selection_optimizer.py --data trainData/merged_data_20250401_213334.csv --output ./models

# 运行高级特征选择，包含特征交互
python advanced_feature_optimizer.py --data trainData/merged_data_20250401_213334.csv --output ./models_advanced --interactions

# 一键运行两种优化并对比
python run_feature_optimization.py --data trainData/merged_data_20250401_213334.csv --mode both --interactions
```

## 注意事项

1. 对于大数据集，特征选择过程可能需要较长时间，请耐心等待
2. 使用`--interactions`和`--polynomials`参数会大幅增加特征数量，可能导致计算时间更长
3. 最终结果中包含重要的性能指标和可视化，可帮助您理解模型和特征的重要性