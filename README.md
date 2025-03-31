# 🧬 蛋白质最适温度预测系统

![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-使用-orange.svg)
![RandomForest](https://img.shields.io/badge/算法-随机森林-green.svg)
![PDB](https://img.shields.io/badge/数据源-PDB结构-yellow.svg)

## 📋 项目概述

本项目开发了一套基于**随机森林算法**的预测系统，能够通过分析蛋白质PDB结构文件的多种特征，准确预测蛋白质的最适工作温度。对于生物技术研究、酶工程设计和热稳定性评估具有重要参考价值。

### ✨ 核心特点

- 🤖 **智能算法**: 采用随机森林回归算法，具有优秀的非线性拟合能力
- 🔍 **特征分析**: 自动从PDB文件提取关键结构特征
- 🛠️ **超参数优化**: 内置网格搜索自动寻找最佳模型参数
- 📊 **可视化报告**: 生成特征重要性和预测效果直观图表
- 📈 **性能评估**: 提供完整的评估指标(RMSE, MAE, R²)
- 🚀 **批量处理**: 支持批量分析多个PDB文件

## 🔧 环境配置

### 依赖安装

```bash
# 安装主要依赖包
pip install pandas numpy scikit-learn matplotlib joblib biopython pymol

# 或使用requirements文件安装(如果有)
# pip install -r requirements.txt
```

### 系统要求

- Python 3.6+
- PyMOL (用于PDB结构分析)
- 足够的内存处理大型PDB文件

## 🚀 使用指南

### 训练新模型

```bash
# 使用默认参数训练模型
python train_rf_model.py

# 自定义训练数据和输出目录
python train_rf_model.py --data ./trainData/your_data.csv --output ./custom_models
```

### 预测蛋白质最适温度

```bash
# 预测目录中所有PDB文件的最适温度
python train_rf_model.py --predict ./your_pdb_directory

# 使用自定义模型进行预测
python train_rf_model.py --predict ./your_pdb_directory --model ./path/to/your_model.joblib
```

## 📊 输出说明

### 训练输出

- `rf_temperature_predictor.joblib` - 训练好的模型文件
- `feature_importance.png` - 特征重要性排序可视化图
- `predictions_vs_actual.png` - 预测值与真实值对比散点图

### 预测输出

- `prediction_results.csv` - 包含每个PDB文件ID及其预测温度的结果表

## 📁 项目结构

```
protein-temperature-predictor/
├── train_rf_model.py     # 主程序：模型训练与预测
├── analyze_pdb.py        # PDB结构特征提取程序
├── merge_data.py         # 数据预处理与合并工具
├── models/               # 存放训练好的模型
├── output/               # 特征提取结果输出目录
├── trainData/            # 训练数据集存放目录
└── README.md             # 项目说明文档
```

## 📝 注意事项

- 模型预测精度依赖于训练数据的质量和代表性
- PDB文件应包含完整的三维结构信息
- 对于非常规结构的蛋白质，可能需要调整特征提取参数
- 默认使用`trainData/analyze_pdb_merged_20250331_164045.csv`作为训练数据

## 🔗 参考信息

- 默认训练数据包含多种特征，如二级结构比例、表面极性、二硫键数量等
- 归一化特征对预测精度有显著影响
- 模型超参数可通过`train_rf_model.py`中的参数网格进行调整

---

*开发者: BIO221*