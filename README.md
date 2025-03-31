# PDB蛋白质最适温度预测模型

本项目使用随机森林算法根据PDB文件的结构特征预测蛋白质的最适温度。该模型使用了从PDB数据分析中提取的多种特征，能够帮助研究人员快速评估蛋白质的热稳定性。

## 功能特点

- 使用随机森林算法训练预测模型
- 自动优化模型超参数
- 生成特征重要性分析图表
- 预测结果可视化
- 直接分析PDB文件的能力
- 完整的评估指标(RMSE, MAE, R²)

## 安装依赖

确保已安装所有必要的依赖包：

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

## 使用方法

### 训练模型

使用默认参数训练模型：

```bash
python train_rf_model.py
```

自定义训练参数：

```bash
python train_rf_model.py --data ./path/to/your/data.csv --output ./models_custom
```

### 预测新的PDB结构最适温度

```bash
python train_rf_model.py --predict ./path/to/your/pdb_file.pdb
```

使用自定义模型文件：

```bash
python train_rf_model.py --predict ./path/to/your/pdb_file.pdb --model ./path/to/your/model.joblib
```

## 输出文件

模型训练会生成以下文件：

- `rf_temperature_predictor.joblib` - 训练好的随机森林模型
- `feature_importance.png` - 特征重要性可视化图
- `predictions_vs_actual.png` - 预测值与真实值对比图

预测模式会生成：

- `prediction_result.csv` - 包含预测结果的CSV文件

## 项目结构

- `train_rf_model.py` - 主要脚本，包含训练和预测功能
- `analyze_pdb.py` - PDB文件分析脚本，用于提取特征
- `models/` - 存放训练好的模型
- `trainData/` - 存放训练数据

## 注意事项

- 该模型需要完整的PDB特征数据才能进行准确预测
- 模型的准确性取决于训练数据的质量和代表性
- 对于特殊的PDB结构，可能需要额外的特征工程

## 示例数据

项目默认使用`trainData/analyze_pdb_merged_20250331_164045.csv`文件作为训练数据。该文件包含了多种PDB特征和对应的最适温度数据。