# 蛋白质最适温度预测系统

该系统基于蛋白质结构（PDB文件）预测其最适温度，并重点优化了对高温蛋白（≥60°C）的预测精度。系统采用二阶段策略：先判断蛋白质是高温还是低温，再进行精确的温度预测。

## 功能特点

- 专注于高温蛋白（≥60°C）的精确预测
- 对低温蛋白采用简化处理，减少计算复杂度
- 自动提取热稳定性相关特征，如离子对密度、IVYWREL指数等
- 支持批量处理多个PDB文件
- 生成直观的预测结果可视化
- 提供完整的训练-预测流程

## 安装要求

- Python 3.7+
- PyMOL Python模块
- 科学计算库：numpy, pandas, scikit-learn, matplotlib, joblib
- Biopython

## 快速开始

1. 准备PDB文件：确保有需要分析的蛋白质PDB文件

2. 运行预测：

```bash
python predict_temp.py your_pdb_folder
```

或者对单个PDB文件进行预测：

```bash
python predict_temp.py path/to/your/protein.pdb
```

3. 查看结果：
   - 程序会输出表格形式的预测结果
   - 结果会保存为CSV文件：`prediction_results_timestamp.csv`
   - 可视化结果图表会保存为：`temperature_prediction_results_timestamp.png`

## 命令行选项

```
python predict_temp.py [input_path] [options]
```

参数说明：
- `input_path`: 输入PDB文件或包含PDB文件的目录路径
- `--model`: 指定模型文件路径（可选）
- `--output`: 指定输出目录路径（可选）
- `--no-extract`: 跳过特征提取步骤，直接使用现有分析文件

## 高级用法

### 完整训练流程

对于全新的蛋白质预测模型训练，提供了简化的完整流程：

```bash
python train_temp_pipeline.py your_pdb_folder --cazy your_cazy_data.csv
```

这个流程会自动完成：
1. 特征提取：分析PDB文件提取结构特征
2. 数据合并：将结构特征与温度数据组合
3. 模型训练：训练适合的温度预测模型
4. 模型保存：将模型保存到models目录

### 高温蛋白精度优化

针对高温蛋白的预测精度提升，提供了特殊的训练选项：

```bash
python train_rf_model.py --data your_training_data.csv --high_temp_specialist --synthetic_samples
```

参数说明：
- `--high_temp_specialist`: 使用高温专精模型
- `--synthetic_samples`: 生成合成样本以增强高温蛋白的训练样本
- `--temperature_threshold`: 定义高温区域的温度阈值（默认70°C）
- `--save_enhanced`: 保存增强型高温模型

### 单独特征提取

单独运行特征提取步骤：

```bash
python analyze_pdb.py your_pdb_folder --thermostability
```

## 文件结构

- `predict_temp.py`: 简化的预测脚本
- `train_rf_model.py`: 模型训练脚本
- `train_temp_pipeline.py`: 完整训练流程脚本
- `analyze_pdb.py`: PDB特征提取脚本
- `merge_data.py`: 数据合并脚本
- `validate_model.py`: 模型验证脚本
- `test_high_temp_prediction.py`: 高温预测测试脚本
- `models/temperature_predictor.joblib`: 预训练的温度预测模型
- `output/`: 特征提取结果目录
- `result/`: 预测和验证结果目录
  - `result/predictions/`: 预测结果子目录
  - `result/validation/`: 验证结果子目录
  - `result/tests/`: 测试结果子目录

## 高温蛋白预测精度

当前模型对高温蛋白的预测精度：
- 高温区域（≥60°C）RMSE: 约6.06°C （增强模型）
- 高温区域（≥60°C）精确率: 0.94
- 高温区域（≥60°C）召回率: 0.93

相比原始模型（高温区域RMSE约17.46°C），增强模型提高了高温蛋白的预测精度。

## 工作原理

1. 特征提取：使用PyMOL和Biopython从PDB结构中提取关键特征
2. 二分类：判断蛋白质是高温蛋白还是低温蛋白
3. 回归预测：根据分类结果使用专精模型预测具体温度值

### 重要特征

系统重点关注以下与热稳定性相关的特征：
- 离子对网络密度
- 核心残基比例
- 表面电荷分布
- IVYWREL指数
- 氢键网络密度
- 二级结构比例
- 芳香族相互作用
- 甘氨酸含量

## 注意事项

- 预测精度受PDB文件质量影响，请使用高质量的蛋白质结构
- 系统对高温蛋白的预测精度更高，低温蛋白预测仅供参考
- 首次运行时需要联网下载必要的依赖包