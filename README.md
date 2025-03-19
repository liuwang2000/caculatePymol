# PDB结构分析工具

## 项目用途
本脚本用于分析蛋白质PDB文件的结构特征，包括催化残基间距、二硫键数量、表面极性比例等指标。

## 依赖安装
```bash
pip install pymol-open-source
```

## 使用参数
```bash
python analyze_pdb.py <输入目录路径>
# 示例（批量处理PDB目录）
python analyze_pdb.py ./pdb_files/
```

## 输出说明
报告文件按时间戳自动命名（如analyze_pdb_YYYYMMDD_HHMMSS.csv），包含以下字段：
- Disulfide Bonds: 二硫键数量
- Surface Polar Ratio: 表面极性残基百分比（基于溶剂可及性表面计算）
- Hydrogen Bonds: 跨残基氢键数量
- Hydrophobic Contacts: 疏水核心接触点
- Salt Bridges: 盐桥数量
- Helix/Sheet/Loop: 二级结构比例

## 常见问题
1. 依赖安装失败：
   - 确认使用Python 3.6+版本
   - 尝试`pip install --pre pymol-open-source`

2. 文件处理要求：
   - 输入目录需包含至少1个PDB文件
   - 自动跳过非PDB格式文件
   - 输出目录自动创建（output/）

3. 文件路径问题：
   - 使用绝对路径或正确相对路径
   - 避免路径包含中文或特殊字符