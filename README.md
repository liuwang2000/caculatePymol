# PDB结构分析工具

## 项目用途
本脚本用于分析蛋白质PDB文件的结构特征，包括催化残基间距、二硫键数量、表面极性比例等指标。

## 依赖安装
```bash
pip install pymol-open-source
```

## 使用参数
```bash
python analyze_pdb.py <PDB文件路径> [催化残基1 催化残基2]
# 示例（计算催化残基间距）
python analyze_pdb.py enzyme.pdb ARG15 HIS57
```

## 输出说明
生成的CSV报告包含以下字段：
- Catalytic Distance: 催化残基间距（Å）
- Disulfide Bonds: 二硫键数量
- Surface Polar Ratio: 表面极性残基百分比
- HBond Network Strength: 氢键网络强度

## 常见问题
1. 依赖安装失败：
   - 确认使用Python 3.6+版本
   - 尝试`pip install --pre pymol-open-source`

2. 催化残基格式要求：
   - 必须为【氨基酸名称+编号】格式（如HIS57）
   - 区分大小写（ARG/arg视为不同）

3. 文件路径问题：
   - 使用绝对路径或正确相对路径
   - 避免路径包含中文或特殊字符