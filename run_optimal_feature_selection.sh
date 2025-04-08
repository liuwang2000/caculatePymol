#!/bin/bash

# 颜色设置
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 标题
echo -e "\e[1m使用train_rf_model进行最优特征选择\e[0m"
echo "==============================="
echo

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: Python 未安装。请先安装Python 3.6+${NC}"
    exit 1
fi

# 检查train_rf_model.py是否存在
if [ ! -f "train_rf_model.py" ]; then
    echo -e "${RED}错误: 找不到train_rf_model.py文件${NC}"
    exit 1
fi

# 检查train_optimal_features.py是否存在
if [ ! -f "train_optimal_features.py" ]; then
    echo -e "${RED}错误: 找不到train_optimal_features.py文件${NC}"
    exit 1
fi

# 检查必要的库
echo -e "${BLUE}检查必要的Python库...${NC}"
python -c "
import sys
missing = []
try:
    import pandas
except ImportError:
    missing.append('pandas')
try:
    import numpy
except ImportError:
    missing.append('numpy')
try:
    import sklearn
except ImportError:
    missing.append('scikit-learn')
try:
    import matplotlib
except ImportError:
    missing.append('matplotlib')
try:
    import joblib
except ImportError:
    missing.append('joblib')
try:
    import tqdm
except ImportError:
    missing.append('tqdm')

if missing:
    print('缺少库: ' + ', '.join(missing))
    sys.exit(1)
else:
    print('所有必要的库已安装')
"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}警告: 一些必要的库未安装。是否要安装它们? (y/n)${NC}"
    read install_libs
    if [[ $install_libs == "y" || $install_libs == "Y" ]]; then
        pip install pandas numpy scikit-learn matplotlib joblib tqdm
    else
        echo -e "${RED}无法继续，请安装缺少的库后再试。${NC}"
        exit 1
    fi
fi

# 检测CPU核心数
CPU_CORES=$(nproc)
THREADS=$((CPU_CORES - 1))
if [ $THREADS -lt 1 ]; then
    THREADS=1
fi
echo -e "${GREEN}检测到${CPU_CORES}个CPU核心，将使用${THREADS}个核心进行并行处理${NC}"

# 默认参数
DEFAULT_DATA_FILE="./trainData/analyze_pdb_merged_20250403_164250.csv"
DEFAULT_OUTPUT_DIR="./optimal_models"
DEFAULT_MIN_FEATURES=3
DEFAULT_MAX_FEATURES=15
DEFAULT_TOP_K=10
DEFAULT_CV=5

# 询问数据文件路径
echo -e "${BLUE}请输入训练数据文件路径 (默认: ${DEFAULT_DATA_FILE}):${NC}"
read data_file
if [ -z "$data_file" ]; then
    data_file=$DEFAULT_DATA_FILE
fi

# 检查文件是否存在
if [ ! -f "$data_file" ]; then
    echo -e "${RED}错误: 找不到文件 ${data_file}${NC}"
    exit 1
fi

# 询问输出目录
echo -e "${BLUE}请输入输出目录路径 (默认: ${DEFAULT_OUTPUT_DIR}):${NC}"
read output_dir
if [ -z "$output_dir" ]; then
    output_dir=$DEFAULT_OUTPUT_DIR
fi

# 询问是否包含氨基酸比例特征
echo -e "${BLUE}是否包含氨基酸比例特征? (y/n, 默认: n):${NC}"
read include_aa
include_aa_flag=""
if [[ $include_aa == "y" || $include_aa == "Y" ]]; then
    include_aa_flag="--include_aa"
    echo -e "${GREEN}将包含氨基酸比例特征进行分析${NC}"
else
    echo -e "${YELLOW}注意: 将排除氨基酸比例特征以减少特征空间${NC}"
fi

# 询问是否创建特征交互项
echo -e "${BLUE}是否创建特征交互项? (y/n, 默认: n):${NC}"
read create_interactions
interactions_flag=""
if [[ $create_interactions == "y" || $create_interactions == "Y" ]]; then
    interactions_flag="--interactions"
    echo -e "${GREEN}将创建特征交互项${NC}"
fi

# 询问是否应用非线性变换
echo -e "${BLUE}是否应用非线性变换? (y/n, 默认: n):${NC}"
read apply_nonlinear
nonlinear_flag=""
if [[ $apply_nonlinear == "y" || $apply_nonlinear == "Y" ]]; then
    nonlinear_flag="--nonlinear"
    echo -e "${GREEN}将应用非线性变换${NC}"
fi

# 询问最小特征数
echo -e "${BLUE}请输入最小特征数量 (默认: ${DEFAULT_MIN_FEATURES}):${NC}"
read min_features
if [ -z "$min_features" ]; then
    min_features=$DEFAULT_MIN_FEATURES
fi

# 询问最大特征数
echo -e "${BLUE}请输入最大特征数量 (默认: ${DEFAULT_MAX_FEATURES}):${NC}"
read max_features
if [ -z "$max_features" ]; then
    max_features=$DEFAULT_MAX_FEATURES
fi

# 询问返回的顶部结果数量
echo -e "${BLUE}请输入返回的顶部结果数量 (默认: ${DEFAULT_TOP_K}):${NC}"
read top_k
if [ -z "$top_k" ]; then
    top_k=$DEFAULT_TOP_K
fi

# 询问交叉验证折数
echo -e "${BLUE}请输入交叉验证折数 (默认: ${DEFAULT_CV}):${NC}"
read cv
if [ -z "$cv" ]; then
    cv=$DEFAULT_CV
fi

# 输出配置信息
echo -e "${GREEN}配置摘要:${NC}"
echo -e "${BLUE}数据文件:${NC} $data_file"
echo -e "${BLUE}输出目录:${NC} $output_dir"
echo -e "${BLUE}最小特征数:${NC} $min_features"
echo -e "${BLUE}最大特征数:${NC} $max_features"
echo -e "${BLUE}顶部结果数量:${NC} $top_k"
echo -e "${BLUE}交叉验证折数:${NC} $cv"
echo -e "${BLUE}包含氨基酸比例特征:${NC} ${include_aa_flag:+是}"
echo -e "${BLUE}创建特征交互项:${NC} ${interactions_flag:+是}"
echo -e "${BLUE}应用非线性变换:${NC} ${nonlinear_flag:+是}"
echo -e "${BLUE}多线程:${NC} 使用${THREADS}个线程"
echo -e "${YELLOW}注意: 将选择并保存测试集R²最高的模型${NC}"

# 确认执行
echo -e "${BLUE}是否继续执行? (y/n):${NC}"
read confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo -e "${RED}操作已取消${NC}"
    exit 0
fi

# 运行优化
echo -e "${GREEN}开始执行特征选择优化...${NC}"
echo -e "${YELLOW}这可能需要几分钟时间，请耐心等待...${NC}"

# 构建命令
cmd="python train_optimal_features.py --data \"$data_file\" --output \"$output_dir\" --min_features $min_features --max_features $max_features --top_k $top_k --cv $cv $include_aa_flag $interactions_flag $nonlinear_flag"

# 执行命令
eval $cmd

if [ $? -eq 0 ]; then
    echo -e "${GREEN}优化完成!${NC}"
    echo -e "${GREEN}优化结果保存在: $output_dir${NC}"
    echo -e "${GREEN}模型选择标准: 测试集R²最高的模型已被保存${NC}"
else
    echo -e "${RED}优化过程出错。请检查日志以获取详细信息。${NC}"
fi

echo
echo -e "\e[32m优化完成！\e[0m"
echo "请查看生成的结果文件和可视化图表。" 