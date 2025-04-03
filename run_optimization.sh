#!/bin/bash

# 特征选择优化工具启动器
echo -e "\e[1m特征选择优化工具启动器\e[0m"
echo "===================="
echo

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo -e "\e[31m[错误] 未检测到Python安装。请安装Python 3.6+后再运行此脚本。\e[0m"
    exit 1
fi

# 检查必要的库是否已安装
echo "检查依赖项..."
if ! python -c "import pandas, numpy, sklearn, matplotlib, seaborn, joblib, tqdm" &> /dev/null; then
    echo -e "\e[33m[警告] 缺少部分依赖库。\e[0m"
    echo "推荐使用conda环境："
    echo "conda env create -f environment.yml"
    echo "conda activate feature_optimizer"
    
    read -p "是否继续尝试使用pip安装依赖? (y/n, 默认: n): " install_deps
    if [ "$install_deps" = "y" ] || [ "$install_deps" = "Y" ]; then
        echo "正在安装依赖项..."
        pip install scikit-learn pandas numpy matplotlib seaborn joblib tqdm
        if [ $? -ne 0 ]; then
            echo -e "\e[31m[错误] 安装依赖项失败，请使用conda环境或手动安装。\e[0m"
            exit 1
        fi
        echo "依赖项安装完成！"
    else
        echo "请设置好依赖后再运行此脚本。"
        exit 1
    fi
else
    echo -e "\e[32m所有依赖项已安装。\e[0m"
fi

# 检测CPU核心数并显示
cpu_cores=$(grep -c ^processor /proc/cpuinfo)
used_cores=$((cpu_cores - 1))
echo -e "\e[32m检测到 $cpu_cores 个CPU核心，将使用 $used_cores 个核心进行并行处理\e[0m"

echo
echo "请选择要运行的优化方式:"
echo "1. 基本特征选择优化"
echo "2. 高级特征选择优化"
echo "3. 同时运行两种优化方法并对比"
echo

read -p "请输入选项编号 (1-3): " choice

echo
read -p "请输入训练数据CSV文件路径 (默认: trainData/merged_data_20250401_213334.csv): " data_file
if [ -z "$data_file" ]; then
    data_file="trainData/merged_data_20250401_213334.csv"
fi

echo
read -p "是否排除氨基酸比例特征? (y/n, 默认: y): " exclude_aa
if [ -z "$exclude_aa" ]; then
    exclude_aa="y"
fi

echo
read -p "最小特征数量 (默认: 5): " min_features
if [ -z "$min_features" ]; then
    min_features=5
fi

echo
read -p "最大特征数量 (默认: 15): " max_features
if [ -z "$max_features" ]; then
    max_features=15
fi

exclude_aa_flag=""
if [ "$exclude_aa" = "n" ] || [ "$exclude_aa" = "N" ]; then
    exclude_aa_flag="--include_aa"
fi

if [ "$choice" = "1" ]; then
    echo
    echo "正在运行基本特征选择优化..."
    python feature_selection_optimizer.py --data "$data_file" --output ./models --min_features $min_features --max_features $max_features $exclude_aa_flag
elif [ "$choice" = "2" ]; then
    echo
    read -p "是否创建特征交互项? (y/n, 默认: n): " interactions
    read -p "是否创建多项式特征? (y/n, 默认: n): " polynomials
    
    cmd="python advanced_feature_optimizer.py --data \"$data_file\" --output ./models_advanced --min_features $min_features --max_features $max_features $exclude_aa_flag"
    if [ "$interactions" = "y" ] || [ "$interactions" = "Y" ]; then
        cmd="$cmd --interactions"
    fi
    if [ "$polynomials" = "y" ] || [ "$polynomials" = "Y" ]; then
        cmd="$cmd --polynomials"
    fi
    
    echo "正在运行高级特征选择优化..."
    eval $cmd
elif [ "$choice" = "3" ]; then
    echo
    read -p "是否创建特征交互项? (y/n, 默认: n): " interactions
    read -p "是否创建多项式特征? (y/n, 默认: n): " polynomials
    
    cmd="python run_feature_optimization.py --data \"$data_file\" --mode both --min_features $min_features --max_features $max_features $exclude_aa_flag"
    if [ "$interactions" = "y" ] || [ "$interactions" = "Y" ]; then
        cmd="$cmd --interactions"
    fi
    if [ "$polynomials" = "y" ] || [ "$polynomials" = "Y" ]; then
        cmd="$cmd --polynomials"
    fi
    
    echo "正在运行两种优化方法..."
    eval $cmd
else
    echo -e "\e[31m[错误] 无效的选项: $choice\e[0m"
    exit 1
fi

echo
echo -e "\e[32m优化完成！结果已保存到相应目录。\e[0m"
echo "请查看生成的结果文件和可视化图表。" 