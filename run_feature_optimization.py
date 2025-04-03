#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import logging
import time
import multiprocessing

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def run_basic_optimization(data_file, output_dir, min_features=3, max_features=15, cv=5, include_aa=False):
    """运行基本特征选择优化"""
    logging.info("开始运行基本特征选择优化...")
    
    if include_aa:
        logging.info("将包含氨基酸比例特征进行分析")
    else:
        logging.info("注意：将自动排除氨基酸比例特征（aa_前缀的特征）以减少特征空间")
    
    cmd = [
        "python", "feature_selection_optimizer.py",
        "--data", data_file,
        "--output", output_dir,
        "--min_features", str(min_features),
        "--max_features", str(max_features),
        "--cv", str(cv)
    ]
    
    if include_aa:
        cmd.append("--include_aa")
    
    try:
        subprocess.run(cmd, check=True)
        logging.info("基本特征选择优化完成")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"基本特征选择优化失败: {str(e)}")
        return False

def run_advanced_optimization(data_file, output_dir, interactions=False, polynomials=False, 
                             min_features=5, max_features=20, cv=5, include_aa=False):
    """运行高级特征优化"""
    logging.info("开始运行高级特征选择优化...")
    
    if include_aa:
        logging.info("将包含氨基酸比例特征进行分析")
    else:
        logging.info("注意：将自动排除氨基酸比例特征（aa_前缀的特征）以减少特征空间")
    
    cmd = [
        "python", "advanced_feature_optimizer.py",
        "--data", data_file,
        "--output", output_dir,
        "--min_features", str(min_features),
        "--max_features", str(max_features),
        "--cv", str(cv)
    ]
    
    if interactions:
        cmd.append("--interactions")
    
    if polynomials:
        cmd.append("--polynomials")
    
    if include_aa:
        cmd.append("--include_aa")
    
    try:
        subprocess.run(cmd, check=True)
        logging.info("高级特征选择优化完成")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"高级特征选择优化失败: {str(e)}")
        return False

def create_output_directories():
    """创建输出目录"""
    basic_dir = "./models_basic"
    advanced_dir = "./models_advanced"
    
    os.makedirs(basic_dir, exist_ok=True)
    os.makedirs(advanced_dir, exist_ok=True)
    
    return basic_dir, advanced_dir

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行特征选择优化工具')
    parser.add_argument('--data', type=str, required=True, help='训练数据文件路径 (CSV格式)')
    parser.add_argument('--mode', type=str, choices=['basic', 'advanced', 'both'], default='both',
                       help='运行模式: 基本优化、高级优化或两者皆是')
    parser.add_argument('--interactions', action='store_true', help='在高级模式中创建特征交互项')
    parser.add_argument('--polynomials', action='store_true', help='在高级模式中创建多项式特征')
    parser.add_argument('--min_features', type=int, default=5, help='最小特征数量')
    parser.add_argument('--max_features', type=int, default=15, help='最大特征数量')
    parser.add_argument('--cv', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--include_aa', action='store_true', help='包含氨基酸比例特征')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    basic_dir, advanced_dir = create_output_directories()
    
    # 记录开始时间
    start_time = time.time()
    
    # 获取CPU核心数并显示
    cpu_count = multiprocessing.cpu_count()
    logging.info(f"检测到{cpu_count}个CPU核心，将使用{cpu_count-1}个核心进行并行处理")
    logging.info("已配置多线程支持以加速特征选择过程")
    
    # 根据模式运行相应的优化
    if args.mode in ['basic', 'both']:
        run_basic_optimization(
            args.data, basic_dir,
            min_features=args.min_features,
            max_features=args.max_features,
            cv=args.cv,
            include_aa=args.include_aa
        )
    
    if args.mode in ['advanced', 'both']:
        run_advanced_optimization(
            args.data, advanced_dir,
            interactions=args.interactions,
            polynomials=args.polynomials,
            min_features=args.min_features,
            max_features=args.max_features,
            cv=args.cv,
            include_aa=args.include_aa
        )
    
    # 计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"所有优化任务完成，总耗时: {elapsed_time:.2f}秒")
    
    # 输出结果目录
    if args.mode in ['basic', 'both']:
        logging.info(f"基本优化结果保存在: {basic_dir}")
    if args.mode in ['advanced', 'both']:
        logging.info(f"高级优化结果保存在: {advanced_dir}")

if __name__ == "__main__":
    main() 