#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import logging
import time
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl

# 时间格式化函数
def format_time(seconds):
    """将秒数格式化为中文时间表示"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)}分钟{int(seconds)}秒"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"

# 配置中文字体支持
try:
    # 尝试设置中文字体
    import sys
    import platform
    
    # 根据操作系统类型选择字体
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'STXihei', 'FangSong']
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Heiti TC', 'Arial Unicode MS']
    else:  # Linux
        # Linux 环境尝试多种中文字体
        linux_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 
                       'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Source Han Sans CN', 
                       'Source Han Sans TW', 'DejaVu Sans']
        for font in linux_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 测试中文显示
    import matplotlib.font_manager as fm
    fonts = [f.name for f in fm.fontManager.ttflist]
    has_chinese_font = False
    for font in plt.rcParams['font.sans-serif']:
        if font in fonts:
            has_chinese_font = True
            logging.info(f"成功找到中文字体: {font}")
            break
    
    if not has_chinese_font:
        logging.warning("未找到支持中文的字体，图表中文可能无法正确显示")
        # 使用通用设置
        plt.rcParams['font.sans-serif'] = ['sans-serif']
        # 尝试使用Agg后端，可能在某些情况下有帮助
        plt.switch_backend('Agg')
    else:
        logging.info("成功配置matplotlib中文字体支持")
except Exception as e:
    logging.warning(f"配置中文字体时出错: {str(e)}，图表中文可能无法正确显示")

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def run_basic_optimization(data_file, output_dir, min_features=3, max_features=15, cv=5, include_aa=False):
    """运行基本特征选择优化"""
    logging.info("开始运行基本特征选择优化...")
    logging.info("此过程可能需要几分钟时间，取决于特征数量和CPU核心数")
    
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
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_str = format_time(elapsed_time)
        logging.info(f"基本特征选择优化完成，耗时: {elapsed_str}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"基本特征选择优化失败: {str(e)}")
        return False

def run_advanced_optimization(data_file, output_dir, interactions=False, polynomials=False, 
                             min_features=5, max_features=20, cv=5, include_aa=False):
    """运行高级特征优化"""
    logging.info("开始运行高级特征选择优化...")
    logging.info("此过程可能需要几分钟到几十分钟时间，取决于特征数量和选择的高级功能")
    
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
        logging.info("已启用特征交互项，这将增加计算复杂度和运行时间")
    
    if polynomials:
        cmd.append("--polynomials")
        logging.info("已启用多项式特征，这将显著增加计算复杂度和运行时间")
    
    if include_aa:
        cmd.append("--include_aa")
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_str = format_time(elapsed_time)
        logging.info(f"高级特征选择优化完成，耗时: {elapsed_str}")
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
    logging.info("已配置图表中文支持，生成的图表可显示中文标题和标签")
    
    # 预估总运行时间
    if args.mode == 'both':
        logging.info("将依次运行基本和高级特征选择优化，总耗时可能在几分钟到几小时之间")
    elif args.mode == 'advanced' and (args.interactions or args.polynomials):
        logging.info("高级特征选择已启用特征工程选项，预计运行时间较长，请耐心等待")
    
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
    elapsed_str = format_time(elapsed_time)
    logging.info(f"所有优化任务完成，总耗时: {elapsed_str}")
    
    # 输出结果目录
    if args.mode in ['basic', 'both']:
        logging.info(f"基本优化结果保存在: {basic_dir}")
    if args.mode in ['advanced', 'both']:
        logging.info(f"高级优化结果保存在: {advanced_dir}")
    
    logging.info("模型已按测试集R²最高排序，测试集上表现最好的模型已被保存")

if __name__ == "__main__":
    main() 