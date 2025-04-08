#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse
import os
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='根据特征评估结果文件中的排名提取数据')
    parser.add_argument('--data', type=str, default='./trainData/analyze_pdb_merged_20250403_164250.csv',
                        help='原始数据文件路径')
    parser.add_argument('--results', type=str, 
                        default='./models/rf_feature_models/feature_evaluation_results_20250403_204300.csv',
                        help='特征评估结果文件路径')
    parser.add_argument('--output', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--rank', type=int, default=1,
                        help='要提取的特征组合排名 (1表示最佳特征组合)')
    parser.add_argument('--top_n', type=int, default=0,
                        help='提取前N个特征组合 (0表示只提取指定rank的组合)')
    parser.add_argument('--include_sequence', action='store_true',
                        help='是否在输出中包含蛋白质序列')
    parser.add_argument('--format', type=str, default='combined', choices=['combined', 'separate'],
                        help='输出格式: combined=合并到一个文件, separate=每个组合一个文件')
    parser.add_argument('--include_metadata', action='store_true',
                        help='是否在输出中包含特征组合的元数据（排名、R²值等）')
    return parser.parse_args()

def load_data(data_file):
    """加载原始数据文件"""
    try:
        data = pd.read_csv(data_file)
        logging.info(f"成功加载原始数据，共{len(data)}条记录")
        return data
    except Exception as e:
        logging.error(f"加载原始数据失败: {str(e)}")
        return None

def load_feature_results(results_file):
    """加载特征评估结果文件"""
    try:
        results = pd.read_csv(results_file)
        logging.info(f"成功加载特征评估结果，共{len(results)}个特征组合")
        return results
    except Exception as e:
        logging.error(f"加载特征评估结果失败: {str(e)}")
        return None

def extract_features_by_rank(data, feature_results, rank=1, top_n=0, include_sequence=False, format='combined', include_metadata=False):
    """根据排名提取特征数据"""
    if top_n > 0:
        # 提取前N个特征组合
        selected_results = feature_results.head(top_n)
        logging.info(f"将提取排名前{top_n}的特征组合")
    else:
        # 只提取指定排名的特征组合
        if rank > len(feature_results):
            logging.error(f"指定的排名{rank}超出特征组合总数{len(feature_results)}")
            return None
        
        selected_results = feature_results[feature_results['rank'] == rank]
        if len(selected_results) == 0:
            logging.error(f"未找到排名为{rank}的特征组合")
            return None
        
        logging.info(f"将提取排名第{rank}的特征组合")
    
    all_extracted_data = []
    
    for _, result_row in selected_results.iterrows():
        current_rank = result_row['rank']
        feature_str = result_row['features']
        features = [f.strip() for f in feature_str.split(',')]
        
        logging.info(f"处理排名第{current_rank}的特征组合: {', '.join(features)}")
        
        # 基本列：ID和最适温度
        columns_to_extract = ['pdb_id', 'optimal_temperature']
        
        # 添加特征列
        columns_to_extract.extend(features)
        
        # 如果需要，添加序列列
        if include_sequence:
            columns_to_extract.append('sequence')
        
        # 过滤不存在的列
        valid_columns = [col for col in columns_to_extract if col in data.columns]
        if len(valid_columns) < len(columns_to_extract):
            missing_cols = set(columns_to_extract) - set(valid_columns)
            logging.warning(f"以下列在原始数据中不存在: {', '.join(missing_cols)}")
        
        # 提取需要的列
        extracted_data = data[valid_columns].copy()
        
        # 如果需要，添加组合排名信息
        if include_metadata:
            extracted_data['feature_rank'] = current_rank
            extracted_data['feature_r2'] = result_row['test_r2']
            extracted_data['feature_set'] = feature_str
        
        # 添加排名信息（仅用于内部标识，不会输出到CSV）
        if format == 'separate':
            extracted_data._rank = current_rank
        
        all_extracted_data.append(extracted_data)
    
    if not all_extracted_data:
        logging.error("没有数据被提取")
        return None
    
    # 根据输出格式返回数据
    if format == 'combined':
        # 合并所有提取的数据
        final_data = pd.concat(all_extracted_data, ignore_index=True)
        logging.info(f"成功提取数据，共{len(final_data)}条记录")
        return [final_data]
    else:
        # 返回单独的DataFrame列表
        logging.info(f"成功提取{len(all_extracted_data)}个特征组合的数据")
        return all_extracted_data

def main():
    """主函数"""
    args = parse_args()
    
    # 加载数据
    data = load_data(args.data)
    if data is None:
        return
    
    # 加载特征评估结果
    feature_results = load_feature_results(args.results)
    if feature_results is None:
        return
    
    # 提取数据
    extracted_data_list = extract_features_by_rank(
        data, 
        feature_results, 
        rank=args.rank, 
        top_n=args.top_n, 
        include_sequence=args.include_sequence,
        format=args.format,
        include_metadata=args.include_metadata
    )
    
    if extracted_data_list is None:
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存结果
    if args.format == 'combined':
        # 合并模式：保存到单个文件
        if args.top_n > 0:
            output_file = os.path.join(args.output, f"top{args.top_n}_features_{timestamp}.csv")
        else:
            output_file = os.path.join(args.output, f"rank{args.rank}_features_{timestamp}.csv")
        
        extracted_data_list[0].to_csv(output_file, index=False)
        logging.info(f"数据已保存至: {output_file}")
    else:
        # 分离模式：每个组合保存到单独的文件
        for df in extracted_data_list:
            rank = df._rank if hasattr(df, '_rank') else 0
            df = df.drop('_rank', axis=1, errors='ignore')  # 删除临时排名列
            output_file = os.path.join(args.output, f"rank{rank}_features_{timestamp}.csv")
            df.to_csv(output_file, index=False)
            logging.info(f"排名第{rank}的特征组合数据已保存至: {output_file}")

if __name__ == "__main__":
    main() 