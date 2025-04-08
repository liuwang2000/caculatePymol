#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import joblib
import argparse
import logging
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from tqdm import tqdm
import time
import datetime
from joblib import Parallel, delayed
import multiprocessing

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

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

def load_data(data_file):
    """加载训练数据"""
    logging.info(f"从{data_file}加载数据")
    try:
        data = pd.read_csv(data_file)
        # 删除没有最适温度的行
        data = data[data['optimal_temperature'].notna() & (data['optimal_temperature'] != '')]
        # 如果最适温度是字符串，转换为浮点数
        if data['optimal_temperature'].dtype == 'object':
            data['optimal_temperature'] = pd.to_numeric(data['optimal_temperature'], errors='coerce')
        # 再次删除NaN值
        data = data.dropna(subset=['optimal_temperature'])
        logging.info(f"成功加载{len(data)}条有效数据")
        return data
    except Exception as e:
        logging.error(f"加载数据失败: {str(e)}")
        raise

def prepare_features(data, include_aa=False):
    """准备所有可能的特征"""
    # 排除不作为特征的列
    exclude_columns = [
        'pdb_id', 'sequence', 'optimal_temperature', 
        'hydrogen_bonds', 'hydrophobic_contacts', 'salt_bridges', 'hydrophobic_sasa'
    ]
    
    # 获取所有列名
    all_columns = data.columns.tolist()
    
    # 排除氨基酸比例特征（除非指定包含它们）
    if not include_aa:
        aa_ratio_columns = [col for col in all_columns if 'aa_' in col.lower()]
        exclude_columns.extend(aa_ratio_columns)
        logging.info(f"排除了{len(aa_ratio_columns)}个氨基酸比例特征")
    else:
        logging.info("保留氨基酸比例特征用于分析")
    
    # 准备特征和目标变量
    feature_cols = [col for col in data.columns if col not in exclude_columns]
    X = data[feature_cols]
    y = data['optimal_temperature']
    
    # 检查是否存在NaN值
    if X.isnull().any().any():
        logging.warning(f"特征中存在NaN值，将进行填充")
        X = X.fillna(X.mean())
    
    return X, y, feature_cols

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return r2, rmse, mae

def feature_selection_by_combination(X, y, feature_cols, min_features=3, max_features=15, 
                                     cv=5, random_state=42, top_k=10, test_size=0.2):
    """通过尝试不同的特征组合来找到最佳特征集"""
    # 检查特征数量是否足够
    n_features = len(feature_cols)
    if n_features <= 1:
        logging.error(f"可用特征数量({n_features})太少，无法进行特征选择。请考虑包含更多特征或启用氨基酸比例特征。")
        return []
    
    # 调整min_features和max_features以确保它们在有效范围内
    if min_features >= n_features:
        min_features = max(1, n_features - 1)
        logging.warning(f"最小特征数量超过可用特征总数，已调整为: {min_features}")
    
    if max_features >= n_features:
        max_features = n_features - 1
        logging.warning(f"最大特征数量超过可用特征总数，已调整为: {max_features}")
    
    logging.info(f"开始特征选择，将尝试从{min_features}到{max_features}个特征的组合")
    
    # 计算总共需要尝试的组合数
    total_combinations = 0
    for num_features in range(min_features, min(max_features + 1, n_features + 1)):
        total_combinations += len(list(combinations(range(n_features), num_features)))
    
    logging.info(f"需要测试{total_combinations}种特征组合")
    
    # 预估所需时间
    avg_time_per_combo = 0.2  # 初步估计每个组合0.2秒
    estimated_time = total_combinations * avg_time_per_combo
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    estimated_time = estimated_time / n_jobs  # 考虑并行加速
    
    estimated_str = format_time(estimated_time)
    logging.info(f"预计完成特征组合测试需要约{estimated_str}，使用{n_jobs}个CPU核心并行处理")
    
    # 创建进度条
    pbar = tqdm(total=total_combinations, desc="测试特征组合")
    
    # 创建结果存储列表
    results = []
    
    # 使用joblib并行处理
    def evaluate_combination(feature_subset):
        # 准备数据
        X_subset = X[list(feature_subset)]
        
        # 数据拆分
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=test_size, random_state=random_state
        )
        
        # 标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 使用与train_rf_model完全相同的基础随机森林配置
        rf = RandomForestRegressor(n_estimators=150, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        # 评估训练集和测试集
        train_pred = rf.predict(X_train_scaled)
        test_pred = rf.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # 交叉验证 - 用于额外参考
        cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='r2')
        cv_r2 = np.mean(cv_scores)
        
        return {
            'features': feature_subset,
            'num_features': len(feature_subset),
            'cv_r2': cv_r2,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'model': rf,
            'scaler': scaler
        }
    
    # 为每个特征数量找到最佳组合
    all_combinations = []
    for num_features in range(min_features, min(max_features + 1, n_features + 1)):
        for feature_subset in combinations(feature_cols, num_features):
            all_combinations.append(feature_subset)
    
    # 记录开始时间
    combo_start_time = time.time()
    
    # 并行执行特征组合评估
    batch_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_combination)(features) 
        for features in all_combinations
    )
    
    # 计算实际耗时
    combo_end_time = time.time()
    combo_elapsed = combo_end_time - combo_start_time
    combo_elapsed_str = format_time(combo_elapsed)
    logging.info(f"特征组合测试完成，实际耗时: {combo_elapsed_str}")
    
    # 更新进度条并收集结果
    pbar.update(total_combinations)
    results.extend(batch_results)
    
    # 关闭进度条
    pbar.close()
    
    # 排序结果（按测试集R^2排序）
    results.sort(key=lambda x: x['test_r2'], reverse=True)
    
    # 返回顶部结果
    return results[:top_k]

def train_and_evaluate_best_combinations(X, y, top_combinations, random_state=42, test_size=0.2):
    """训练并评估最佳特征组合"""
    logging.info("评估最佳特征组合...")
    
    final_results = []
    
    for i, combo in enumerate(top_combinations):
        features = combo['features']
        X_subset = X[list(features)]
        
        # 使用已训练好的模型
        model = combo['model']
        scaler = combo['scaler']
        
        # 获取已有的评估指标
        train_r2 = combo['train_r2']
        test_r2 = combo['test_r2']
        
        # 计算RMSE和MAE（使用combo中的数据拆分结果）
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=test_size, random_state=random_state
        )
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # 存储结果
        result = {
            'rank': i + 1,
            'features': features,
            'num_features': len(features),
            'cv_r2': combo['cv_r2'],
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': rmse,
            'test_mae': mae,
            'model': model,
            'scaler': scaler
        }
        final_results.append(result)
        
        logging.info(f"组合 #{i+1} - 特征数: {len(features)} - 测试集 R²: {test_r2:.4f} - 训练集 R²: {train_r2:.4f}")
    
    # 按测试集R²排序
    final_results.sort(key=lambda x: x['test_r2'], reverse=True)
    
    # 更新排名
    for i, result in enumerate(final_results):
        result['rank'] = i + 1
    
    return final_results

def save_model_and_results(final_results, output_dir):
    """保存最佳模型和结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存最佳模型（基于测试集R^2）
    best_result = max(final_results, key=lambda x: x['test_r2'])
    model_file = os.path.join(output_dir, f"best_model_{timestamp}.pkl")
    joblib.dump({
        'model': best_result['model'],
        'scaler': best_result['scaler'],
        'features': best_result['features']
    }, model_file)
    logging.info(f"最佳模型已保存至 {model_file}")
    logging.info(f"最佳模型 - 测试集 R²: {best_result['test_r2']:.4f}, 训练集 R²: {best_result['train_r2']:.4f}")
    
    # 保存所有结果
    results_df = pd.DataFrame([
        {
            'rank': r['rank'],
            'num_features': r['num_features'],
            'cv_r2': r['cv_r2'],
            'train_r2': r['train_r2'],
            'test_r2': r['test_r2'],
            'test_rmse': r['test_rmse'],
            'test_mae': r['test_mae'],
            'features': ','.join(r['features'])
        }
        for r in final_results
    ])
    
    # 按测试集R²排序
    results_df = results_df.sort_values('test_r2', ascending=False).reset_index(drop=True)
    results_df['rank'] = results_df.index + 1
    
    results_file = os.path.join(output_dir, f"feature_selection_results_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    logging.info(f"结果已保存至 {results_file}")
    
    # 绘制最佳模型的特征重要性
    plot_feature_importance(best_result, output_dir, timestamp)
    
    return model_file, results_file

def plot_feature_importance(best_result, output_dir, timestamp):
    """绘制最佳模型的特征重要性图"""
    model = best_result['model']
    features = best_result['features']
    
    # 获取特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 绘制柱状图
    plt.figure(figsize=(12, 8))
    plt.title('特征重要性', fontsize=16)
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.xlabel('特征名称', fontsize=14)
    plt.ylabel('重要性分数', fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    plot_file = os.path.join(output_dir, f"feature_importance_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"特征重要性图已保存至 {plot_file}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='特征选择和模型训练，找到最优的特征组合以获得最高的R²。')
    parser.add_argument('--data', type=str, required=True, help='训练数据文件路径 (CSV格式)')
    parser.add_argument('--output', type=str, default='./models', help='输出目录，用于保存模型和结果')
    parser.add_argument('--min_features', type=int, default=3, help='最小特征数量')
    parser.add_argument('--max_features', type=int, default=15, help='最大特征数量')
    parser.add_argument('--top_k', type=int, default=10, help='返回的顶部结果数量')
    parser.add_argument('--cv', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--include_aa', action='store_true', help='包含氨基酸比例特征')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    start_time = time.time()
    logging.info("开始特征选择优化过程...")
    logging.info(f"预计总耗时取决于特征数量和CPU核心数，可能需要几分钟到几十分钟")
    
    # 加载数据
    data = load_data(args.data)
    
    # 准备特征
    X, y, feature_cols = prepare_features(data, include_aa=args.include_aa)
    logging.info(f"原始特征数量: {len(feature_cols)}")
    
    # 根据特征数量给出粗略估计
    if len(feature_cols) > 30:
        logging.info(f"特征数量较多 ({len(feature_cols)}个)，请耐心等待...")
    
    # 特征选择
    top_combinations = feature_selection_by_combination(
        X, y, feature_cols,
        min_features=args.min_features,
        max_features=args.max_features,
        cv=args.cv,
        top_k=args.top_k,
        random_state=args.random_state,
        test_size=args.test_size
    )
    
    # 检查是否有有效的特征组合
    if not top_combinations:
        logging.error("未找到有效的特征组合，请尝试包含更多特征或修改参数。")
        return
    
    # 训练并评估最佳组合
    final_results = train_and_evaluate_best_combinations(
        X, y, top_combinations,
        random_state=args.random_state,
        test_size=args.test_size
    )
    
    # 保存模型和结果
    model_file, results_file = save_model_and_results(final_results, args.output)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_str = format_time(elapsed_time)
    logging.info(f"特征选择优化完成，总耗时: {elapsed_str}")
    # 打印最佳测试集R²的模型信息
    logging.info(f"最佳特征组合(测试集R²最高)的测试集 R²: {max(final_results, key=lambda x: x['test_r2'])['test_r2']:.4f}")
    logging.info(f"最佳模型已保存至: {model_file}")
    logging.info(f"结果摘要已保存至: {results_file}")

if __name__ == "__main__":
    main() 