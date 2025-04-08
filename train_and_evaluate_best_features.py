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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from itertools import combinations
from tqdm import tqdm
import time
import datetime
import multiprocessing
from joblib import Parallel, delayed
import sys

# 从train_rf_model.py导入必要的函数
from train_rf_model import setup_chinese_font, load_data, plot_feature_importance, plot_predictions, plot_residuals

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 设置中文字体
setup_chinese_font()

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

def prepare_features(data, exclude_aa=True):
    """准备特征数据和目标变量"""
    # 排除不作为特征的列
    exclude_columns = [
        'pdb_id', 'sequence', 'optimal_temperature', 
        'hydrogen_bonds', 'hydrophobic_contacts', 'salt_bridges', 'hydrophobic_sasa'
    ]
    
    # 准备特征和目标变量
    feature_cols = [col for col in data.columns if col not in exclude_columns]
    
    # 如果需要排除氨基酸比例特征
    if exclude_aa:
        feature_cols = [col for col in feature_cols if not col.startswith('aa_')]
    
    # 验证特征列都存在于数据中
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        logging.warning(f"以下特征在数据集中不存在: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in data.columns]
        if not feature_cols:
            logging.error("没有有效的特征列")
            return None, None, []
    
    # 提取特征数据
    X = data[feature_cols]
    y = data['optimal_temperature']
    
    # 检查是否存在NaN值
    if X.isnull().any().any():
        logging.warning(f"特征中存在NaN值，将进行填充")
        X = X.fillna(X.mean())
    
    return X, y, feature_cols

def train_and_evaluate_rf(X, y, feature_subset, test_size=0.2, random_state=42, return_model=True):
    """训练随机森林模型并评估性能"""
    # 将元组转换为列表，防止pandas索引错误
    if isinstance(feature_subset, tuple):
        feature_subset = list(feature_subset)
        
    # 检查所有特征是否都在X中
    missing_features = [f for f in feature_subset if f not in X.columns]
    if missing_features:
        logging.warning(f"特征子集中有不存在的特征: {missing_features}")
        # 返回错误结果
        return {
            'train_r2': -1.0,
            'test_r2': -1.0,
            'test_rmse': float('inf'),
            'test_mae': float('inf'),
            'feature_names': feature_subset,
            'num_features': len(feature_subset),
            'error': f"特征不存在: {missing_features}"
        }
    
    # 准备数据
    X_subset = X[feature_subset]
    
    # 检查是否存在NaN值
    if X_subset.isnull().any().any():
        logging.warning(f"特征子集中存在NaN值，将进行填充")
        X_subset = X_subset.fillna(X_subset.mean())
    
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=test_size, random_state=random_state
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 使用与train_rf_model.py相同的基础随机森林配置
    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # 预测
    y_train_pred = rf.predict(X_train_scaled)
    y_test_pred = rf.predict(X_test_scaled)
    
    # 评估指标
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # 返回结果
    result = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'feature_names': feature_subset,
        'num_features': len(feature_subset)
    }
    
    if return_model:
        result['model'] = rf
        result['scaler'] = scaler
        result['X_test'] = X_test
        result['y_test'] = y_test
        result['y_pred'] = y_test_pred
    
    return result

def generate_feature_combinations(features, min_features=3, max_features=15, max_combinations=None):
    """生成特征组合列表"""
    all_combinations = []
    n_features = len(features)
    
    # 调整max_features不超过特征总数
    max_features = min(max_features, n_features)
    
    # 计算组合数
    total_combinations = 0
    for i in range(min_features, max_features + 1):
        from scipy.special import comb
        total_combinations += int(comb(n_features, i))
    
    logging.info(f"包含{min_features}到{max_features}个特征的组合总数: {total_combinations}")
    
    # 如果设置了最大组合数且小于总组合数
    if max_combinations and max_combinations < total_combinations:
        logging.info(f"由于设置了限制，将随机选择{max_combinations}个组合进行测试")
        import random
        random.seed(42)
        
        for i in range(min_features, max_features + 1):
            # 每个特征数量的组合
            combos = list(combinations(features, i))
            # 为每个特征数量分配一定数量的组合
            num_for_this_size = max(1, int(max_combinations * len(combos) / total_combinations))
            # 随机选择
            selected = random.sample(combos, min(num_for_this_size, len(combos)))
            all_combinations.extend(selected)
        
        # 再次随机采样确保总数不超过max_combinations
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)
    else:
        # 生成所有组合
        for i in range(min_features, max_features + 1):
            all_combinations.extend(combinations(features, i))
    
    logging.info(f"实际将测试{len(all_combinations)}个特征组合")
    return all_combinations

def evaluate_feature_combinations(X, y, feature_combinations, test_size=0.2, random_state=42, n_jobs=-1):
    """评估所有特征组合"""
    if n_jobs == -1:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    logging.info(f"使用{n_jobs}个CPU核心并行评估特征组合")
    
    start_time = time.time()
    
    # 并行处理函数
    def evaluate_one_combination(combo):
        try:
            # 将元组转换为列表，防止pandas索引错误
            if isinstance(combo, tuple):
                combo = list(combo)
                
            # 检查特征组合中的每个特征是否在数据集中
            invalid_features = [f for f in combo if f not in X.columns]
            if invalid_features:
                logging.warning(f"特征组合中包含无效特征: {invalid_features}")
                # 返回错误结果而不是引发异常
                return {
                    'train_r2': -1.0,
                    'test_r2': -1.0,
                    'test_rmse': float('inf'),
                    'test_mae': float('inf'),
                    'feature_names': combo,
                    'num_features': len(combo),
                    'error': f"特征不存在: {invalid_features}"
                }
            
            return train_and_evaluate_rf(X, y, combo, test_size, random_state)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.warning(f"评估特征组合时出错: {combo}, 错误: {str(e)}\n{error_details}")
            # 返回带有详细错误信息的结果
            return {
                'train_r2': -1.0,
                'test_r2': -1.0,
                'test_rmse': float('inf'),
                'test_mae': float('inf'),
                'feature_names': combo,
                'num_features': len(combo),
                'error': str(e)
            }
    
    # 使用tqdm显示进度
    results = []
    
    # 使用更健壮的批处理方式
    batch_size = min(n_jobs, 10)  # 限制批量大小，避免内存问题
    total_batches = (len(feature_combinations) + batch_size - 1) // batch_size
    
    with tqdm(total=len(feature_combinations), desc="评估特征组合") as pbar:
        for i in range(0, len(feature_combinations), batch_size):
            batch = feature_combinations[i:i+batch_size]
            try:
                # 使用更安全的并行处理方式
                batch_results = Parallel(n_jobs=min(n_jobs, len(batch)), timeout=None, backend='loky')(
                    delayed(evaluate_one_combination)(combo) for combo in batch
                )
                results.extend(batch_results)
                pbar.update(len(batch))
            except Exception as e:
                logging.error(f"批量处理时出错: {str(e)}")
                # 回退到顺序处理
                for combo in batch:
                    try:
                        result = evaluate_one_combination(combo)
                        results.append(result)
                    except Exception as e2:
                        logging.error(f"顺序处理特征组合时出错: {combo}, 错误: {str(e2)}")
                        # 添加一个标记为错误的结果占位符
                        results.append({
                            'train_r2': -1.0,
                            'test_r2': -1.0,
                            'test_rmse': float('inf'),
                            'test_mae': float('inf'),
                            'feature_names': combo,
                            'num_features': len(combo),
                            'error': str(e2)
                        })
                    pbar.update(1)
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    elapsed_str = format_time(elapsed_time)
    logging.info(f"特征组合评估完成，耗时: {elapsed_str}")
    
    # 确保结果不为空，避免返回None
    if not results:
        logging.warning("没有生成任何结果，返回空列表")
        return []
    
    # 过滤掉错误的结果并按测试集R²排序
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        logging.warning("没有成功评估的特征组合，返回所有结果")
        # 即使都失败了，也返回所有结果，让调用者处理
        return results
    
    # 按测试集R²排序
    valid_results.sort(key=lambda x: x['test_r2'], reverse=True)
    
    # 合并有效结果和错误结果
    all_results = valid_results + [r for r in results if 'error' in r]
    
    logging.info(f"成功评估{len(valid_results)}/{len(results)}个特征组合")
    
    return all_results

def save_results(results, output_dir):
    """保存评估结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存所有结果到CSV (包括错误的结果)
    results_df = pd.DataFrame([
        {
            'rank': i+1,
            'features': ','.join(r['feature_names']),
            'num_features': r['num_features'],
            'test_r2': r['test_r2'],
            'train_r2': r['train_r2'],
            'test_rmse': r['test_rmse'],
            'test_mae': r['test_mae'],
            'error': r.get('error', '')  # 添加错误信息
        }
        for i, r in enumerate(results)
    ])
    
    # 保存CSV
    csv_path = os.path.join(output_dir, f'feature_evaluation_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    logging.info(f"结果摘要已保存至: {csv_path}")
    
    # 检查是否有有效的结果
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        logging.error("没有有效的评估结果，无法保存模型")
        return csv_path, None
    
    try:
        # 保存最佳模型 (仅使用有效的结果)
        best_result = valid_results[0]
        
        # 确保最佳结果包含必要的模型数据
        required_keys = ['model', 'feature_names', 'scaler', 'y_test', 'y_pred']
        missing_keys = [k for k in required_keys if k not in best_result]
        if missing_keys:
            logging.error(f"最佳结果缺少必要的键: {missing_keys}")
            return csv_path, None
        
        model_package = {
            'model': best_result['model'],
            'feature_names': best_result['feature_names'],
            'scaler': best_result['scaler'],
            'model_type': 'RandomForest',
            'performance': {
                'test_r2': best_result['test_r2'],
                'train_r2': best_result['train_r2'],
                'test_rmse': best_result['test_rmse'],
                'test_mae': best_result['test_mae']
            }
        }
        
        # 保存模型
        model_path = os.path.join(output_dir, f'best_feature_model_{timestamp}.joblib')
        joblib.dump(model_package, model_path)
        logging.info(f"最佳模型已保存至: {model_path}")
        
        # 生成可视化
        try:
            # 特征重要性
            plot_feature_importance(best_result['model'], 
                                    pd.DataFrame(columns=best_result['feature_names']), 
                                    os.path.join(output_dir, f'feature_importance_{timestamp}.png'))
            
            # 预测对比图
            plot_predictions(best_result['y_test'], best_result['y_pred'], 
                             output_dir, suffix=f'_{timestamp}')
            
            # 残差图
            plot_residuals(best_result['y_test'], best_result['y_pred'], 
                          output_dir, suffix=f'_{timestamp}')
        except Exception as e:
            logging.warning(f"创建可视化时出错: {str(e)}")
        
        # 输出最佳特征组合详情
        logging.info("最佳特征组合详情:")
        logging.info(f"测试集 R²: {best_result['test_r2']:.4f}")
        logging.info(f"训练集 R²: {best_result['train_r2']:.4f}")
        logging.info(f"测试集 RMSE: {best_result['test_rmse']:.4f}")
        logging.info(f"测试集 MAE: {best_result['test_mae']:.4f}")
        logging.info(f"特征数量: {best_result['num_features']}")
        logging.info(f"特征列表: {', '.join(best_result['feature_names'])}")
        
        return csv_path, model_path
    except Exception as e:
        logging.error(f"保存最佳模型时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return csv_path, None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用与train_rf_model相同的模型评估特征组合')
    parser.add_argument('--data', type=str, required=True, help='训练数据文件路径 (CSV格式)')
    parser.add_argument('--output', type=str, default='./rf_feature_models', help='输出目录')
    parser.add_argument('--min_features', type=int, default=3, help='最小特征数量')
    parser.add_argument('--max_features', type=int, default=15, help='最大特征数量')
    parser.add_argument('--max_combinations', type=int, default=None, help='最多评估的组合数量')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--include_aa', action='store_true', help='包含氨基酸比例特征')
    parser.add_argument('--n_jobs', type=int, default=-1, help='并行处理的核心数 (-1表示使用全部可用核心减1)')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    logging.info("开始使用与train_rf_model相同的随机森林模型评估特征组合")
    
    # 加载数据
    data = load_data(args.data)
    if data is None:
        return
    
    # 打印所有可用的列名，帮助调试
    logging.info(f"数据集中的列名: {list(data.columns)}")
    
    # 准备特征
    X, y, feature_cols = prepare_features(data, exclude_aa=not args.include_aa)
    logging.info(f"加载了{len(feature_cols)}个特征")
    
    # 验证特征列是否都在数据集中存在
    missing_cols = [col for col in feature_cols if col not in X.columns]
    if missing_cols:
        logging.error(f"以下特征在数据集中不存在: {missing_cols}")
        logging.info("可用特征列: {}".format(", ".join(X.columns.tolist())))
        return
    
    # 确保有足够的特征进行组合
    if len(feature_cols) < args.min_features:
        logging.error(f"可用特征数量({len(feature_cols)})小于最小特征数量({args.min_features})")
        return

    # 调整最大特征数，确保不超过可用特征数
    if args.max_features > len(feature_cols):
        old_max = args.max_features
        args.max_features = len(feature_cols)
        logging.warning(f"最大特征数量已从{old_max}调整为{args.max_features}(可用特征总数)")
    
    # 打印可用特征列表供参考
    logging.info(f"可用特征列表:")
    for i, feature in enumerate(feature_cols):
        logging.info(f"  {i+1}. {feature}")
    
    try:
        # 生成特征组合
        feature_combinations = generate_feature_combinations(
            feature_cols, 
            min_features=args.min_features,
            max_features=args.max_features,
            max_combinations=args.max_combinations
        )
        
        if not feature_combinations:
            logging.error("未生成任何有效的特征组合")
            return
        
        # 评估特征组合
        results = evaluate_feature_combinations(
            X, y, 
            feature_combinations, 
            test_size=args.test_size,
            random_state=args.random_state,
            n_jobs=args.n_jobs
        )
        
        # 确保结果不为None且是可迭代对象
        if results is None:
            logging.error("特征评估返回了None结果")
            results = []
        
        # 过滤掉错误的结果
        valid_results = [r for r in results if r is not None and 'error' not in r]
        
        if not valid_results:
            logging.error("所有特征组合评估都失败了")
            if results:  # 如果至少有一些结果（即使全是错误）
                # 保存错误结果以便分析
                csv_path, _ = save_results(results, args.output)
                logging.info(f"已保存错误结果摘要: {csv_path}")
            return
        
        # 按测试集R²排序
        valid_results.sort(key=lambda x: x['test_r2'], reverse=True)
        
        # 保存结果
        csv_path, model_path = save_results(valid_results, args.output)
        
        logging.info(f"评估完成！")
        logging.info(f"结果摘要: {csv_path}")
        if model_path:
            logging.info(f"最佳模型: {model_path}")
            logging.info(f"最佳R²: {valid_results[0]['test_r2']:.4f}")
        else:
            logging.warning("未能保存最佳模型")
    
    except Exception as e:
        logging.error(f"评估过程中出现错误: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 