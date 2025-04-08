#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import argparse
import logging
import joblib
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import platform
from tqdm import tqdm
import datetime
from sklearn.ensemble import RandomForestRegressor

# 从train_rf_model.py导入所需的函数
from train_rf_model import setup_chinese_font, load_data, train_model
from train_rf_model import plot_feature_importance, plot_predictions, plot_residuals

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 设置中文字体
setup_chinese_font()

def load_all_feature_combinations(results_file):
    """从特征选择结果文件中加载所有特征组合"""
    try:
        results_df = pd.read_csv(results_file)
        if len(results_df) == 0:
            logging.error(f"结果文件 {results_file} 为空")
            return None
        
        num_combinations = len(results_df)
        logging.info(f"已加载{num_combinations}个特征组合")
        
        # 创建特征组合列表
        feature_combinations = []
        for i, row in results_df.iterrows():
            features = row['features'].split(',')
            test_r2 = row['test_r2']
            feature_combinations.append({
                'rank': i + 1,
                'features': features,
                'test_r2': test_r2
            })
            
        logging.info(f"排名第一的特征组合: {', '.join(feature_combinations[0]['features'])}")
        logging.info(f"特征组合数量: {len(feature_combinations)}")
        
        return feature_combinations
    except Exception as e:
        logging.error(f"加载特征选择结果失败: {str(e)}")
        return None

def prepare_selected_features(data, selected_features, use_advanced_processing=False):
    """准备选定的特征，使用与train_and_evaluate_best_features.py相同的方法"""
    # 检查所有选定的特征是否都存在于数据中
    missing_features = [f for f in selected_features if f not in data.columns]
    if missing_features:
        logging.error(f"以下特征在数据中不存在: {', '.join(missing_features)}")
        return None, None
    
    # 准备特征和目标变量
    X = data[selected_features].copy()  # 使用copy避免警告
    y = data['optimal_temperature']
    
    # 检查是否存在NaN值
    if X.isnull().any().any():
        logging.warning(f"特征中存在NaN值，将进行填充")
        X = X.fillna(X.mean())
    
    # 如果需要高级特征处理
    if use_advanced_processing:
        logging.info("应用高级特征处理...")
        # 应用非线性变换 (来自train_rf_model.py)
        X_transformed = X.copy()
        
        # 默认变换：对适合的特征应用对数、平方、平方根变换
        for col in X.columns:
            # 确保特征都是正数才应用对数变换
            if X[col].min() > 0:
                # 对数变换
                X_transformed[f"{col}_log"] = np.log1p(X[col])
                
                # 平方根变换
                X_transformed[f"{col}_sqrt"] = np.sqrt(X[col])
            
            # 平方变换
            X_transformed[f"{col}_squared"] = X[col] ** 2
        
        # 只对原始特征的一部分应用多项式特征，避免维度爆炸
        if len(X.columns) > 2:
            from sklearn.preprocessing import PolynomialFeatures
            # 只对前3个特征应用多项式变换
            top_cols = X.columns[:min(3, len(X.columns))]
            poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
            try:
                poly_features = poly.fit_transform(X[top_cols])
                feature_names = poly.get_feature_names_out(top_cols)
                
                # 添加多项式特征
                for i, name in enumerate(feature_names):
                    if i >= len(top_cols):  # 跳过原始特征
                        X_transformed[name] = poly_features[:, i]
            except Exception as e:
                logging.warning(f"多项式特征生成失败: {str(e)}")
        
        logging.info(f"特征处理前: {X.shape[1]}个特征, 处理后: {X_transformed.shape[1]}个特征")
        X = X_transformed
    
    return X, y

def train_and_evaluate(X, y, selected_features, output_dir=None, use_advanced_models=False):
    """训练并评估模型，使用与train_and_evaluate_rf相同的基础方法"""
    try:
        # 将特征子集列表转换为列表，防止pandas索引错误
        if isinstance(selected_features, tuple):
            selected_features = list(selected_features)
            
        # 检查所有特征是否都在X中
        missing_features = [f for f in selected_features if f not in X.columns]
        if missing_features:
            logging.warning(f"特征子集中有不存在的特征: {missing_features}")
            return None
        
        # 数据拆分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 根据参数选择使用高级模型或基础随机森林
        if use_advanced_models:
            logging.info("使用高级模型训练")
            # 创建一个临时DataFrame传递给train_model
            X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            model, _, _, y_pred, model_type = train_model(
                pd.concat([X_train_df, X_test_df]), 
                pd.concat([y_train, y_test]),
                output_dir, 
                use_advanced_models=True
            )
        else:
            # 使用与train_and_evaluate_best_features.py相同的基础随机森林配置
            model = RandomForestRegressor(n_estimators=150, random_state=42)
            model.fit(X_train_scaled, y_train)
            model_type = "RandomForest"
            y_pred = model.predict(X_test_scaled)
        
        # 计算评估指标
        y_train_pred = model.predict(X_train_scaled)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        
        logging.info(f"最终模型: {model_type}")
        logging.info(f"训练集 R²: {train_r2:.4f}")
        logging.info(f"测试集 R²: {test_r2:.4f}")
        logging.info(f"测试集 RMSE: {test_rmse:.4f}")
        logging.info(f"测试集 MAE: {test_mae:.4f}")
        
        # 创建模型包
        model_package = {
            'model': model,
            'feature_names': selected_features,
            'scaler': scaler,
            'model_type': model_type,
            'performance': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae
            },
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return model_package
    except Exception as e:
        logging.error(f"训练和评估过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

# 自定义可视化函数，解决路径问题
def plot_predictions(y_true, y_pred, output_path):
    """绘制预测值与真实值对比图"""
    plt.figure(figsize=(10, 6))
    
    # 计算相关系数
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 绘制散点图
    plt.scatter(y_true, y_pred, alpha=0.7)
    
    # 添加对角线 (理想预测线)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 添加相关系数和RMSE信息
    plt.text(min_val + 0.05*(max_val-min_val), max_val - 0.1*(max_val-min_val), 
             f'R = {correlation:.2f}\nRMSE = {rmse:.2f}°C', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('真实最适温度 (°C)')
    plt.ylabel('预测最适温度 (°C)')
    plt.title('模型 - 预测值与真实值对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path)
    logging.info(f"预测对比图已保存至: {output_path}")
    plt.close()

def plot_residuals(y_true, y_pred, output_path):
    """绘制残差图"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    
    # 绘制残差散点图
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    
    # 计算残差统计量
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # 添加残差统计信息
    plt.text(min(y_pred) + 0.05*(max(y_pred)-min(y_pred)), 
             max(residuals) - 0.2*(max(residuals)-min(residuals)), 
             f'均值 = {mean_residual:.2f}\n标准差 = {std_residual:.2f}', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('预测值 (°C)')
    plt.ylabel('残差 (实际值 - 预测值)')
    plt.title('模型残差分析')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path)
    logging.info(f"残差分析图已保存至: {output_path}")
    plt.close()

def save_best_model(model_package, output_dir, suffix=""):
    """保存最佳模型"""
    if output_dir is None:
        output_dir = "./best_feature_model"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f'best_feature_model{suffix}_{timestamp}.joblib')
    joblib.dump(model_package, model_path)
    logging.info(f"模型已保存至: {model_path}")
    
    # 创建可视化图表
    try:
        # 创建一个临时DataFrame用于plot_feature_importance
        X_dummy = pd.DataFrame(columns=model_package['feature_names'])
        
        # 输出可视化
        feature_importance_path = os.path.join(output_dir, f'feature_importance{suffix}_{timestamp}.png')
        plot_feature_importance(model_package['model'], X_dummy, feature_importance_path)
        
        # 获取测试数据和预测数据，如果它们存在
        model = model_package['model']
        
        # 检查模型包中是否有测试数据
        if 'performance' in model_package and 'y_test' in model_package and 'y_pred' in model_package:
            y_test = model_package['y_test']
            y_pred = model_package['y_pred']
            
            # 预测图路径
            pred_path = os.path.join(output_dir, f'predictions_vs_actual{suffix}_{timestamp}.png')
            res_path = os.path.join(output_dir, f'residuals{suffix}_{timestamp}.png')
            
            plot_predictions(y_test, y_pred, pred_path)
            plot_residuals(y_test, y_pred, res_path)
        else:
            logging.warning("模型包中没有测试数据，无法生成预测图和残差图")
    except Exception as e:
        logging.warning(f"创建可视化图表时出错: {str(e)}")
        import traceback
        logging.warning(traceback.format_exc())
    
    return model_path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用最优特征组合训练模型')
    parser.add_argument('--data', type=str, required=True, help='训练数据文件路径 (CSV格式)')
    parser.add_argument('--results', type=str, required=True, help='特征选择结果文件路径 (CSV格式)')
    parser.add_argument('--output', type=str, default='./best_feature_model', help='输出目录，用于保存模型和结果')
    parser.add_argument('--advanced_models', action='store_true', help='使用高级模型进行训练')
    parser.add_argument('--advanced_features', action='store_true', help='应用高级特征处理（非线性变换、交互特征等）')
    parser.add_argument('--max_combinations', type=int, default=0, help='最多训练的特征组合数量 (0表示全部)')
    parser.add_argument('--save_all', action='store_true', help='保存所有训练的模型，而不仅仅是最佳模型')
    parser.add_argument('--verbose', action='store_true', help='显示更详细的训练过程信息')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("详细模式已启用")
    
    logging.info("开始使用特征组合训练模型...")
    
    # 加载所有特征组合
    feature_combinations = load_all_feature_combinations(args.results)
    if feature_combinations is None:
        return
    
    # 如果设置了最大组合数，则限制组合数量
    if args.max_combinations > 0 and args.max_combinations < len(feature_combinations):
        logging.info(f"将只训练前{args.max_combinations}个特征组合")
        feature_combinations = feature_combinations[:args.max_combinations]
    
    # 加载数据
    data = load_data(args.data)
    if data is None:
        return
    
    # 用于存储所有模型的结果
    all_models = []
    
    # 训练所有特征组合
    for combo in tqdm(feature_combinations, desc="训练特征组合"):
        rank = combo['rank']
        features = combo['features']
        old_test_r2 = combo['test_r2']
        
        logging.info(f"训练第{rank}名特征组合 (原有测试集R²: {old_test_r2:.4f})")
        logging.info(f"特征: {', '.join(features)}")
        
        # 准备特征
        X, y = prepare_selected_features(data, features, use_advanced_processing=args.advanced_features)
        if X is None:
            logging.warning(f"跳过第{rank}名特征组合，无法准备特征")
            continue
        
        # 如果开启了高级特征处理，打印特征信息
        if args.advanced_features:
            logging.info(f"应用高级特征处理后，特征数量: {X.shape[1]}")
            if args.verbose:
                logging.debug(f"特征列: {list(X.columns)}")
        
        # 训练并评估模型
        model_package = train_and_evaluate(X, y, list(X.columns), args.output if args.save_all else None, args.advanced_models)
        
        # 检查模型训练是否成功
        if model_package is None:
            logging.warning(f"第{rank}名特征组合的模型训练失败，跳过")
            continue
            
        # 记录训练结果和排名信息
        model_package['rank'] = rank
        model_package['old_test_r2'] = old_test_r2
        all_models.append(model_package)
        
        # 如果需要保存所有模型
        if args.save_all:
            save_best_model(model_package, args.output, suffix=f"_rank{rank}")
    
    # 如果没有成功训练任何模型，退出
    if len(all_models) == 0:
        logging.error("没有成功训练任何模型")
        return
    
    # 按测试集R²排序
    all_models.sort(key=lambda x: x['performance']['test_r2'], reverse=True)
    
    # 获取最佳模型
    best_model = all_models[0]
    best_rank = best_model['rank']
    best_test_r2 = best_model['performance']['test_r2']
    best_features = best_model['feature_names']
    
    logging.info("=" * 60)
    logging.info(f"最佳模型来自原始排名第{best_rank}名的特征组合")
    logging.info(f"特征数量: {len(best_features)}")
    logging.info(f"特征: {', '.join(best_features)}")
    logging.info(f"测试集 R²: {best_test_r2:.4f}")
    logging.info(f"测试集 RMSE: {best_model['performance']['test_rmse']:.4f}")
    logging.info(f"测试集 MAE: {best_model['performance']['test_mae']:.4f}")
    
    # 保存最佳模型
    best_model_path = save_best_model(best_model, args.output)
    
    # 创建结果摘要
    results_summary = pd.DataFrame([
        {
            'original_rank': m['rank'],
            'features': ','.join(m['feature_names']),
            'num_features': len(m['feature_names']),
            'old_test_r2': m['old_test_r2'],
            'new_test_r2': m['performance']['test_r2'],
            'test_rmse': m['performance']['test_rmse'],
            'test_mae': m['performance']['test_mae'],
            'model_type': m['model_type']
        }
        for m in all_models
    ])
    
    # 按新的测试集R²排序
    results_summary = results_summary.sort_values('new_test_r2', ascending=False).reset_index(drop=True)
    
    # 保存结果摘要
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.output, f'model_comparison_results_{timestamp}.csv')
    results_summary.to_csv(summary_path, index=False)
    logging.info(f"模型比较结果已保存至: {summary_path}")
    
    logging.info("模型训练完成！")

if __name__ == "__main__":
    main() 