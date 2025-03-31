#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import joblib
import argparse
import logging
import subprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import pathlib
import time
import glob

# 配置中文字体支持
def setup_chinese_font():
    """配置matplotlib支持中文显示"""
    try:
        # 判断操作系统类型
        if sys.platform.startswith('win'):
            # Windows系统，尝试使用微软雅黑
            mpl.rc('font', family='Microsoft YaHei')
        elif sys.platform.startswith('darwin'):
            # macOS系统，尝试使用苹方
            mpl.rc('font', family='PingFang SC')
        else:
            # Linux系统，尝试使用文泉驿微米黑
            mpl.rc('font', family='WenQuanYi Micro Hei')
        
        # 解决负号显示问题
        mpl.rcParams['axes.unicode_minus'] = False
        logging.info("成功配置中文字体支持")
    except Exception as e:
        logging.warning(f"配置中文字体失败: {str(e)}")
        logging.warning("图表中的中文可能无法正确显示")

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

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

def prepare_features(data):
    """准备特征数据和目标变量"""
    # 排除不作为特征的列
    exclude_columns = ['pdb_id', 'sequence', 'optimal_temperature']
    
    # 准备特征和目标变量
    feature_cols = [col for col in data.columns if col not in exclude_columns]
    X = data[feature_cols]
    y = data['optimal_temperature']
    
    # 检查是否存在NaN值
    if X.isnull().any().any():
        logging.warning(f"特征中存在NaN值，将进行填充")
        X = X.fillna(X.mean())
    
    return X, y, feature_cols

def train_model(X, y, output_dir=None):
    """训练随机森林模型"""
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 创建并训练基础随机森林模型
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    logging.info("基础随机森林模型训练完成")
    
    # 模型评估
    train_pred = rf_model.predict(X_train)
    test_pred = rf_model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    logging.info(f"训练集 RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.2f}")
    logging.info(f"测试集 RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.2f}")
    
    # 进行网格搜索优化超参数
    logging.info("开始网格搜索优化模型...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    grid_search.fit(X_train, y_train)
    logging.info(f"最佳参数: {grid_search.best_params_}")
    
    # 使用最佳参数训练最终模型
    best_rf_model = RandomForestRegressor(
        **grid_search.best_params_,
        random_state=42
    )
    best_rf_model.fit(X_train, y_train)
    logging.info("优化后的随机森林模型训练完成")
    
    # 评估最终模型
    best_train_pred = best_rf_model.predict(X_train)
    best_test_pred = best_rf_model.predict(X_test)
    
    best_train_rmse = np.sqrt(mean_squared_error(y_train, best_train_pred))
    best_test_rmse = np.sqrt(mean_squared_error(y_test, best_test_pred))
    best_train_mae = mean_absolute_error(y_train, best_train_pred)
    best_test_mae = mean_absolute_error(y_test, best_test_pred)
    best_train_r2 = r2_score(y_train, best_train_pred)
    best_test_r2 = r2_score(y_test, best_test_pred)
    
    logging.info(f"优化后 - 训练集 RMSE: {best_train_rmse:.2f}, MAE: {best_train_mae:.2f}, R²: {best_train_r2:.2f}")
    logging.info(f"优化后 - 测试集 RMSE: {best_test_rmse:.2f}, MAE: {best_test_mae:.2f}, R²: {best_test_r2:.2f}")
    
    # 保存模型
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'rf_temperature_predictor.joblib')
    joblib.dump(best_rf_model, model_path)
    logging.info(f"模型已保存至: {model_path}")
    
    # 生成特征重要性图
    plot_feature_importance(best_rf_model, X, output_dir)
    
    # 生成预测值与真实值对比图
    plot_predictions(y_test, best_test_pred, output_dir)
    
    return best_rf_model, X_test, y_test, best_test_pred

def plot_feature_importance(model, X, output_dir):
    """绘制特征重要性图"""
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 选择前20个特征
    top_features = feature_importance.head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.title('随机森林模型 - 特征重要性 (前20)')
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    logging.info(f"特征重要性图已保存至: {plot_path}")
    plt.close()

def plot_predictions(y_true, y_pred, output_dir):
    """绘制预测值与真实值对比图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    
    # 添加对角线 (理想预测线)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('真实最适温度')
    plt.ylabel('预测最适温度')
    plt.title('随机森林模型 - 预测值与真实值对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, 'predictions_vs_actual.png')
    plt.savefig(plot_path)
    logging.info(f"预测对比图已保存至: {plot_path}")
    plt.close()

def process_directory(directory_path, model_path, output_dir=None):
    """处理目录中的所有PDB文件"""
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output')
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"处理目录: {directory_path}")
    
    # 检查输入是否是目录
    if not os.path.isdir(directory_path):
        logging.error(f"输入路径不是目录: {directory_path}")
        return False
    
    # 查找所有分析文件
    analyze_files = glob.glob(os.path.join('output', 'analyze_pdb_*.csv'))
    logging.info(f"在目录 output 中找到 {len(analyze_files)} 个分析文件")
    
    if not analyze_files:
        logging.warning("未找到任何现有分析文件，将运行 analyze_pdb.py")
        # 直接调用analyze_pdb.py处理整个目录
        cmd = ['python', 'analyze_pdb.py', directory_path]
        logging.info(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 检查是否执行成功
        if result.returncode != 0:
            logging.error(f"分析PDB文件失败: {result.stderr}")
            return False
        
        # 等待两秒确保文件已经写入磁盘
        logging.info("等待文件系统更新...")
        time.sleep(2)
        
        # 再次查找分析文件
        analyze_files = glob.glob(os.path.join('output', 'analyze_pdb_*.csv'))
        logging.info(f"再次检查，找到 {len(analyze_files)} 个分析文件")
        if not analyze_files:
            logging.error("运行analyze_pdb.py后仍未找到任何分析结果文件")
            logging.info("尝试查找所有CSV文件...")
            all_csv_files = glob.glob(os.path.join('output', '*.csv'))
            logging.info(f"在输出目录中找到 {len(all_csv_files)} 个CSV文件")
            if all_csv_files:
                for file in all_csv_files:
                    logging.info(f"CSV文件: {os.path.basename(file)}")
            return False
    
    # 使用最新的分析文件
    result_file = max(analyze_files, key=os.path.getmtime)
    logging.info(f"使用分析文件: {os.path.basename(result_file)}")
    
    # 加载分析结果
    try:
        pdb_data = pd.read_csv(result_file)
        logging.info(f"成功加载分析结果，包含 {len(pdb_data)} 条记录")
    except Exception as e:
        logging.error(f"无法加载分析文件: {str(e)}")
        return False
    
    # 预测温度
    try:
        # 加载模型
        model = joblib.load(model_path)
        logging.info(f"已加载模型: {model_path}")
        
        # 准备特征
        feature_cols = model.feature_names_in_
        
        # 检查缺失的特征并添加
        for col in feature_cols:
            if col not in pdb_data.columns:
                logging.warning(f"在输入数据中未找到特征: {col}，添加并设置为0")
                pdb_data[col] = 0.0
        
        # 确保只选择模型需要的特征列
        X = pdb_data[feature_cols]
        
        # 检查并填充NaN值
        if X.isnull().any().any():
            logging.warning("特征中存在NaN值，将进行填充")
            X = X.fillna(X.mean())
        
        # 预测
        predictions = model.predict(X)
        
        # 添加预测结果到DataFrame
        pdb_data['predicted_temperature'] = predictions
        
        # 重排列，将pdb_id和预测温度放在前面
        columns = ['pdb_id', 'predicted_temperature'] + [col for col in pdb_data.columns 
                                                        if col not in ['pdb_id', 'predicted_temperature']]
        pdb_data = pdb_data[columns]
        
        # 保存结果
        result_file = os.path.join(os.getcwd(), 'prediction_results.csv')
        pdb_data.to_csv(result_file, index=False)
        logging.info(f"预测结果已保存至: {result_file}")
        
        # 打印每个PDB文件的预测结果
        for index, row in pdb_data.iterrows():
            logging.info(f"预测结果 - {row['pdb_id']}: {row['predicted_temperature']:.2f}°C")
        
        return True
    except Exception as e:
        logging.error(f"预测过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用随机森林训练最适温度预测模型')
    parser.add_argument('--data', default='./trainData/analyze_pdb_merged_20250331_164045.csv',
                      help='训练数据文件路径')
    parser.add_argument('--output', default='./models',
                      help='模型输出目录')
    parser.add_argument('--predict', default=None,
                      help='预测模式：指定要预测的PDB文件或目录路径')
    parser.add_argument('--model', default=None,
                      help='预测模式：指定要使用的模型文件路径')
    return parser.parse_args()

if __name__ == "__main__":
    # 设置中文字体支持
    setup_chinese_font()
    
    args = parse_args()
    
    # 如果是预测模式
    if args.predict:
        if not args.model:
            args.model = os.path.join(os.getcwd(), 'models', 'rf_temperature_predictor.joblib')
            if not os.path.exists(args.model):
                logging.error(f"未找到模型文件: {args.model}")
                logging.error("请先训练模型或指定正确的模型路径")
                exit(1)
        
        # 判断是否是目录
        if not os.path.isdir(args.predict):
            logging.error(f"输入路径不是目录: {args.predict}")
            logging.error("请提供一个包含PDB文件的目录")
            exit(1)
        
        # 处理目录
        result_dir = os.path.dirname(args.model)
        process_directory(args.predict, args.model, result_dir)
    else:
        # 训练模式
        try:
            # 加载数据
            data = load_data(args.data)
            
            # 准备特征
            X, y, feature_cols = prepare_features(data)
            logging.info(f"使用{len(feature_cols)}个特征进行训练")
            
            # 训练模型
            model, X_test, y_test, y_pred = train_model(X, y, args.output)
            
            logging.info("训练完成！模型可用于预测新的PDB文件的最适温度")
        except Exception as e:
            logging.error(f"训练过程中出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc()) 