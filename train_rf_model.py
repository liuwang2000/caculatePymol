#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import joblib
import argparse
import logging
import subprocess
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.pipeline import Pipeline
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

def prepare_features(data, apply_feature_selection=True):
    """准备特征数据和目标变量"""
    # 排除不作为特征的列
    exclude_columns = [
        'pdb_id', 'sequence', 'optimal_temperature', 
        'hydrogen_bonds', 'hydrophobic_contacts', 'salt_bridges', 'hydrophobic_sasa'
    ]
    
    # 准备特征和目标变量
    feature_cols = [col for col in data.columns if col not in exclude_columns]
    X = data[feature_cols]
    y = data['optimal_temperature']
    
    # 检查是否存在NaN值
    if X.isnull().any().any():
        logging.warning(f"特征中存在NaN值，将进行填充")
        X = X.fillna(X.mean())
    
    # 保存原始特征列名
    original_features = X.columns.tolist()
    
    # 执行特征选择
    if apply_feature_selection and len(X) > 20:  # 确保有足够的样本进行特征选择
        try:
            logging.info("开始进行特征选择...")
            # 初始化预选择模型
            pre_selector = RandomForestRegressor(n_estimators=100, random_state=42)
            pre_selector.fit(X, y)
            
            # 使用特征重要性进行初步选择
            sfm = SelectFromModel(pre_selector, threshold='mean')
            sfm.fit(X, y)
            selected_features_mask = sfm.get_support()
            selected_features = [f for f, selected in zip(original_features, selected_features_mask) if selected]
            
            if len(selected_features) >= 5:  # 确保至少保留5个特征
                logging.info(f"初步特征选择后保留{len(selected_features)}个特征")
                X = X[selected_features]
                feature_cols = selected_features
            else:
                logging.warning(f"特征选择后只剩{len(selected_features)}个特征，使用全部原始特征")
        except Exception as e:
            logging.error(f"特征选择过程出错: {str(e)}")
            logging.warning("将使用全部原始特征")
    
    return X, y, feature_cols

def cross_validate_model(X, y, n_splits=5):
    """交叉验证评估模型性能"""
    logging.info(f"执行{n_splits}折交叉验证...")
    
    # 创建评估模型
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 创建包含数据标准化的管道
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', base_model)
    ])
    
    # 执行交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 计算多个指标
    rmse_scores = -cross_val_score(model_pipeline, X, y, cv=kf, 
                                  scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(model_pipeline, X, y, cv=kf, 
                                 scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model_pipeline, X, y, cv=kf, 
                               scoring='r2')
    
    # 输出结果
    logging.info(f"交叉验证结果 - RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
    logging.info(f"交叉验证结果 - MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
    logging.info(f"交叉验证结果 - R²: {r2_scores.mean():.2f} ± {r2_scores.std():.2f}")
    
    return rmse_scores.mean(), mae_scores.mean(), r2_scores.mean()

def train_model(X, y, output_dir=None):
    """训练随机森林模型"""
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 执行数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建并训练基础随机森林模型
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    logging.info("基础随机森林模型训练完成")
    
    # 模型评估
    train_pred = rf_model.predict(X_train_scaled)
    test_pred = rf_model.predict(X_test_scaled)
    
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
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 0.3, 0.5]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    grid_search.fit(X_train_scaled, y_train)
    logging.info(f"最佳参数: {grid_search.best_params_}")
    
    # 使用最佳参数训练最终模型
    best_rf_model = RandomForestRegressor(
        **grid_search.best_params_,
        random_state=42
    )
    best_rf_model.fit(X_train_scaled, y_train)
    logging.info("优化后的随机森林模型训练完成")
    
    # 评估最终模型
    best_train_pred = best_rf_model.predict(X_train_scaled)
    best_test_pred = best_rf_model.predict(X_test_scaled)
    
    best_train_rmse = np.sqrt(mean_squared_error(y_train, best_train_pred))
    best_test_rmse = np.sqrt(mean_squared_error(y_test, best_test_pred))
    best_train_mae = mean_absolute_error(y_train, best_train_pred)
    best_test_mae = mean_absolute_error(y_test, best_test_pred)
    best_train_r2 = r2_score(y_train, best_train_pred)
    best_test_r2 = r2_score(y_test, best_test_pred)
    
    logging.info(f"优化后 - 训练集 RMSE: {best_train_rmse:.2f}, MAE: {best_train_mae:.2f}, R²: {best_train_r2:.2f}")
    logging.info(f"优化后 - 测试集 RMSE: {best_test_rmse:.2f}, MAE: {best_test_mae:.2f}, R²: {best_test_r2:.2f}")
    
    # 尝试梯度提升模型
    logging.info("尝试训练梯度提升模型作为对比...")
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    
    gb_test_pred = gb_model.predict(X_test_scaled)
    gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
    gb_test_r2 = r2_score(y_test, gb_test_pred)
    
    logging.info(f"梯度提升模型 - 测试集 RMSE: {gb_test_rmse:.2f}, R²: {gb_test_r2:.2f}")
    
    # 选择性能更好的模型
    final_model = best_rf_model
    final_test_rmse = best_test_rmse
    final_method = "随机森林"
    
    if gb_test_rmse < best_test_rmse:
        final_model = gb_model
        final_test_rmse = gb_test_rmse
        final_method = "梯度提升"
        logging.info(f"最终选择{final_method}模型 (RMSE: {final_test_rmse:.2f})")
    else:
        logging.info(f"最终选择{final_method}模型 (RMSE: {final_test_rmse:.2f})")
    
    # 保存模型和数据转换器
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'temperature_predictor.joblib')
    scaler_path = os.path.join(output_dir, 'feature_scaler.joblib')
    
    # 创建一个模型包，包含模型、特征名称和缩放器
    model_package = {
        'model': final_model,
        'feature_names': X.columns.tolist(),
        'scaler': scaler,
        'model_type': final_method
    }
    
    joblib.dump(model_package, model_path)
    logging.info(f"模型包已保存至: {model_path}")
    
    # 生成特征重要性图
    plot_feature_importance(final_model, X, output_dir)
    
    # 生成预测值与真实值对比图
    if final_method == "随机森林":
        plot_predictions(y_test, best_test_pred, output_dir)
    else:
        plot_predictions(y_test, gb_test_pred, output_dir)
    
    # 创建一个残差图
    plot_residuals(y_test, best_test_pred, output_dir)
    
    return final_model, X_test, y_test, best_test_pred

def plot_feature_importance(model, X, output_dir):
    """绘制特征重要性图"""
    # 判断模型类型，提取特征重要性
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        logging.warning("模型不支持直接提取特征重要性")
        return
    
    # 选择前20个特征或全部（如果少于20个）
    top_n = min(20, len(feature_importance))
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.title('模型 - 特征重要性')
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    logging.info(f"特征重要性图已保存至: {plot_path}")
    plt.close()

def plot_predictions(y_true, y_pred, output_dir):
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
    plot_path = os.path.join(output_dir, 'predictions_vs_actual.png')
    plt.savefig(plot_path)
    logging.info(f"预测对比图已保存至: {plot_path}")
    plt.close()

def plot_residuals(y_true, y_pred, output_dir):
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
    plot_path = os.path.join(output_dir, 'residuals.png')
    plt.savefig(plot_path)
    logging.info(f"残差分析图已保存至: {plot_path}")
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
        # 加载模型包
        model_package = joblib.load(model_path)
        logging.info(f"已加载模型包: {model_path}")
        
        model = model_package['model']
        scaler = model_package['scaler']
        feature_names = model_package['feature_names']
        model_type = model_package.get('model_type', '未知')
        
        logging.info(f"使用{model_type}模型进行预测")
        
        # 检查缺失的特征并添加
        for col in feature_names:
            if col not in pdb_data.columns:
                logging.warning(f"在输入数据中未找到特征: {col}，添加并设置为0")
                pdb_data[col] = 0.0
        
        # 确保只选择模型需要的特征列
        X = pdb_data[feature_names]
        
        # 检查并填充NaN值
        if X.isnull().any().any():
            logging.warning("特征中存在NaN值，将进行填充")
            X = X.fillna(X.mean())
        
        # 应用特征标准化
        X_scaled = scaler.transform(X)
        
        # 预测
        predictions = model.predict(X_scaled)
        
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
    parser = argparse.ArgumentParser(description='使用机器学习训练最适温度预测模型')
    parser.add_argument('--data', default='./trainData/analyze_pdb_merged_20250331_164045.csv',
                      help='训练数据文件路径')
    parser.add_argument('--output', default='./models',
                      help='模型输出目录')
    parser.add_argument('--predict', default=None,
                      help='预测模式：指定要预测的PDB文件或目录路径')
    parser.add_argument('--model', default=None,
                      help='预测模式：指定要使用的模型文件路径')
    parser.add_argument('--feature_selection', action='store_true',
                      help='是否启用特征选择')
    parser.add_argument('--no_normalization', action='store_true',
                      help='关闭特征标准化')
    return parser.parse_args()

if __name__ == "__main__":
    # 设置中文字体支持
    setup_chinese_font()
    
    args = parse_args()
    
    # 如果是预测模式
    if args.predict:
        if not args.model:
            args.model = os.path.join(os.getcwd(), 'models', 'temperature_predictor.joblib')
            if not os.path.exists(args.model):
                # 尝试查找旧格式的模型文件
                old_model = os.path.join(os.getcwd(), 'models', 'rf_temperature_predictor.joblib')
                if os.path.exists(old_model):
                    logging.warning(f"未找到新格式模型文件，使用旧格式模型: {old_model}")
                    args.model = old_model
                else:
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
            X, y, feature_cols = prepare_features(data, apply_feature_selection=args.feature_selection)
            logging.info(f"使用{len(feature_cols)}个特征进行训练")
            
            # 执行交叉验证评估
            cv_rmse, cv_mae, cv_r2 = cross_validate_model(X, y)
            
            # 训练模型
            model, X_test, y_test, y_pred = train_model(X, y, args.output)
            
            logging.info("训练完成！模型可用于预测新的PDB文件的最适温度")
            logging.info(f"交叉验证RMSE: {cv_rmse:.2f}, 测试集R²: {r2_score(y_test, y_pred):.2f}")
        except Exception as e:
            logging.error(f"训练过程中出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc()) 