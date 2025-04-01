#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import joblib
import argparse
import logging
import subprocess
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel, RFECV, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import pathlib
import time
import glob
import itertools
from imblearn.over_sampling import SMOTE
from datetime import datetime
import inspect
from scipy import stats
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from matplotlib.lines import Line2D

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)

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

def temperature_weighted_sampling(X, y, temp_threshold=70, weight_factor=3):
    """为高温样本增加权重，用于训练时平衡不同温度区间样本的影响
    
    参数:
        X: 特征矩阵
        y: 温度标签
        temp_threshold: 高温阈值，默认70°C
        weight_factor: 高温样本权重系数，默认3.0
        
    返回:
        样本权重数组
    """
    logging.info(f"应用温度加权采样，温度阈值={temp_threshold}°C，权重系数={weight_factor}")
    sample_weights = np.ones(len(y))
    high_temp_mask = y >= temp_threshold
    sample_weights[high_temp_mask] = weight_factor
    
    # 温度统计
    high_temp_count = sum(high_temp_mask)
    total_count = len(y)
    logging.info(f"高温样本占比: {high_temp_count}/{total_count} ({high_temp_count/total_count*100:.1f}%)")
    
    return sample_weights

def generate_synthetic_high_temp_samples(X, y, temp_threshold=60, n_samples=50):
    """生成合成的高温蛋白样本，用于增加高温区域的训练数据
    
    参数:
        X: 特征矩阵
        y: 温度标签
        temp_threshold: 高温阈值，默认60°C
        n_samples: 要生成的合成样本数量，默认50个
    """
    logging.info(f"生成高温合成样本，温度阈值={temp_threshold}°C，目标样本数={n_samples}")
    
    # 定位高温样本
    high_temp_indices = np.where(y >= temp_threshold)[0]
    high_temp_count = len(high_temp_indices)
    
    if high_temp_count < 3:
        logging.warning(f"高温样本数量({high_temp_count})不足，无法生成合成样本")
        return X, y
        
    logging.info(f"原始高温样本数量: {high_temp_count}")
    
    # 提取高温样本
    high_temp_X = X.iloc[high_temp_indices].copy()
    high_temp_y = y.iloc[high_temp_indices].copy()
    
    # 生成合成样本
    synthetic_X_list = []
    synthetic_y_list = []
    
    for _ in range(n_samples):
        # 随机选择两个高温样本
        idx1, idx2 = np.random.choice(high_temp_count, 2, replace=False)
        
        # 生成随机插值系数
        alpha = np.random.random()
        
        # 线性插值生成新样本
        new_X = high_temp_X.iloc[idx1] * alpha + high_temp_X.iloc[idx2] * (1-alpha)
        new_y = high_temp_y.iloc[idx1] * alpha + high_temp_y.iloc[idx2] * (1-alpha)
        
        # 加入微小扰动以增加多样性
        perturbation = np.random.normal(0, 0.01, size=len(new_X))
        new_X = new_X + pd.Series(perturbation, index=new_X.index)
        
        synthetic_X_list.append(new_X)
        synthetic_y_list.append(new_y)
    
    # 转换为DataFrame和Series
    synthetic_X = pd.DataFrame(synthetic_X_list, columns=X.columns)
    synthetic_y = pd.Series(synthetic_y_list)
    
    # 合并原始数据和合成数据
    X_combined = pd.concat([X, synthetic_X], ignore_index=True)
    y_combined = pd.concat([y, synthetic_y], ignore_index=True)
    
    logging.info(f"合成样本生成完成，最终高温样本数量: {sum(y_combined >= temp_threshold)}")
    logging.info(f"数据集扩充: {len(X)} → {len(X_combined)} 个样本")
    
    return X_combined, y_combined

def create_interaction_features(X, top_n=10):
    """创建特征交互项"""
    logging.info("创建特征交互项...")
    feature_names = X.columns.tolist()
    
    # 计算原始特征的重要性
    y = X[feature_names[0]].values  # 临时使用第一列作为目标变量
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    # 选择重要性排名靠前的特征
    important_indices = np.argsort(mi_scores)[-top_n:]
    important_features = [feature_names[i] for i in important_indices]
    
    # 创建交互特征
    interaction_features = pd.DataFrame(index=X.index)
    
    # 所有重要特征的两两组合
    for f1, f2 in itertools.combinations(important_features, 2):
        if f1 != f2:
            # 创建交互项
            interaction_name = f"{f1}_x_{f2}"
            interaction_features[interaction_name] = X[f1] * X[f2]
    
    # 只保留变化足够的交互特征
    final_interactions = pd.DataFrame(index=X.index)
    cols_added = 0
    
    for col in interaction_features.columns:
        if interaction_features[col].std() > 0:  # 确保特征有变化
            final_interactions[col] = interaction_features[col]
            cols_added += 1
            if cols_added >= 15:  # 限制添加的特征数量
                break
    
    logging.info(f"创建了{cols_added}个交互特征")
    return pd.concat([X, final_interactions], axis=1)

def apply_nonlinear_transform(X):
    """应用非线性特征转换，增加数据表达能力"""
    # 创建一个新的DataFrame来存储转换后的特征
    X_transformed = X.copy()
    
    # 选择数值型特征
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 对每个数值特征应用常见的非线性变换
    for feature in numeric_features:
        # 确保特征没有负值或零值(针对对数变换)
        if X[feature].min() > 0:
            # 对数变换 - 压缩大值
            X_transformed[f'{feature}_log'] = np.log1p(X[feature])
        
        # 平方变换 - 强调大值
        X_transformed[f'{feature}_squared'] = X[feature] ** 2
        
        # 立方根变换 - 处理有符号值
        X_transformed[f'{feature}_cbrt'] = np.cbrt(X[feature])
        
        # Sigmoid变换 - 将值压缩到0-1范围
        X_transformed[f'{feature}_sigmoid'] = 1 / (1 + np.exp(-X[feature]))
    
    # 只保留最有信息量的非线性特征
    from sklearn.feature_selection import VarianceThreshold
    
    try:
        # 使用方差阈值过滤掉低信息特征
        selector = VarianceThreshold(threshold=0.05)
        selector.fit(X_transformed)
        
        # 获取选择的特征
        selected_features = X_transformed.columns[selector.get_support()].tolist()
        logging.info(f"非线性变换后保留了{len(selected_features)}个特征，移除了{len(X_transformed.columns) - len(selected_features)}个低方差特征")
        
        return X_transformed[selected_features]
    except Exception as e:
        logging.warning(f"非线性特征筛选失败: {str(e)}，将返回所有转换特征")
    return X_transformed

def prepare_features(data, apply_feature_selection=True, use_interaction=False, 
                    use_nonlinear=False, use_polynomial=False, balance_temperatures=False,
                    custom_features=None):
    """准备特征数据，支持多种特征工程选项
    
    参数:
        data: 输入数据DataFrame
        apply_feature_selection: 是否应用特征选择
        use_interaction: 是否添加特征交互
        use_nonlinear: 是否添加非线性变换
        use_polynomial: 是否添加多项式特征
        balance_temperatures: 是否平衡温度样本
        custom_features: 自定义特征列表文件
        
    返回:
        (X, y, feature_cols)
    """
    logging.info("准备特征数据...")
    
    # 使用自定义特征列表（如果提供）
    if custom_features and os.path.exists(custom_features):
        try:
            with open(custom_features, 'r') as f:
                feature_cols = [line.strip() for line in f if line.strip()]
            logging.info(f"已加载{len(feature_cols)}个自定义特征")
        except Exception as e:
            logging.error(f"加载自定义特征失败: {str(e)}")
            feature_cols = None
    else:
        feature_cols = None
    
    # 获取所有可能的特征列
    if feature_cols is None:
        # 排除非特征列
        exclude_cols = ['pdb_id', 'optimal_temperature', 'Unnamed: 0', 'ec_number', 
                       'source_organism', 'protein_name', 'sequence']
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    # 创建特征矩阵和目标变量
    X = data[feature_cols].copy()
    y = data['optimal_temperature'].copy()
    
    # 处理缺失值
    logging.info("处理缺失值...")
    for col in X.columns:
        if X[col].isnull().any():
            # 对于缺失值较多的列，使用中位数填充
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                logging.info(f"  - 列 '{col}' 有 {missing_count} 个缺失值，使用中位数填充")
                X[col] = X[col].fillna(X[col].median())
    
    # 移除低方差特征
    if apply_feature_selection:
        logging.info("移除低方差特征...")
        initial_features = X.shape[1]
        selector = VarianceThreshold(threshold=0.01)
        X_selected = selector.fit_transform(X)
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
        
        # 更新DataFrame和特征列表
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        feature_cols = selected_features
        logging.info(f"  - 移除了 {initial_features - len(feature_cols)} 个低方差特征，剩余 {len(feature_cols)} 个特征")
    
    # 添加非线性变换
    if use_nonlinear:
        logging.info("添加非线性特征变换...")
        
        # 创建原始特征的副本
        original_features = X.columns.tolist()
        
        for col in original_features:
            # 只对数值型特征进行变换
            if pd.api.types.is_numeric_dtype(X[col]):
                # 添加对数变换 (对于正值特征)
                if X[col].min() > 0:
                    X[f'{col}_log'] = np.log1p(X[col])
                
                # 添加平方变换
                X[f'{col}_squared'] = X[col] ** 2
                
                # 添加立方根变换
                X[f'{col}_cbrt'] = np.cbrt(X[col])
                
                # 添加sigmoid变换
                X[f'{col}_sigmoid'] = 1 / (1 + np.exp(-X[col]))
        
        # 更新特征列表
        feature_cols = X.columns.tolist()
        logging.info(f"  - 添加非线性变换后的特征数量: {len(feature_cols)}")
    
    # 添加特征交互
    if use_interaction:
        logging.info("添加特征交互...")
        
        # 训练一个初步的随机森林以获取特征重要性
        rf_prelim = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_prelim.fit(X, y)
        
        # 获取前10个最重要的特征
        importances = rf_prelim.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # 取前10个
        top_features = [X.columns[i] for i in indices]
        
        logging.info(f"  - 选择了 {len(top_features)} 个重要特征用于交互")
        
        # 对这些特征做交互项
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                feat1 = top_features[i]
                feat2 = top_features[j]
                # 创建交互特征
                X[f'{feat1}_X_{feat2}'] = X[feat1] * X[feat2]
        
        # 更新特征列表
        feature_cols = X.columns.tolist()
        logging.info(f"  - 添加特征交互后的特征数量: {len(feature_cols)}")
    
    # 添加多项式特征
    if use_polynomial:
        logging.info("添加多项式特征...")
        
        # 首先训练一个随机森林来找出最重要的特征
        rf_prelim = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_prelim.fit(X, y)
        
        # 选择前15个最重要的特征
        importances = rf_prelim.feature_importances_
        indices = np.argsort(importances)[::-1][:5]  # 只取前5个，避免特征爆炸
        top_features = [X.columns[i] for i in indices]
        
        # 只对最重要的特征生成多项式
        X_top = X[top_features].copy()
        
        # 使用PolynomialFeatures生成2阶多项式特征
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = poly.fit_transform(X_top)
        
        # 获取特征名称
        poly_feature_names = []
        powers = poly.powers_
        for i, p in enumerate(powers):
            if i < len(top_features):  # 跳过原始特征
                continue
                
            # 创建特征名称
            name = ""
            for j, exp in enumerate(p):
                if exp > 0:
                    name += f"{top_features[j]}"
                    if exp > 1:
                        name += f"^{exp}"
                    name += "_"
            if name:
                name = name[:-1]  # 移除末尾的下划线
                poly_feature_names.append(name)
        
        # 将多项式特征添加到原始特征矩阵
        X_poly_df = pd.DataFrame(X_poly[:, len(top_features):], columns=poly_feature_names, index=X.index)
        X = pd.concat([X, X_poly_df], axis=1)
        
        # 更新特征列表
        feature_cols = X.columns.tolist()
        logging.info(f"  - 添加多项式特征后的特征数量: {len(feature_cols)}")
    
    # 应用特征选择（使用随机森林的特征重要性）
    if apply_feature_selection and len(feature_cols) > 20:  # 如果特征数量较多
        logging.info("基于特征重要性进行特征选择...")
        
        # 使用随机森林识别重要特征
        feature_selector = SelectFromModel(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            threshold='median'  # 使用中位数阈值，保留约一半的特征
        )
        
        # 拟合特征选择器
        feature_selector.fit(X, y)
        
        # 获取选中的特征名称
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if feature_selector.get_support()[i]]
        
        # 如果特征过多，则通过累积重要性法进一步筛选
        if len(selected_features) > 30:
            logging.info("  - 特征仍然较多，应用累积重要性筛选...")
            
            # 训练模型获取特征重要性
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            # 计算特征重要性
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # 按重要性排序的特征
            sorted_features = [feature_cols[i] for i in indices]
            sorted_importances = [importances[i] for i in indices]
            
            # 计算累积重要性
            cumulative_importance = np.cumsum(sorted_importances)
            
            # 选择重要性累积达到90%的特征
            importance_threshold = 0.90
            last_idx = np.where(cumulative_importance >= importance_threshold)[0][0]
            selected_features = sorted_features[:last_idx + 1]
            
            logging.info(f"  - 通过累积重要性(>{importance_threshold*100}%)选择了 {len(selected_features)} 个特征")
        
        # 应用特征筛选
        X = X[selected_features]
        feature_cols = selected_features
        logging.info(f"  - 特征选择后的特征数量: {len(feature_cols)}")
        
        # 显示选择的特征
        if len(feature_cols) <= 20:
            logging.info(f"  - 选中的特征: {', '.join(feature_cols)}")
    
    # 平衡温度样本（可选）
    if balance_temperatures:
        logging.info("平衡温度样本分布...")
        
        # 定义温度区间
        temp_bins = [0, 30, 45, 60, 75, 100]
        labels = [f"{temp_bins[i]}-{temp_bins[i+1]}" for i in range(len(temp_bins)-1)]
        
        # 将温度分到各个区间
        y_binned = pd.cut(y, bins=temp_bins, labels=labels)
        bin_counts = y_binned.value_counts()
        
        logging.info("  - 原始温度区间分布:")
        for bin_label, count in bin_counts.items():
            logging.info(f"    * {bin_label}°C: {count}个样本 ({count/len(y)*100:.1f}%)")
        
        # 计算权重以平衡样本
        max_count = bin_counts.max()
        sample_weights = np.ones(len(y))
        
        # 为每个温度区间分配权重
        for bin_label in labels:
            mask = y_binned == bin_label
            if mask.any():
                weight = max_count / bin_counts[bin_label]
                sample_weights[mask] = weight
        
        # 通过加权采样创建平衡数据集
        balanced_indices = np.random.choice(
            range(len(y)), 
            size=len(y), 
            replace=True, 
            p=sample_weights/sample_weights.sum()
        )
        
        X = X.iloc[balanced_indices].reset_index(drop=True)
        y = y.iloc[balanced_indices].reset_index(drop=True)
        
        # 检查平衡后的分布
        y_binned = pd.cut(y, bins=temp_bins, labels=labels)
        balanced_counts = y_binned.value_counts()
        
        logging.info("  - 平衡后温度区间分布:")
        for bin_label, count in balanced_counts.items():
            logging.info(f"    * {bin_label}°C: {count}个样本 ({count/len(y)*100:.1f}%)")
    
    logging.info(f"最终特征处理完成，使用 {len(feature_cols)} 个特征, {len(y)} 个样本")
    
    return X, y, feature_cols

def train_neural_network(X_train, y_train, X_test, y_test):
    """使用神经网络训练温度预测模型"""
    # 删除整个函数
    pass

def train_stacking_model(X_train, y_train, X_test, y_test):
    """使用堆叠模型训练温度预测模型"""
    # 删除整个函数
    pass

def cross_validate_model(X, y, n_splits=5):
    """交叉验证评估模型性能，使用嵌套交叉验证和更多评估指标"""
    logging.info(f"执行{n_splits}折交叉验证...")
    
    # 自定义评分器：温度预测的相对误差
    def temp_relative_error(y_true, y_pred):
        """计算预测温度的相对误差（百分比）"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 273.15))) * 100  # 返回相对误差百分比
    
    # 高温区域性能评估
    def high_temp_performance(y_true, y_pred, threshold=65.0):
        """评估模型在高温区域的性能"""
        high_temp_mask = y_true >= threshold
        if np.sum(high_temp_mask) > 0:
            high_temp_rmse = np.sqrt(mean_squared_error(y_true[high_temp_mask], y_pred[high_temp_mask]))
            return high_temp_rmse
        return 0.0
    
    # 创建自定义评分器
    from sklearn.metrics import make_scorer
    rel_error_scorer = make_scorer(temp_relative_error, greater_is_better=False)
    high_temp_scorer = make_scorer(high_temp_performance, greater_is_better=False)
    
    # 使用更多树提高评估精度
    base_model = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=42, n_jobs=-1)
    
    # 创建包含数据标准化的管道
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', base_model)
    ])
    
    # 实施嵌套交叉验证（外层用于评估，内层用于参数调优）
    logging.info("开始嵌套交叉验证（内部参数优化）...")
    
    # 参数网格
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 30],
        'model__min_samples_split': [2, 5],
        'model__max_features': ['sqrt', 0.33]
    }
    
    # 内层交叉验证
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # 创建嵌套CV评估器
    cv_model = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1,
        refit=True
    )
    
    # 外层交叉验证
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 收集所有指标
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    rel_error_scores = []
    high_temp_rmse_scores = []
    
    # 手动执行外层交叉验证，以便收集更多指标
    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 拟合模型（内部会通过GridSearchCV执行内层交叉验证）
        cv_model.fit(X_train, y_train)
        
        # 获取最佳参数
        best_params = cv_model.best_params_
        logging.info(f"内层CV最佳参数: {best_params}")
        
        # 预测
        y_pred = cv_model.predict(X_test)
        
        # 计算性能指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rel_error = temp_relative_error(y_test, y_pred)
        high_temp_rmse = high_temp_performance(y_test, y_pred)
        
        # 收集指标
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        rel_error_scores.append(rel_error)
        if high_temp_rmse > 0:
            high_temp_rmse_scores.append(high_temp_rmse)
    
    # 转换为numpy数组
    rmse_scores = np.array(rmse_scores)
    mae_scores = np.array(mae_scores)
    r2_scores = np.array(r2_scores)
    rel_error_scores = np.array(rel_error_scores)
    
    # 处理可能没有高温样本的情况
    if high_temp_rmse_scores:
        high_temp_rmse_scores = np.array(high_temp_rmse_scores)
        logging.info(f"高温区域RMSE: {high_temp_rmse_scores.mean():.2f} ± {high_temp_rmse_scores.std():.2f}")
    else:
        logging.info(f"测试集中没有高温样本，无法评估高温性能")
    
    # 输出详细的交叉验证结果
    logging.info(f"嵌套交叉验证结果:")
    logging.info(f"  RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
    logging.info(f"  MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
    logging.info(f"  R²: {r2_scores.mean():.2f} ± {r2_scores.std():.2f}")
    logging.info(f"  相对误差: {rel_error_scores.mean():.2f}% ± {rel_error_scores.std():.2f}%")
    
    # 执行最终参数优化并返回最佳参数
    final_cv = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    
    logging.info("执行最终参数优化...")
    final_cv.fit(X, y)
    
    logging.info(f"最佳参数组合: {final_cv.best_params_}")
    logging.info(f"最佳CV得分: {final_cv.best_score_:.4f}")
    
    # 提供推荐参数用于最终模型训练
    recommended_params = {
        'n_estimators': final_cv.best_params_.get('model__n_estimators', 200),
        'max_depth': final_cv.best_params_.get('model__max_depth', None),
        'min_samples_split': final_cv.best_params_.get('model__min_samples_split', 2),
        'max_features': final_cv.best_params_.get('model__max_features', 'sqrt')
    }
    
    logging.info(f"推荐的随机森林参数: {recommended_params}")
    
    return rmse_scores.mean(), mae_scores.mean(), r2_scores.mean(), recommended_params

def train_model(X, y, output_dir='./models', test_size=0.2, recommended_params=None, model_type='random_forest', high_temp_focus=False):
    """训练模型并返回测试集性能

    参数:
        X: 特征矩阵
        y: 温度标签
        output_dir: 模型输出目录
        test_size: 测试集比例
        recommended_params: 推荐参数(从交叉验证获得)，如为None则使用最优默认参数
        model_type: 模型类型 ('random_forest' 或 'gradient_boosting')
        high_temp_focus: 是否使用高温样本权重

    返回:
        (model, test_rmse, high_temp_results)
        model: 训练好的模型
        test_rmse: 测试集均方根误差
        high_temp_results: 高温样本评估结果
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 创建适当的样本权重（可选）
    if high_temp_focus:
        # 为高温样本分配更高权重
        temp_weights = np.ones(len(y_train))
        high_temp_indices = np.where(y_train >= 65.0)[0]
        temp_weights[high_temp_indices] = 3.0  # 为高温样本赋予3倍权重
        logging.info(f"应用高温样本权重: {len(high_temp_indices)}个样本分配了3倍权重")
    else:
        temp_weights = None
    
    # 如果没有提供推荐参数，使用最优默认参数
    if recommended_params is None:
        if model_type == 'random_forest':
            # 随机森林最优默认参数
            recommended_params = {
                'n_estimators': 300,
                'max_depth': 30,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True
            }
            logging.info("使用随机森林最优默认参数")
        else:  # gradient_boosting
            # 梯度提升树最优默认参数
            recommended_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'min_samples_split': 5,
                'learning_rate': 0.05,
                'subsample': 0.8
            }
            logging.info("使用梯度提升最优默认参数")
    
    # 选择模型类型
    if model_type == 'gradient_boosting':
        # 创建验证集用于早停
        X_train_part, X_val, y_train_part, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
            
        # 使用推荐的参数
        base_params = {
            'n_estimators': recommended_params.get('n_estimators', 500),  # 使用更多树
            'max_depth': recommended_params.get('max_depth', 5),
            'min_samples_split': recommended_params.get('min_samples_split', 2),
            'learning_rate': recommended_params.get('learning_rate', 0.05),  # 默认较低的学习率
            'subsample': recommended_params.get('subsample', 0.8),
            'random_state': 42
        }
        
        logging.info(f"训练梯度提升回归树模型，参数: {base_params}")
        
        # 创建模型
        model = GradientBoostingRegressor(**base_params)
        
        # 使用早停策略训练
        model.fit(
            X_train_part, y_train_part,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=20,  # 20轮无改善则早停
            verbose=False,
            sample_weight=temp_weights if temp_weights is not None else None
        )
        
        # 获取最佳迭代次数
        if hasattr(model, 'best_iteration_'):
            best_iteration = model.best_iteration_
            logging.info(f"早停在第{best_iteration}轮，使用最佳迭代进行最终训练")
            
            # 使用最佳迭代重新训练完整模型
            final_params = base_params.copy()
            final_params['n_estimators'] = best_iteration
            
            # 在完整训练集上训练最终模型
            model = GradientBoostingRegressor(**final_params)
            model.fit(X_train, y_train, sample_weight=temp_weights)
            logging.info(f"最终模型使用参数: {final_params}")
        else:
            # 如果早停未生效，在完整训练集上重新训练
            logging.info("早停未生效，使用完整训练集训练模型")
            model = GradientBoostingRegressor(**base_params)
            model.fit(X_train, y_train, sample_weight=temp_weights)
            
    else:
        # 随机森林模型
        rf_params = {
            'n_estimators': recommended_params.get('n_estimators', 300),
            'max_depth': recommended_params.get('max_depth', 30),
            'min_samples_split': recommended_params.get('min_samples_split', 5),
            'min_samples_leaf': recommended_params.get('min_samples_leaf', 2),
            'max_features': recommended_params.get('max_features', 'sqrt'),
            'bootstrap': recommended_params.get('bootstrap', True),
            'random_state': 42,
            'n_jobs': -1
        }
        logging.info(f"使用随机森林模型，参数: {rf_params}")
        
        model = RandomForestRegressor(**rf_params)
        
        # 训练模型
        logging.info("开始训练模型...")
        model.fit(X_train, y_train, sample_weight=temp_weights)
    
    # 预测并评估测试集
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    logging.info(f"测试集评估结果:")
    logging.info(f"  RMSE: {test_rmse:.2f}")
    logging.info(f"  MAE: {test_mae:.2f}")
    logging.info(f"  R²: {test_r2:.2f}")
    
    # 特别评估高温样本的性能（温度 >= 65°C）
    high_temp_mask = y_test >= 65.0
    if high_temp_mask.any():
        high_temp_y = y_test[high_temp_mask]
        high_temp_pred = y_pred[high_temp_mask]
        high_temp_rmse = np.sqrt(mean_squared_error(high_temp_y, high_temp_pred))
        high_temp_mae = mean_absolute_error(high_temp_y, high_temp_pred)
        high_temp_r2 = r2_score(high_temp_y, high_temp_pred)
        
        logging.info(f"高温样本评估结果 (>= 65°C, {high_temp_mask.sum()}个样本):")
        logging.info(f"  RMSE: {high_temp_rmse:.2f}")
        logging.info(f"  MAE: {high_temp_mae:.2f}")
        logging.info(f"  R²: {high_temp_r2:.2f}")
        
        high_temp_results = {
            'rmse': high_temp_rmse,
            'mae': high_temp_mae,
            'r2': high_temp_r2,
            'count': high_temp_mask.sum()
        }
    else:
        logging.warning("测试集中没有高温样本")
        high_temp_results = {}
    
    # 分析特征重要性
    if hasattr(model, 'feature_importances_'):
        feature_names = X.columns
        feature_importances = model.feature_importances_
        
        # 按重要性排序
        indices = np.argsort(feature_importances)[::-1]
        
        logging.info("特征重要性分析:")
        for i, idx in enumerate(indices[:15]):  # 只显示前15个特征
            logging.info(f"  {i+1}. {feature_names[idx]}: {feature_importances[idx]:.4f}")
        
        # 提取重要特征列表 (重要性 > 0.01)
        important_features = []
        for i, idx in enumerate(indices):
            if feature_importances[idx] > 0.01:
                important_features.append(feature_names[idx])
        
        logging.info(f"重要特征(>0.01)数量: {len(important_features)}")
    
    # 保存模型和特征名称
        model_package = {
            'model': model,
            'feature_names': X.columns.tolist(),
        'model_type': model_type,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': {
            'rmse': test_rmse,
            'r2': test_r2,
            'high_temp_results': high_temp_results
        }
    }
    
    # 保存模型包
    model_file = os.path.join(output_dir, f'temperature_predictor_{model_type}.joblib')
    joblib.dump(model_package, model_file)
    logging.info(f"模型已保存至: {model_file}")
    
    # 绘制预测vs实际图
    plt.figure(figsize=(8, 8))
    
    # 配置中文字体
    setup_chinese_font()
    
    # 高温和低温样本使用不同颜色
    high_temp_mask = y_test >= 65.0
    
    # 画出高温和低温样本
    if np.any(high_temp_mask):
        plt.scatter(y_test[high_temp_mask], y_pred[high_temp_mask], color='red', alpha=0.6, label='高温样本 (≥65°C)')
    
    if np.any(~high_temp_mask):
        plt.scatter(y_test[~high_temp_mask], y_pred[~high_temp_mask], color='blue', alpha=0.6, label='低温样本 (<65°C)')
    
    # 添加对角线（完美预测）
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='理想预测')
    
    plt.xlabel('实际温度 (°C)')
    plt.ylabel('预测温度 (°C)')
    plt.title('测试集: 预测温度 vs 实际温度')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 添加RMSE和R²文本
    plt.annotate(f'整体 RMSE: {test_rmse:.2f}\n整体 R²: {test_r2:.2f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # 如果有高温样本，添加高温性能
    if high_temp_mask.any():
        plt.annotate(f'高温 RMSE: {high_temp_rmse:.2f}\n高温 R²: {high_temp_r2:.2f}', 
                    xy=(0.05, 0.82), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", alpha=0.8))
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'prediction_performance_{model_type}.png'), dpi=150)
    plt.close()
        
    # 绘制误差分布图
    plt.figure(figsize=(10, 6))
    
    # 配置中文字体
    setup_chinese_font()
    
    errors = y_pred - y_test
    plt.hist(errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='零误差')
    
    # 计算误差统计
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    plt.title(f'预测误差分布 ({model_type})')
    plt.xlabel('预测误差 (°C)')
    plt.ylabel('频率')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加误差统计
    plt.annotate(f'平均误差: {mean_error:.2f}°C\n标准差: {std_error:.2f}°C', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'error_distribution_{model_type}.png'), dpi=150)
    plt.close()
    
    return model, test_rmse, high_temp_results

def plot_model_comparison(X_test, y_test, models, output_dir):
    """绘制不同模型的性能比较"""
    model_names = list(models.keys())
    rmse_values = [models[name][1] for name in model_names]
    r2_values = [models[name][2] for name in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE比较（越低越好）
    ax1.bar(model_names, rmse_values, color='salmon')
    ax1.set_title('各模型RMSE比较')
    ax1.set_ylabel('RMSE (°C)')
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # R2比较（越高越好）
    ax2.bar(model_names, r2_values, color='skyblue')
    ax2.set_title('各模型R²比较')
    ax2.set_ylabel('R²')
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    compare_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(compare_path)
    logging.info(f"模型比较图已保存至: {compare_path}")
    plt.close()

def plot_feature_importance(model, X, output_dir, model_type):
    """绘制特征重要性图"""
    # 配置中文字体
    setup_chinese_font()
    
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
    plt.title(f'{model_type.replace("_", " ").title()} - 特征重要性')
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, f'feature_importance_{model_type}.png')
    plt.savefig(plot_path)
    logging.info(f"特征重要性图已保存至: {plot_path}")
    plt.close()
    
    # 提供简化特征提取建议
    top_features = feature_importance.head(15)  # 获取前15个最重要的特征
    cumulative_importance = top_features['importance'].sum()
    
    logging.info("\n=== 特征提取简化建议 ===")
    logging.info(f"前15个特征的总重要性: {cumulative_importance:.2f} (占总重要性的{cumulative_importance*100:.1f}%)")
    
    # 按类别分组特征
    aa_features = [f for f in top_features['feature'] if f.startswith('aa_')]
    structure_features = [f for f in top_features['feature'] if f in ['helix', 'sheet', 'loop', 'helix_ratio', 'sheet_ratio', 'helix_sheet_ratio', 'compactness_index']]
    interaction_features = [f for f in top_features['feature'] if f in ['ion_pair_density', 'aromatic_interactions', 'hydrophobic_core_density', 'dense_hbond_network', 'surface_charge_ratio']]
    content_features = [f for f in top_features['feature'] if f in ['hydrophobic_content', 'polar_content', 'glycine_content', 'proline_content', 'ivywrel_index']]
    
    logging.info("\n关键特征类别:")
    if aa_features:
        logging.info(f"1. 氨基酸组成: {', '.join(aa_features)}")
    if structure_features:
        logging.info(f"2. 结构特征: {', '.join(structure_features)}")
    if interaction_features:
        logging.info(f"3. 相互作用特征: {', '.join(interaction_features)}")
    if content_features:
        logging.info(f"4. 内容比例特征: {', '.join(content_features)}")
    
    logging.info("\n建议简化特征提取函数仅计算以上关键特征，可减少计算开销并保持较高预测精度。")

def plot_predictions(y_true, y_pred, output_dir):
    """绘制预测vs实际图表"""
    # 配置中文字体
    setup_chinese_font()
    
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    
    # 添加理想线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 添加标题和轴标签
    plt.title('预测值 vs 真实值')
    plt.xlabel('真实温度 (°C)')
    plt.ylabel('预测温度 (°C)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 计算并显示性能指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    plt.text(min_val + 0.05*(max_val-min_val), 
             max_val - 0.15*(max_val-min_val),
             f'RMSE = {rmse:.2f}\nR² = {r2:.2f}',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存图片
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'prediction_scatter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plot_path)
    logging.info(f"预测对比图已保存至: {plot_path}")
    plt.close()

def plot_residuals(y_true, y_pred, output_dir):
    """绘制残差图"""
    # 配置中文字体
    setup_chinese_font()
    
    plt.figure(figsize=(10, 6))
    
    # 计算残差
    residuals = y_true - y_pred
    
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
        cmd = ['python', 'analyze_pdb.py', directory_path, '--thermostability']
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
        
        # 直接使用高温专精模型进行预测，无需额外的缩放
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
            # 检查是否为高温蛋白
            is_high_temp = row['predicted_temperature'] >= 70
            temp_indicator = "【高温】" if is_high_temp else "【低温】"
            logging.info(f"预测结果 - {row['pdb_id']}: {row['predicted_temperature']:.2f}°C {temp_indicator}")
        
        return True
    except Exception as e:
        logging.error(f"预测过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

def show_help():
    """显示详细的使用说明"""
    help_text = """
蛋白质温度预测模型 - 使用说明
===========================

本工具用于训练和预测蛋白质的最适温度，支持完整流程训练和单独预测。

【基本用法】

1. 训练模型 (使用现有数据):
   python train_rf_model.py --data trainData/your_data.csv --output ./models

2. 完整流程 (从PDB文件开始):
   python train_rf_model.py --full_pipeline --pdb_dir ./pdb_files --cazy_file ./cazy_data.csv --output ./models

3. 预测温度:
   python train_rf_model.py --predict ./pdb_files --model ./models/temperature_predictor_random_forest.joblib

【常用参数】

输入输出参数:
  --data              训练数据CSV文件路径
  --output            模型和结果的输出目录 (默认: ./models)
  --output_dir        预测结果输出目录 (默认: ./result/predictions)
  
训练参数:
  --test_size         测试集比例 (默认: 0.2)
  --model_type        模型类型: random_forest 或 gradient_boosting (默认: random_forest)
  --high_temp_focus   对高温样本应用3倍权重

特征工程参数:
  --interactions      使用特征交互
  --nonlinear         应用非线性特征变换
  --polynomial        添加多项式特征
  --balance           平衡训练数据中的温度区间
  --features          自定义特征列表文件路径

完整流程参数:
  --full_pipeline     执行完整流程: 特征提取->数据合并->模型训练
  --pdb_dir           PDB文件目录
  --cazy_file         CAZY数据文件路径

预测参数:
  --predict           要预测的PDB文件/目录路径
  --model             模型文件路径 (.joblib)
  --no-extract        预测时跳过特征提取 (使用现有的分析结果)

【示例】

1. 训练基本随机森林模型:
   python train_rf_model.py --data trainData/merged_data.csv

2. 训练增强梯度提升树模型:
   python train_rf_model.py --data trainData/merged_data.csv --model_type gradient_boosting --nonlinear --balance --high_temp_focus

3. 完整流程训练:
   python train_rf_model.py --full_pipeline --pdb_dir ./pdb_files --cazy_file ./cazy_data.csv

4. 预测单个PDB文件:
   python train_rf_model.py --predict ./new_proteins/protein.pdb --model ./models/temperature_predictor.joblib

5. 预测整个目录:
   python train_rf_model.py --predict ./new_proteins --model ./models/temperature_predictor.joblib

【输出说明】

训练结果:
- 模型文件 (.joblib)
- 性能评估图表 (预测vs实际, 误差分布)
- 特征重要性分析

预测结果:
- CSV文件 (包含PDB ID, 预测温度, 高温概率等)
- 温度预测可视化图表
- 高温概率分布图

【注意事项】

- 确保PDB文件格式正确
- 推荐使用高质量的CAZY数据获得更准确的模型
- 对于高度不平衡的数据集，建议使用--balance和--high_temp_focus参数
- 大数据集训练时，可以尝试--model_type gradient_boosting以获得更好的性能
"""
    print(help_text)

def parse_args():
    """解析命令行参数"""
    # 检查是否要显示帮助
    if '--help' in sys.argv or '-h' in sys.argv:
        show_help()
        
    parser = argparse.ArgumentParser(description='Protein temperature prediction model training')
    parser.add_argument('--data', default='./trainData/analyze_pdb_merged.csv',
                      help='Path to merged training data CSV file')
    parser.add_argument('--output', default='./result',
                      help='Output directory for models and evaluation results')
    parser.add_argument('--predict', default=None,
                      help='Prediction mode: specify PDB file or directory path')
    parser.add_argument('--model', default=None,
                      help='Prediction mode: specify model file path')
    # 添加模型类型参数
    parser.add_argument('--model_type', default='random_forest', choices=['random_forest', 'gradient_boosting'],
                      help='Model type to use: random_forest or gradient_boosting')
    # 简化选项，默认值设为True
    parser.add_argument('--interactions', action='store_true', default=True,
                      help='Use feature interactions')
    parser.add_argument('--high_temp_focus', action='store_true', default=True,
                      help='Apply high temperature sample weighting (3x)')
    parser.add_argument('--synthetic_samples', action='store_true',
                      help='Generate synthetic samples for high temperature range')
    parser.add_argument('--nonlinear', action='store_true', default=True,
                      help='Apply nonlinear feature transformations')
    parser.add_argument('--polynomial', action='store_true', default=True,
                      help='Add polynomial features')
    parser.add_argument('--balance', action='store_true', default=True,
                      help='Balance temperature ranges in training data')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Test set size, default is 0.2 (20%%)')
    parser.add_argument('--full_pipeline', action='store_true',
                      help='Run full pipeline: feature extraction -> data merge -> model training')
    parser.add_argument('--pdb_dir', 
                      help='PDB files directory for full pipeline training')
    parser.add_argument('--cazy_file',
                      help='CAZY data file path for full pipeline training')
    parser.add_argument('--features',
                      help='Path to custom feature list')
    # 预测选项
    parser.add_argument('--optimized', action='store_true',
                      help='Use optimized feature extraction (only for prediction mode)')
    parser.add_argument('--no-extract', action='store_true',
                      help='Skip feature extraction when predicting')
    parser.add_argument('--output_dir', default=None,
                      help='Optional custom directory for prediction results (default: ./result/predictions)')
    return parser.parse_args()

def evaluate_high_temperature_performance(y_true, y_pred, high_temp_threshold=65):
    """专门评估高温区域的预测性能 - 优化版
    
    参数:
        y_true: 真实温度值
        y_pred: 预测温度值
        high_temp_threshold: 高温阈值，默认65°C（调整后）
    """
    # 筛选高温数据
    high_temp_mask = y_true >= high_temp_threshold
    high_temp_count = sum(high_temp_mask)
    
    if high_temp_count == 0:
        logging.warning(f"没有高温样本(≥{high_temp_threshold}°C)用于评估")
        return {
            'count': 0,
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan'),
            'rel_error': float('nan')
        }
    
    # 提取高温区域的真实值和预测值
    high_temp_true = y_true[high_temp_mask]
    high_temp_pred = y_pred[high_temp_mask]
    
    # 计算性能指标
    high_temp_rmse = np.sqrt(mean_squared_error(high_temp_true, high_temp_pred))
    high_temp_mae = mean_absolute_error(high_temp_true, high_temp_pred)
    
    # 对于只有一个样本的情况，R²无法计算
    if high_temp_count > 1:
        high_temp_r2 = r2_score(high_temp_true, high_temp_pred)
    else:
        high_temp_r2 = float('nan')
    
    # 相对误差 (%)
    high_temp_rel_error = np.mean(np.abs((high_temp_true - high_temp_pred) / (high_temp_true))) * 100
    
    # 增加更详细的误差分析
    error_stats = {
        'mean_error': np.mean(high_temp_pred - high_temp_true),
        'median_error': np.median(high_temp_pred - high_temp_true),
        'error_std': np.std(high_temp_pred - high_temp_true),
        'max_error': np.max(np.abs(high_temp_pred - high_temp_true)),
        'q90_error': np.percentile(np.abs(high_temp_pred - high_temp_true), 90)
    }
    
    logging.info(f"高温区域(≥{high_temp_threshold}°C)评估结果:")
    logging.info(f"  高温样本数量: {high_temp_count}")
    logging.info(f"  高温平均真实值: {np.mean(high_temp_true):.2f}°C")
    logging.info(f"  高温平均预测值: {np.mean(high_temp_pred):.2f}°C")
    logging.info(f"  高温RMSE: {high_temp_rmse:.2f}")
    logging.info(f"  高温MAE: {high_temp_mae:.2f}")
    logging.info(f"  高温R²: {high_temp_r2:.2f}")
    logging.info(f"  高温相对误差: {high_temp_rel_error:.2f}%")
    logging.info(f"  高温平均误差: {error_stats['mean_error']:.2f}°C")
    logging.info(f"  高温中位数误差: {error_stats['median_error']:.2f}°C")
    logging.info(f"  高温误差标准差: {error_stats['error_std']:.2f}°C")
    logging.info(f"  高温最大误差: {error_stats['max_error']:.2f}°C")
    logging.info(f"  高温90%分位误差: {error_stats['q90_error']:.2f}°C")
    
    # 超高温区域评估 (>80°C)
    very_high_temp_mask = y_true >= 80
    if sum(very_high_temp_mask) > 0:
        very_high_temp_true = y_true[very_high_temp_mask]
        very_high_temp_pred = y_pred[very_high_temp_mask]
        very_high_temp_rmse = np.sqrt(mean_squared_error(very_high_temp_true, very_high_temp_pred))
        very_high_temp_mae = mean_absolute_error(very_high_temp_true, very_high_temp_pred)
        
        logging.info(f"  超高温区域(≥80°C)样本数量: {sum(very_high_temp_mask)}")
        logging.info(f"  超高温RMSE: {very_high_temp_rmse:.2f}")
        logging.info(f"  超高温MAE: {very_high_temp_mae:.2f}")
    
    return {
        'count': high_temp_count,
        'rmse': high_temp_rmse,
        'mae': high_temp_mae,
        'r2': high_temp_r2,
        'rel_error': high_temp_rel_error,
        'error_stats': error_stats,
        'very_high_temp_count': sum(very_high_temp_mask) if 'very_high_temp_mask' in locals() else 0,
        'very_high_temp_rmse': very_high_temp_rmse if 'very_high_temp_rmse' in locals() else float('nan'),
        'very_high_temp_mae': very_high_temp_mae if 'very_high_temp_mae' in locals() else float('nan')
    }

def plot_high_temp_performance(y_true, y_pred, output_dir, threshold=65):
    """生成高温区域性能评估图表
    
    参数:
        y_true: 真实温度值
        y_pred: 预测温度值
        output_dir: 输出目录
        threshold: 高温阈值，默认65°C
    """
    # 配置中文字体
    setup_chinese_font()
    
    plt.figure(figsize=(12, 8))
    
    # 划分温度区间
    low_mask = y_true < threshold
    high_mask = y_true >= threshold
    
    # 绘制低温区间预测
    plt.scatter(y_true[low_mask], y_pred[low_mask], 
                alpha=0.6, c='blue', label=f'<{threshold}°C')
    
    # 突出显示高温区间预测
    plt.scatter(y_true[high_mask], y_pred[high_mask], 
                alpha=0.8, c='red', s=80, label=f'≥{threshold}°C')
    
    # 添加理想预测线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    # 计算并显示高温区域指标
    if sum(high_mask) > 0:
        high_rmse = np.sqrt(mean_squared_error(y_true[high_mask], y_pred[high_mask]))
        high_r2 = r2_score(y_true[high_mask], y_pred[high_mask]) if sum(high_mask) > 2 else float('nan')
        
        # 修复格式化字符串错误
        r2_text = f"{high_r2:.2f}" if not np.isnan(high_r2) else "N/A"
        
        plt.text(min_val + 0.05*(max_val-min_val), 
                max_val - 0.1*(max_val-min_val),
                f'高温区域 (≥{threshold}°C):\nRMSE = {high_rmse:.2f}\nR² = {r2_text}',
                bbox=dict(facecolor='red', alpha=0.2))
    
    plt.xlabel('真实最适温度 (°C)')
    plt.ylabel('预测最适温度 (°C)')
    plt.title('高温区域预测性能评估')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图片
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'high_temp_performance.png')
    plt.savefig(plot_path)
    logging.info(f"高温区域性能图已保存至: {plot_path}")
    plt.close()

def full_training_pipeline(pdb_dir, cazy_file=None, model_output_dir='./models', high_temp_focus=False, 
                          test_size=0.2, model_type='random_forest', use_nonlinear=False, 
                          use_interactions=False, use_polynomial=False, balance_temperatures=False):
    """完整训练流程: 特征提取->合并数据->训练模型
    
    参数:
        pdb_dir: PDB文件目录
        cazy_file: CAZY数据文件（可选）
        model_output_dir: 模型输出目录
        high_temp_focus: 使用高温专家模型
        test_size: 测试集比例
        model_type: 模型类型，'random_forest'或'gradient_boosting'
        use_nonlinear: 是否使用非线性特征
        use_interactions: 是否使用特征交互
        use_polynomial: 是否使用多项式特征
        balance_temperatures: 是否平衡温度区间样本
    
    返回:
        是否成功
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保输出目录存在
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 步骤1: 特征提取
        logging.info("第1步: 从PDB文件提取特征")
        cmd = ['python', 'analyze_pdb.py', pdb_dir, '--thermostability']
        logging.info(f"执行命令: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        if proc.returncode != 0:
            logging.error(f"特征提取失败: {proc.stderr}")
            return False
            
        # 查找最新的分析结果文件
        output_dir = 'output'  # analyze_pdb.py的默认输出目录
        analysis_files = []
        for file in os.listdir(output_dir):
            if file.startswith('analyze_pdb_') and file.endswith('.csv'):
                file_path = os.path.join(output_dir, file)
                analysis_files.append((file_path, os.path.getmtime(file_path)))
        
        if not analysis_files:
            logging.error("未找到分析结果文件")
            return False
        
        # 获取最新的文件
        analysis_files.sort(key=lambda x: x[1], reverse=True)
        latest_analysis_file = analysis_files[0][0]
        logging.info(f"使用最新分析结果: {latest_analysis_file}")
        
        # 步骤2: 合并CAZY数据（如果提供）
        if cazy_file and os.path.exists(cazy_file):
            logging.info("第2步: 合并CAZY温度数据")
            
            # 执行数据合并脚本，使用正确的参数格式
            merge_cmd = ['python', 'merge_data.py', 
                          '--analyze', latest_analysis_file,
                          '--cazy', cazy_file]
            logging.info(f"执行命令: {' '.join(merge_cmd)}")
            merge_proc = subprocess.run(merge_cmd, capture_output=True, text=True)
            
            if merge_proc.returncode != 0:
                logging.error(f"数据合并失败: {merge_proc.stderr}")
                return False
            
            # 查找合并后的数据文件
            merged_files = []
            for file in os.listdir('trainData'):
                if file.startswith('analyze_pdb_merged') and file.endswith('.csv'):
                    file_path = os.path.join('trainData', file)
                    merged_files.append((file_path, os.path.getmtime(file_path)))
            
            if not merged_files:
                logging.error("未找到合并后的数据文件")
                return False
            
            # 获取最新的合并文件
            merged_files.sort(key=lambda x: x[1], reverse=True)
            latest_merged_file = merged_files[0][0]
            logging.info(f"使用最新合并数据: {latest_merged_file}")
            
            # 步骤3: 训练模型
            logging.info("第3步: 训练温度预测模型")
            data = load_data(latest_merged_file)
        else:
            # 如果没有CAZY数据，直接用分析结果并添加虚拟温度
            logging.warning("未提供CAZY数据，使用随机温度进行测试训练")
            
            # 读取分析结果
            analysis_data = pd.read_csv(latest_analysis_file)
            
            # 添加随机温度（仅用于测试）
            np.random.seed(42)
            analysis_data['optimal_temperature'] = np.random.uniform(20, 100, size=len(analysis_data))
            
            # 保存为合并数据格式
            os.makedirs('trainData', exist_ok=True)
            mock_data_file = os.path.join('trainData', f'mock_data_{timestamp}.csv')
            analysis_data.to_csv(mock_data_file, index=False)
            
            logging.info(f"创建模拟数据用于测试: {mock_data_file}")
            data = analysis_data
        
        # 数据分析
        temp_threshold = 65.0
        high_temp_count = sum(data['optimal_temperature'] >= temp_threshold)
        total_count = len(data)
        
        logging.info(f"数据集温度分布统计:")
        logging.info(f"  总样本数: {total_count}")
        logging.info(f"  数据集中高温样本(≥{temp_threshold}°C): {high_temp_count} ({high_temp_count/total_count*100:.1f}%)")
        
        # 准备特征 - 增强版本
        X, y, feature_cols = prepare_features(
            data, 
            apply_feature_selection=True,
            use_interaction=use_interactions,
            use_nonlinear=use_nonlinear,
            use_polynomial=use_polynomial,
            balance_temperatures=balance_temperatures
        )
        
        # 直接使用最优参数，不需要交叉验证搜索参数
        logging.info("使用最优默认参数训练模型...")
        
        # 训练最终模型 - 这里传入None将使用train_model内部的最优默认参数
        model, test_rmse, high_temp_results = train_model(
            X, y, 
            output_dir=model_output_dir,
            test_size=test_size,
            recommended_params=None,  # 将使用函数内的最优默认参数
            model_type=model_type,
            high_temp_focus=high_temp_focus
        )
        
        # 输出完整评估报告
        logging.info("\n" + "="*50)
        logging.info("训练完成! 模型评估报告:")
        logging.info(f"  数据集大小: {len(X)} 样本")
        logging.info(f"  特征数量: {len(feature_cols)} 个")
        logging.info(f"  模型类型: {model_type}")
        logging.info(f"  测试集RMSE: {test_rmse:.2f}")
        logging.info(f"  高温区域RMSE: {high_temp_results.get('rmse', 0.0):.2f}")
        logging.info(f"  高温区域R²: {high_temp_results.get('r2', 0.0):.2f}")
        logging.info(f"  模型保存路径: {os.path.join(model_output_dir, f'temperature_predictor_{model_type}.joblib')}")
        logging.info("="*50)
        
        return True
        
    except Exception as e:
        logging.error(f"模型训练过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main(args):
    """标准训练流程"""
    try:
        # 配置中文字体
        setup_chinese_font()
        
        # 如果是预测模式
        if args.predict and args.model:
            return predict_temperature(
                args.predict, 
                args.model, 
                args=args,
                output_dir=args.output_dir,
                extract_features=not args.no_extract
            )
        
        # 如果是全流程训练模式
        if args.full_pipeline:
            if not args.pdb_dir:
                logging.error("必须指定PDB文件目录 (--pdb_dir)")
                return False
            
            return full_training_pipeline(
                pdb_dir=args.pdb_dir,
                cazy_file=args.cazy_file,
                model_output_dir=args.output,
                test_size=args.test_size,
                model_type=args.model_type,
                high_temp_focus=args.high_temp_focus,
                use_nonlinear=args.nonlinear,
                use_interactions=args.interactions,
                use_polynomial=args.polynomial,
                balance_temperatures=args.balance
            )
        
        # 常规训练模式
        data = load_data(args.data)
        
        # 输出温度分布统计信息
        temp_threshold = 60.0
        high_temp_count = sum(data['optimal_temperature'] >= temp_threshold)
        total_count = len(data)
        logging.info(f"数据集温度分布统计:")
        logging.info(f"  总样本数: {total_count}")
        logging.info(f"  高温样本数(≥{temp_threshold}°C): {high_temp_count} ({high_temp_count/total_count*100:.1f}%)")
        logging.info(f"  中低温样本数(<{temp_threshold}°C): {total_count - high_temp_count} ({(total_count - high_temp_count)/total_count*100:.1f}%)")
        
        # 准备特征 - 增强版本
        X, y, feature_cols = prepare_features(
            data, 
            apply_feature_selection=True,
            use_interaction=args.interactions,
            use_nonlinear=args.nonlinear,
            use_polynomial=args.polynomial,
            balance_temperatures=args.balance
        )
        
        # 不再需要交叉验证，直接使用最优默认参数
        logging.info("使用最优默认参数训练模型...")
        
        # 直接训练最终模型
        logging.info("准备开始训练，评估数据集...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
        logging.info(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
        
        model, rmse, high_temp_results = train_model(
            X, y, 
            output_dir=args.output,
            test_size=args.test_size,
            recommended_params=None,  # 将使用函数内的最优默认参数
            model_type=args.model_type,
            high_temp_focus=args.high_temp_focus
        )
        
        logging.info(f"模型训练完成，测试集RMSE: {rmse:.2f}")
        logging.info(f"模型和性能指标已保存至 {args.output} 目录")
        
        return True
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def predict_temperature(pdb_dir, model_path, args=None, output_dir=None, extract_features=True):
    """预测PDB文件的最适温度
    
    参数:
        pdb_dir: PDB文件目录或单个PDB文件路径
        model_path: 模型文件路径
        args: 其他参数
        output_dir: 输出目录，默认为result/predictions
        extract_features: 是否需要提取特征，如果为False则直接使用现有的分析结果文件
    
    返回:
        处理是否成功
    """
    if output_dir is None:
        output_dir = os.path.join('result', 'predictions')
        
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳，用于文件命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.info(f"处理目录/文件: {pdb_dir}")
    
    # 检查输入路径是否存在
    if not os.path.exists(pdb_dir):
        logging.error(f"输入路径不存在: {pdb_dir}")
        return False
    
    # 确定是目录还是单个文件
    if os.path.isdir(pdb_dir):
        # 如果需要提取特征，则运行analyze_pdb.py
        if extract_features:
            cmd = ['python', 'analyze_pdb.py', pdb_dir, '--thermostability']
            logging.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"分析PDB文件失败: {result.stderr}")
                return False
                
            # 等待确保文件已经写入磁盘
            logging.info("等待文件系统更新...")
            time.sleep(1)
            
            # 查找最新的分析结果文件
            analysis_files = []
            for file in os.listdir('output'):
                if file.startswith('analyze_pdb_') and file.endswith('.csv'):
                    file_path = os.path.join('output', file)
                    analysis_files.append((file_path, os.path.getmtime(file_path)))
            
            if not analysis_files:
                logging.error("未找到分析结果文件")
                return False
            
            # 获取最新的文件
            analysis_files.sort(key=lambda x: x[1], reverse=True)
            latest_analysis_file = analysis_files[0][0]
            logging.info(f"使用分析文件: {os.path.basename(latest_analysis_file)}")
            
            # 加载分析结果
            try:
                pdb_data = pd.read_csv(latest_analysis_file)
                logging.info(f"成功加载分析结果，包含 {len(pdb_data)} 条记录")
            except Exception as e:
                logging.error(f"无法加载分析文件: {str(e)}")
                return False
        else:
            # 查找最新的分析结果文件
            analysis_files = []
            for file in os.listdir('output'):
                if file.startswith('analyze_pdb_') and file.endswith('.csv'):
                    file_path = os.path.join('output', file)
                    analysis_files.append((file_path, os.path.getmtime(file_path)))
            
            if not analysis_files:
                logging.error("未找到分析结果文件，请先运行特征提取")
                return False
            
            # 获取最新的文件
            analysis_files.sort(key=lambda x: x[1], reverse=True)
            latest_analysis_file = analysis_files[0][0]
            logging.info(f"使用现有分析文件: {os.path.basename(latest_analysis_file)}")
            
            # 加载分析结果
            try:
                pdb_data = pd.read_csv(latest_analysis_file)
                logging.info(f"成功加载分析结果，包含 {len(pdb_data)} 条记录")
            except Exception as e:
                logging.error(f"无法加载分析文件: {str(e)}")
                return False
    else:
        # 处理单个PDB文件
        if extract_features:
            cmd = ['python', 'analyze_pdb.py', pdb_dir, '--thermostability']
            logging.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"分析PDB文件失败: {result.stderr}")
                return False
                
            # 等待确保文件已经写入磁盘
            logging.info("等待文件系统更新...")
            time.sleep(1)
            
            # 查找最新的分析结果文件
            analysis_files = []
            for file in os.listdir('output'):
                if file.startswith('analyze_pdb_') and file.endswith('.csv'):
                    file_path = os.path.join('output', file)
                    analysis_files.append((file_path, os.path.getmtime(file_path)))
            
            if not analysis_files:
                logging.error("未找到分析结果文件")
                return False
            
            # 获取最新的文件
            analysis_files.sort(key=lambda x: x[1], reverse=True)
            latest_analysis_file = analysis_files[0][0]
            
            # 加载分析结果
            try:
                pdb_data = pd.read_csv(latest_analysis_file)
            except Exception as e:
                logging.error(f"无法加载分析文件: {str(e)}")
                return False
        else:
            logging.error("单个PDB文件必须进行特征提取")
            return False
    
    # 预测温度
    try:
        # 加载模型包
        model_package = joblib.load(model_path)
        logging.info(f"已加载模型包: {model_path}")
        
        model = model_package['model']
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
            X = X.fillna(X.median())
        
        # 直接使用模型进行预测
        temperatures = model.predict(X)
        
        # 添加预测结果
        pdb_data['predicted_temperature'] = temperatures
        pdb_data['is_high_temp'] = temperatures >= 65.0
        
        # 尝试获取高温概率（如果是使用RandomForestClassifier/RandomForestRegressor）
        has_probs = False
        probabilities = []
        
        # 对于随机森林，我们可以估计高温概率
        if hasattr(model, 'predict_proba') or (hasattr(model, 'estimators_') and model_type == 'random_forest'):
            try:
                # 对于分类器，可以直接获取概率
                if hasattr(model, 'predict_proba'):
                    # 对于分类器，获取高温类的概率
                    class_probs = model.predict_proba(X)
                    # 假设最后一个类是高温类
                    high_temp_probs = class_probs[:, -1]
                    probabilities = high_temp_probs
                else:
                    # 对于随机森林回归器，使用树的预测分布估计概率
                    tree_preds = []
                    for tree in model.estimators_:
                        tree_preds.append(tree.predict(X))
                    
                    # 计算每个样本的预测温度分布
                    tree_preds = np.column_stack(tree_preds)
                    
                    # 估计每个样本是高温的概率（树预测温度超过65°C的比例）
                    high_temp_probs = np.mean(tree_preds >= 65.0, axis=1)
                    probabilities = high_temp_probs
                
                # 添加到数据框
                pdb_data['high_temp_probability'] = probabilities
                has_probs = True
                logging.info("成功获取高温概率预测")
            except Exception as e:
                logging.warning(f"无法计算高温概率: {str(e)}")
        
        # 重排列，将pdb_id和预测温度放在前面
        if has_probs:
            columns = ['pdb_id', 'predicted_temperature', 'is_high_temp', 'high_temp_probability'] + [col for col in pdb_data.columns 
                                                             if col not in ['pdb_id', 'predicted_temperature', 'is_high_temp', 'high_temp_probability']]
        else:
            columns = ['pdb_id', 'predicted_temperature', 'is_high_temp'] + [col for col in pdb_data.columns 
                                                        if col not in ['pdb_id', 'predicted_temperature', 'is_high_temp']]
        
        pdb_data = pdb_data[columns]
        
        # 保存结果
        result_file = os.path.join(output_dir, f'prediction_results_{timestamp}.csv')
        pdb_data.to_csv(result_file, index=False)
        logging.info(f"预测结果已保存至: {result_file}")
        
        # 打印每个PDB文件的预测结果
        for index, row in pdb_data.iterrows():
            # 检查是否为高温蛋白
            is_high_temp = row['is_high_temp']
            temp_indicator = "【高温】" if is_high_temp else "【低温】"
            
            # 添加概率信息（如果有）
            if has_probs:
                prob_str = f"概率: {row['high_temp_probability']:.2f}"
            else:
                prob_str = ""
                
            logging.info(f"预测结果 - {row['pdb_id']}: {row['predicted_temperature']:.2f}°C {temp_indicator} {prob_str}")
        
        # 可视化预测结果
        logging.info("生成预测结果可视化...")
        try:
            # 配置中文字体
            setup_chinese_font()
            
            # 获取PDB ID和温度数据
            pdb_ids = pdb_data['pdb_id'].values
            temperatures = pdb_data['predicted_temperature'].values
            
            # 按温度排序
            sorted_indices = np.argsort(temperatures)
            sorted_ids = [pdb_ids[i] for i in sorted_indices]
            sorted_temps = [temperatures[i] for i in sorted_indices]
            
            # 使用不同颜色标记高温和低温
            colors = ['red' if temp >= 65 else 'blue' for temp in sorted_temps]
            
            # 创建坐标轴
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # 绘制温度条形图
            bars = ax.bar(sorted_ids, sorted_temps, color=colors, alpha=0.7)
            
            # 在条形图上方添加温度数值
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{sorted_temps[i]:.1f}°C',
                      ha='center', va='bottom', rotation=0, fontsize=8)
            
            # 添加高温和分类阈值线
            ax.axhline(y=70, color='r', linestyle='--', alpha=0.7)  # 高温阈值
            ax.axhline(y=65, color='orange', linestyle='--', alpha=0.7)  # 分类阈值
            
            # 添加颜色图例
            if has_probs:
                # 创建颜色渐变映射
                sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('高温概率')
                ax.legend(['高温阈值 (70°C)', '分类阈值 (65°C)'], loc='upper right')
            else:
                # 创建高温和低温的模拟图例
                from matplotlib.patches import Patch
                legend_elements = [
                    Line2D([0], [0], color='r', linestyle='--', lw=2, label='高温阈值 (70°C)'),
                    Line2D([0], [0], color='orange', linestyle='--', lw=2, label='温度分类阈值 (65°C)'),
                    Patch(facecolor='red', alpha=0.7, label='高温蛋白'),
                    Patch(facecolor='blue', alpha=0.7, label='低温蛋白')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
            
            # 旋转x轴标签以防止重叠
            plt.xticks(rotation=45, ha='right')
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 设置标题和标签
            ax.set_title('蛋白质温度预测结果')
            ax.set_xlabel('PDB ID')
            ax.set_ylabel('预测最适温度 (°C)')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表到指定目录
            output_file = os.path.join(output_dir, f'temperature_prediction_results_{timestamp}.png')
            plt.savefig(output_file, dpi=150)
            logging.info(f"预测结果图表已保存至: {output_file}")
            
            # 绘制第二个图表：高温概率分布（如果有概率数据）
            if has_probs:
                plt.figure(figsize=(14, 6))
                
                # 排序PDB ID和概率
                sorted_indices = np.argsort(probabilities)[::-1]  # 按概率从高到低排序
                sorted_ids = [pdb_ids[i] for i in sorted_indices]
                sorted_probs = [probabilities[i] for i in sorted_indices]
                sorted_temps = [temperatures[i] for i in sorted_indices]
                
                # 创建双y轴图
                fig, ax1 = plt.subplots(figsize=(14, 6))
                
                # 绘制概率条形图
                bars1 = ax1.bar(sorted_ids, sorted_probs, color='lightblue', alpha=0.7)
                ax1.set_xlabel('PDB ID（按高温概率排序）')
                ax1.set_ylabel('高温概率', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.set_ylim(0, 1.0)
                
                # 添加阈值线
                ax1.axhline(y=0.4, color='purple', linestyle='--', alpha=0.7, label='概率阈值 (0.4)')
                
                # 创建第二个y轴
                ax2 = ax1.twinx()
                line = ax2.plot(sorted_ids, sorted_temps, 'ro-', alpha=0.7, label='预测温度')
                
                # 在温度线上添加温度数值
                for i, (x, y) in enumerate(zip(sorted_ids, sorted_temps)):
                    # 每隔几个点显示一个温度值，防止过于拥挤
                    if i % 2 == 0 or i == len(sorted_ids) - 1:
                        ax2.text(x, y + 2, f'{y:.1f}°C', ha='center', va='bottom', 
                                color='red', fontsize=8, rotation=45)
                        
                ax2.set_ylabel('预测温度 (°C)', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # 添加图例
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper right')
                
                # 设置标题
                plt.title('蛋白质高温概率分布')
                
                # 旋转x轴标签
                plt.xticks(rotation=45, ha='right')
                
                # 添加网格线
                ax1.grid(True, linestyle='--', alpha=0.3)
                
                # 调整布局
                plt.tight_layout()
                
                # 保存图表
                prob_file = os.path.join(output_dir, f'high_temp_probability_{timestamp}.png')
                plt.savefig(prob_file, dpi=150)
                logging.info(f"高温概率分布图已保存至: {prob_file}")
            
            plt.close('all')
        except Exception as e:
            logging.error(f"生成可视化图表失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
        
        return True
    except Exception as e:
        logging.error(f"预测过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Protein temperature prediction model training')
    
    # 输入输出参数
    parser.add_argument('--data', type=str, help='Path to merged training data CSV file')
    parser.add_argument('--output', type=str, default='./models', help='Output directory for models and results')
    parser.add_argument('--output_dir', type=str, help='Optional custom directory for prediction results (default: ./result/predictions)')
    
    # 训练参数
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size, default is 0.2 (20%%)')
    parser.add_argument('--model_type', type=str, default='random_forest', choices=['random_forest', 'gradient_boosting'], 
                        help='Model type to use: random_forest or gradient_boosting')
    parser.add_argument('--high_temp_focus', action='store_true', help='Apply high temperature sample weighting (3x)')
    
    # 特征工程参数
    parser.add_argument('--interactions', action='store_true', help='Use feature interactions')
    parser.add_argument('--nonlinear', action='store_true', help='Apply nonlinear feature transformations')
    parser.add_argument('--polynomial', action='store_true', help='Add polynomial features')
    parser.add_argument('--balance', action='store_true', help='Balance temperature ranges in training data')
    parser.add_argument('--features', type=str, help='Path to custom feature list')
    
    # 完整流程参数
    parser.add_argument('--full_pipeline', action='store_true', help='Run full pipeline: feature extraction -> data merge -> model training')
    parser.add_argument('--pdb_dir', type=str, help='PDB files directory for full pipeline training')
    parser.add_argument('--cazy_file', type=str, help='Path to CAZY data file')
    
    # 预测模式参数
    parser.add_argument('--predict', type=str, help='Path to PDB file/directory for prediction')
    parser.add_argument('--model', type=str, help='Path to model file (.joblib)')
    parser.add_argument('--no-extract', action='store_true', help='Skip feature extraction when predicting')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('train_model.log', mode='a')
        ]
    )
    
    # 显示帮助信息
    if len(sys.argv) == 1:
        show_help()
        sys.exit(0)
        
    # 执行主函数
    success = main(args)
    sys.exit(0 if success else 1)