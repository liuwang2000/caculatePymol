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
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SequentialFeatureSelector, RFE, RFECV, mutual_info_regression
from sklearn.inspection import permutation_importance
from itertools import combinations
from tqdm import tqdm
import time
import datetime

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
        # 设置seaborn样式
        sns.set(font='sans-serif')
        # 尝试使用Agg后端，可能在某些情况下有帮助
        plt.switch_backend('Agg')
    else:
        # 设置seaborn中文字体
        sns.set(font=plt.rcParams['font.sans-serif'][0])
        logging.info("成功配置matplotlib和seaborn中文字体支持")
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

def prepare_features(data, create_interactions=False, create_polynomials=False, degree=2, include_aa=False):
    """准备特征数据和目标变量"""
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
    
    # 初始特征集
    original_features = X.columns.tolist()
    
    # 创建多项式特征
    if create_polynomials:
        logging.info(f"创建{degree}次多项式特征")
        # 选择数值型特征的前10个最重要特征
        num_features = min(10, len(original_features))
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = [original_features[i] for i in indices[:num_features]]
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X[top_features])
        feature_names = poly.get_feature_names_out(top_features)
        
        # 将多项式特征添加到原始特征中
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        # 只保留交互项，不要原始列
        X_poly_df = X_poly_df.iloc[:, num_features:]
        
        X = pd.concat([X, X_poly_df], axis=1)
        logging.info(f"创建了{X_poly_df.shape[1]}个多项式特征，总特征数量：{X.shape[1]}")
    
    # 创建特征交互项
    if create_interactions:
        logging.info("创建特征交互项")
        # 使用互信息找出与目标相关性高的特征
        mi_scores = mutual_info_regression(X[original_features], y, random_state=42)
        mi_indices = np.argsort(mi_scores)[::-1]
        top_mi_features = [original_features[i] for i in mi_indices[:min(10, len(original_features))]]
        
        # 创建交互特征
        interaction_features = pd.DataFrame(index=X.index)
        for i, f1 in enumerate(top_mi_features):
            for f2 in top_mi_features[i+1:]:
                # 创建乘积特征
                name = f"{f1}_x_{f2}"
                interaction_features[name] = X[f1] * X[f2]
                
                # 创建比率特征（确保除数不为0）
                if (X[f2] != 0).all():
                    ratio_name = f"{f1}_div_{f2}"
                    interaction_features[ratio_name] = X[f1] / (X[f2] + 1e-10)
        
        X = pd.concat([X, interaction_features], axis=1)
        logging.info(f"创建了{interaction_features.shape[1]}个交互特征，总特征数量：{X.shape[1]}")
    
    return X, y, list(X.columns)

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return r2, rmse, mae, y_pred

def forward_feature_selection(X, y, max_features=15, cv=5, random_state=42):
    """前向特征选择"""
    # 确保max_features不超过特征数量
    n_features = X.shape[1]
    if max_features >= n_features:
        logging.warning(f"特征总数({n_features})小于要选择的特征数({max_features})，调整为{n_features-1}")
        max_features = n_features - 1
    
    logging.info(f"执行前向特征选择，最大特征数量: {max_features}")
    
    # 创建模型
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # 获取CPU核心数(留一个核心给系统)
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"使用{n_jobs}个CPU核心进行并行处理")
    
    # 创建前向特征选择器
    sfs = SequentialFeatureSelector(
        model, 
        n_features_to_select=max_features,
        direction='forward',
        scoring='r2',
        cv=cv,
        n_jobs=n_jobs  # 使用多线程
    )
    
    # 训练特征选择器
    sfs.fit(X, y)
    
    # 获取选择的特征
    selected_features = X.columns[sfs.get_support()]
    
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=0.2, random_state=random_state
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练完整模型
    full_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    full_model.fit(X_train_scaled, y_train)
    
    # 计算训练集R^2
    y_train_pred = full_model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 计算测试集R^2
    y_test_pred = full_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 评估交叉验证性能
    cv_scores = cross_val_score(model, X[selected_features], y, cv=cv, scoring='r2')
    mean_cv_score = np.mean(cv_scores)
    
    logging.info(f"前向特征选择完成，选择了{len(selected_features)}个特征，训练集R²: {train_r2:.4f}, 交叉验证R²: {mean_cv_score:.4f}")
    
    return list(selected_features), mean_cv_score, train_r2, test_r2, full_model, scaler

def backward_feature_elimination(X, y, min_features=3, cv=5, random_state=42):
    """后向特征消除"""
    # 确保min_features合法
    n_features = X.shape[1]
    if min_features >= n_features:
        logging.warning(f"特征总数({n_features})小于或等于要保留的特征数({min_features})，调整为{max(1, n_features-1)}")
        min_features = max(1, n_features-1)
    
    logging.info(f"执行后向特征消除，最小特征数量: {min_features}")
    
    # 创建模型
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # 获取CPU核心数(留一个核心给系统)
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"使用{n_jobs}个CPU核心进行并行处理")
    
    # 创建后向特征消除器
    sfs = SequentialFeatureSelector(
        model, 
        n_features_to_select=min_features,
        direction='backward',
        scoring='r2',
        cv=cv,
        n_jobs=n_jobs  # 使用多线程
    )
    
    # 训练特征选择器
    sfs.fit(X, y)
    
    # 获取选择的特征
    selected_features = X.columns[sfs.get_support()]
    
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=0.2, random_state=random_state
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练完整模型
    full_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    full_model.fit(X_train_scaled, y_train)
    
    # 计算训练集R^2
    y_train_pred = full_model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 计算测试集R^2
    y_test_pred = full_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 评估交叉验证性能
    cv_scores = cross_val_score(model, X[selected_features], y, cv=cv, scoring='r2')
    mean_cv_score = np.mean(cv_scores)
    
    logging.info(f"后向特征消除完成，选择了{len(selected_features)}个特征，训练集R²: {train_r2:.4f}, 交叉验证R²: {mean_cv_score:.4f}")
    
    return list(selected_features), mean_cv_score, train_r2, test_r2, full_model, scaler

def recursive_feature_elimination_cv(X, y, min_features=3, max_features=15, cv=5, random_state=42):
    """递归特征消除（带交叉验证）"""
    # 确保特征数范围合法
    n_features = X.shape[1]
    if min_features >= n_features:
        logging.warning(f"特征总数({n_features})小于要保留的最小特征数({min_features})，调整为{max(1, n_features-1)}")
        min_features = max(1, n_features-1)
    
    if max_features >= n_features:
        logging.warning(f"特征总数({n_features})小于要选择的最大特征数({max_features})，调整为{n_features-1}")
        max_features = n_features - 1
    
    logging.info(f"执行递归特征消除（交叉验证），特征范围: {min_features}-{max_features}")
    
    # 创建模型
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # 获取CPU核心数(留一个核心给系统)
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"使用{n_jobs}个CPU核心进行并行处理")
    
    # 创建递归特征消除器
    rfecv = RFECV(
        estimator=model,
        min_features_to_select=min_features,
        step=1,
        cv=cv,
        scoring='r2',
        n_jobs=n_jobs  # 使用多线程
    )
    
    # 训练特征选择器
    rfecv.fit(X, y)
    
    # 获取选择的特征
    selected_features = X.columns[rfecv.support_]
    
    # 限制最大特征数量
    if len(selected_features) > max_features:
        # 使用RFE选择指定数量的特征
        rfe = RFE(estimator=model, n_features_to_select=max_features)
        rfe.fit(X[selected_features], y)
        selected_features = selected_features[rfe.support_]
    
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=0.2, random_state=random_state
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练完整模型
    full_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    full_model.fit(X_train_scaled, y_train)
    
    # 计算训练集R^2
    y_train_pred = full_model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 计算测试集R^2
    y_test_pred = full_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 评估交叉验证性能
    cv_scores = cross_val_score(model, X[selected_features], y, cv=cv, scoring='r2', n_jobs=n_jobs)
    mean_cv_score = np.mean(cv_scores)
    
    logging.info(f"递归特征消除完成，选择了{len(selected_features)}个特征，训练集R²: {train_r2:.4f}, 交叉验证R²: {mean_cv_score:.4f}")
    
    return list(selected_features), mean_cv_score, train_r2, test_r2, full_model, scaler

def permutation_importance_selection(X, y, n_features=15, cv=5, random_state=42):
    """基于排列重要性的特征选择"""
    # 确保n_features不超过特征总数
    n_total_features = X.shape[1]
    if n_features >= n_total_features:
        logging.warning(f"特征总数({n_total_features})小于要选择的特征数({n_features})，调整为{n_total_features}")
        n_features = n_total_features
    
    logging.info(f"执行基于排列重要性的特征选择，目标特征数量: {n_features}")
    
    # 创建模型
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # 获取CPU核心数(留一个核心给系统)
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"使用{n_jobs}个CPU核心进行并行处理")
    
    # 训练模型
    model.fit(X, y)
    
    # 计算排列重要性
    perm_importance = permutation_importance(
        model, X, y, 
        n_repeats=10, 
        random_state=random_state,
        n_jobs=n_jobs  # 使用多线程
    )
    
    # 按重要性排序
    indices = np.argsort(perm_importance.importances_mean)[::-1]
    
    # 选择前n个特征
    selected_features = X.columns[indices[:n_features]]
    
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=0.2, random_state=random_state
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练完整模型
    full_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    full_model.fit(X_train_scaled, y_train)
    
    # 计算训练集R^2
    y_train_pred = full_model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 计算测试集R^2
    y_test_pred = full_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 评估交叉验证性能
    cv_scores = cross_val_score(model, X[selected_features], y, cv=cv, scoring='r2')
    mean_cv_score = np.mean(cv_scores)
    
    logging.info(f"排列重要性特征选择完成，选择了{len(selected_features)}个特征，训练集R²: {train_r2:.4f}, 交叉验证R²: {mean_cv_score:.4f}")
    
    return list(selected_features), mean_cv_score, train_r2, test_r2, full_model, scaler

def mutual_info_selection(X, y, n_features=15, cv=5, random_state=42):
    """基于互信息的特征选择"""
    # 确保n_features不超过特征总数
    n_total_features = X.shape[1]
    if n_features >= n_total_features:
        logging.warning(f"特征总数({n_total_features})小于要选择的特征数({n_features})，调整为{n_total_features}")
        n_features = n_total_features
    
    logging.info(f"执行基于互信息的特征选择，目标特征数量: {n_features}")
    
    # 计算互信息
    mi_scores = mutual_info_regression(X, y, random_state=random_state)
    
    # 按重要性排序
    indices = np.argsort(mi_scores)[::-1]
    
    # 选择前n个特征
    selected_features = X.columns[indices[:n_features]]
    
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=0.2, random_state=random_state
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练完整模型
    full_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    full_model.fit(X_train_scaled, y_train)
    
    # 计算训练集R^2
    y_train_pred = full_model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 计算测试集R^2
    y_test_pred = full_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 创建模型
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # 评估交叉验证性能
    cv_scores = cross_val_score(model, X[selected_features], y, cv=cv, scoring='r2')
    mean_cv_score = np.mean(cv_scores)
    
    logging.info(f"互信息特征选择完成，选择了{len(selected_features)}个特征，训练集R²: {train_r2:.4f}, 交叉验证R²: {mean_cv_score:.4f}")
    
    return list(selected_features), mean_cv_score, train_r2, test_r2, full_model, scaler

def find_best_feature_combination(X, y, feature_candidates, max_combinations=10, cv=5, random_state=42):
    """找到最佳特征组合"""
    logging.info(f"尝试寻找最佳特征组合，候选特征数量: {len(feature_candidates)}")
    
    # 合并所有候选特征并去重
    all_features = set()
    for features, _ in feature_candidates:
        all_features.update(features)
    all_features = list(all_features)
    
    logging.info(f"候选特征总数: {len(all_features)}")
    
    # 按照交叉验证性能对特征集进行排序
    feature_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 最佳组合和性能
    best_features = feature_candidates[0][0]
    best_score = feature_candidates[0][1]
    
    # 尝试组合顶部的特征集
    n_candidates = min(5, len(feature_candidates))  # 最多使用前5个特征集
    
    logging.info(f"尝试组合前{n_candidates}个特征集...")
    
    # 创建模型
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # 创建所有可能的组合
    combinations_tried = 0
    for i in range(2, n_candidates + 1):
        for combo in combinations(range(n_candidates), i):
            # 合并特征集
            feature_set = set()
            for idx in combo:
                feature_set.update(feature_candidates[idx][0])
            feature_set = list(feature_set)
            
            # 如果特征集大小合理
            if len(feature_set) <= max_combinations:
                # 评估性能
                cv_scores = cross_val_score(model, X[feature_set], y, cv=cv, scoring='r2')
                mean_cv_score = np.mean(cv_scores)
                
                combinations_tried += 1
                
                # 更新最佳特征集
                if mean_cv_score > best_score:
                    best_features = feature_set
                    best_score = mean_cv_score
                    logging.info(f"找到更好的组合! 特征数: {len(best_features)}, 交叉验证R²: {best_score:.4f}")
            
            # 限制尝试的组合数量
            if combinations_tried >= 30:
                break
        
        if combinations_tried >= 30:
            break
    
    logging.info(f"最佳特征组合: {len(best_features)}个特征, 交叉验证R²: {best_score:.4f}")
    
    return best_features, best_score

def train_final_model(X, y, features, test_size=0.2, random_state=42):
    """训练最终模型"""
    logging.info(f"使用{len(features)}个特征训练最终模型")
    
    # 准备数据
    X_subset = X[features]
    
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=test_size, random_state=random_state
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练随机森林模型
    rf_model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # 评估随机森林性能
    rf_r2, rf_rmse, rf_mae, rf_y_pred = evaluate_model(rf_model, X_test_scaled, y_test)
    
    # 训练梯度提升模型
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state
    )
    gb_model.fit(X_train_scaled, y_train)
    
    # 评估梯度提升性能
    gb_r2, gb_rmse, gb_mae, gb_y_pred = evaluate_model(gb_model, X_test_scaled, y_test)
    
    # 选择性能更好的模型
    if gb_r2 > rf_r2:
        best_model = gb_model
        model_name = "GradientBoosting"
        r2, rmse, mae, y_pred = gb_r2, gb_rmse, gb_mae, gb_y_pred
        logging.info(f"梯度提升模型性能更好: R² = {r2:.4f}")
    else:
        best_model = rf_model
        model_name = "RandomForest"
        r2, rmse, mae, y_pred = rf_r2, rf_rmse, rf_mae, rf_y_pred
        logging.info(f"随机森林模型性能更好: R² = {r2:.4f}")
    
    logging.info(f"最终模型 ({model_name}) - 测试集 R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return {
        'model': best_model,
        'model_name': model_name,
        'scaler': scaler,
        'features': features,
        'test_r2': r2,
        'test_rmse': rmse,
        'test_mae': mae,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

def save_results(result, output_dir):
    """保存模型和结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_file = os.path.join(output_dir, f"best_model_{result['model_name']}_{timestamp}.pkl")
    joblib.dump({
        'model': result['model'],
        'scaler': result['scaler'],
        'features': result['features'],
        'model_name': result['model_name'],
        'train_r2': result['train_r2'],
        'test_r2': result['test_r2'],
        'cv_r2': result['cv_r2'],
        'method': result['method']
    }, model_file)
    logging.info(f"模型已保存至 {model_file}")
    
    # 保存特征信息
    feature_file = os.path.join(output_dir, f"selected_features_{timestamp}.csv")
    pd.DataFrame({'feature': result['features']}).to_csv(feature_file, index=False)
    logging.info(f"特征列表已保存至 {feature_file}")
    
    # 保存结果摘要
    summary_file = os.path.join(output_dir, f"results_summary_{timestamp}.csv")
    summary_df = pd.DataFrame({
        'method': [result['method']],
        'num_features': [len(result['features'])],
        'train_r2': [result['train_r2']],
        'test_r2': [result['test_r2']],
        'cv_r2': [result['cv_r2']],
        'features': [','.join(result['features'])]
    })
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"结果摘要已保存至 {summary_file}")
    
    # 绘制实际值与预测值的散点图
    plot_predictions(result, output_dir, timestamp)
    
    # 绘制特征重要性
    plot_feature_importance(result, output_dir, timestamp)
    
    return model_file

def plot_predictions(result, output_dir, timestamp):
    """绘制预测值与实际值对比图"""
    # 提取所需数据
    model = result['model']
    scaler = result['scaler']
    features = result['features']
    
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        result['X'][features], result['y'], 
        test_size=0.2, random_state=42
    )
    
    # 标准化数据
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 获取预测值
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 训练集预测对比
    ax1.scatter(y_train, y_train_pred, alpha=0.6, color='blue')
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax1.set_xlabel('实际值', fontsize=14)
    ax1.set_ylabel('预测值', fontsize=14)
    ax1.set_title(f'训练集 (R² = {result["train_r2"]:.4f})', fontsize=16)
    
    # 测试集预测对比
    ax2.scatter(y_test, y_test_pred, alpha=0.6, color='red')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax2.set_xlabel('实际值', fontsize=14)
    ax2.set_ylabel('预测值', fontsize=14)
    ax2.set_title(f'测试集 (R² = {result["test_r2"]:.4f})', fontsize=16)
    
    plt.suptitle(f'预测值与实际值对比 - {result["method"]} - {len(features)}个特征', fontsize=18)
    
    # 保存图像
    plot_file = os.path.join(output_dir, f"predictions_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"预测对比图已保存至 {plot_file}")

def plot_feature_importance(result, output_dir, timestamp):
    """绘制特征重要性图"""
    # 提取所需数据
    model = result['model']
    features = result['features']
    
    # 获取特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]
    
    # 根据特征数量调整图表大小
    plt.figure(figsize=(12, 8))
    
    # 绘制水平条形图（便于显示长特征名）
    y_pos = np.arange(len(sorted_features))
    plt.barh(y_pos, sorted_importances, align='center')
    plt.yticks(y_pos, sorted_features)
    plt.xlabel('重要性分数', fontsize=14)
    plt.title(f'特征重要性 - {result["method"]}', fontsize=16)
    
    # 保存图像
    plot_file = os.path.join(output_dir, f"feature_importance_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"特征重要性图已保存至 {plot_file}")

def feature_correlation_matrix(X, features, output_dir, timestamp):
    """绘制特征相关性矩阵"""
    # 计算相关性矩阵
    corr_matrix = X[features].corr()
    
    # 根据特征数量调整图表大小
    plt.figure(figsize=(14, 12))
    
    # 设置热图
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, 1)] = True  # 仅显示下三角
    
    # 绘制热图
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True,   # 显示数值
        fmt=".2f",    # 数值格式化
        cmap='coolwarm',  # 使用蓝红渐变色
        vmin=-1, vmax=1,  # 相关性范围
        linewidths=0.5,
        annot_kws={"size": 8}  # 调整注释大小
    )
    
    plt.title('特征相关性矩阵', fontsize=16)
    plt.tight_layout()
    
    # 保存图像
    plot_file = os.path.join(output_dir, f"feature_correlation_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"特征相关性矩阵已保存至 {plot_file}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='高级特征选择和模型训练，寻找最优特征组合提高R²。')
    parser.add_argument('--data', type=str, required=True, help='训练数据文件路径 (CSV格式)')
    parser.add_argument('--output', type=str, default='./models_advanced', help='输出目录，用于保存模型和结果')
    parser.add_argument('--interactions', action='store_true', help='创建特征交互项')
    parser.add_argument('--polynomials', action='store_true', help='创建多项式特征')
    parser.add_argument('--degree', type=int, default=2, help='多项式特征次数')
    parser.add_argument('--min_features', type=int, default=5, help='最小特征数量')
    parser.add_argument('--max_features', type=int, default=20, help='最大特征数量')
    parser.add_argument('--cv', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--include_aa', action='store_true', help='包含氨基酸比例特征')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    start_time = time.time()
    logging.info("开始高级特征选择优化...")
    logging.info(f"预计需要几分钟到几十分钟时间，取决于特征数量和CPU核心数")
    
    # 加载数据
    data = load_data(args.data)
    
    # 准备特征
    X, y, feature_cols = prepare_features(
        data, 
        create_interactions=args.interactions,
        create_polynomials=args.polynomials,
        degree=args.degree,
        include_aa=args.include_aa
    )
    logging.info(f"特征处理后的特征数量: {len(feature_cols)}")
    
    total_features = len(feature_cols)
    if total_features > 30:
        estimated_time = (total_features * 2) // 60  # 粗略估计，每个特征约2秒
        logging.info(f"特征较多，预计完成时间约为{estimated_time}分钟以上，请耐心等待...")
    
    # 检查特征数量是否足够
    if len(feature_cols) <= 1:
        logging.error(f"可用特征数量({len(feature_cols)})太少，无法进行特征选择。请考虑包含更多特征或启用氨基酸比例特征。")
        return
    
    # 进度信息
    logging.info("将依次执行5种特征选择方法：前向特征选择、后向特征消除、递归特征消除、排列重要性和互信息特征选择")
    
    # 运行不同的特征选择方法
    feature_candidates = []
    
    # 前向特征选择
    features_fwd, score_fwd, train_r2_fwd, test_r2_fwd, model_fwd, scaler_fwd = forward_feature_selection(
        X, y, max_features=args.max_features, cv=args.cv, random_state=args.random_state
    )
    feature_candidates.append({
        'features': features_fwd, 
        'cv_r2': score_fwd,
        'train_r2': train_r2_fwd,
        'test_r2': test_r2_fwd,
        'model': model_fwd,
        'scaler': scaler_fwd,
        'method': '前向特征选择'
    })
    
    # 后向特征消除
    features_bwd, score_bwd, train_r2_bwd, test_r2_bwd, model_bwd, scaler_bwd = backward_feature_elimination(
        X, y, min_features=args.min_features, cv=args.cv, random_state=args.random_state
    )
    feature_candidates.append({
        'features': features_bwd, 
        'cv_r2': score_bwd,
        'train_r2': train_r2_bwd,
        'test_r2': test_r2_bwd,
        'model': model_bwd,
        'scaler': scaler_bwd,
        'method': '后向特征消除'
    })
    
    # 递归特征消除
    features_rfe, score_rfe, train_r2_rfe, test_r2_rfe, model_rfe, scaler_rfe = recursive_feature_elimination_cv(
        X, y, min_features=args.min_features, max_features=args.max_features, 
        cv=args.cv, random_state=args.random_state
    )
    feature_candidates.append({
        'features': features_rfe, 
        'cv_r2': score_rfe,
        'train_r2': train_r2_rfe,
        'test_r2': test_r2_rfe,
        'model': model_rfe,
        'scaler': scaler_rfe,
        'method': '递归特征消除'
    })
    
    # 排列重要性
    features_perm, score_perm, train_r2_perm, test_r2_perm, model_perm, scaler_perm = permutation_importance_selection(
        X, y, n_features=args.max_features, cv=args.cv, random_state=args.random_state
    )
    feature_candidates.append({
        'features': features_perm, 
        'cv_r2': score_perm,
        'train_r2': train_r2_perm,
        'test_r2': test_r2_perm,
        'model': model_perm,
        'scaler': scaler_perm,
        'method': '排列重要性'
    })
    
    # 互信息特征选择
    features_mi, score_mi, train_r2_mi, test_r2_mi, model_mi, scaler_mi = mutual_info_selection(
        X, y, n_features=args.max_features, cv=args.cv, random_state=args.random_state
    )
    feature_candidates.append({
        'features': features_mi, 
        'cv_r2': score_mi,
        'train_r2': train_r2_mi,
        'test_r2': test_r2_mi,
        'model': model_mi,
        'scaler': scaler_mi,
        'method': '互信息特征选择'
    })
    
    # 按测试集R^2排序
    feature_candidates.sort(key=lambda x: x['test_r2'], reverse=True)
    
    # 找到测试集R^2最高的模型
    best_candidate = feature_candidates[0]
    logging.info(f"最佳特征选择方法: {best_candidate['method']}, 测试集R²: {best_candidate['test_r2']:.4f}, 训练集R²: {best_candidate['train_r2']:.4f}")
    
    # 使用测试集R^2最高的特征集找到最佳特征组合
    best_features = best_candidate['features']
    
    # 保存最终模型结果（使用最佳特征集已训练的模型）
    result = {
        'model': best_candidate['model'],
        'model_name': 'RandomForest',
        'scaler': best_candidate['scaler'],
        'features': best_features,
        'train_r2': best_candidate['train_r2'],
        'test_r2': best_candidate['test_r2'],
        'cv_r2': best_candidate['cv_r2'],
        'method': best_candidate['method'],
        'X': X,
        'y': y
    }
    
    # 保存结果
    model_file = save_results(result, args.output)
    
    # 绘制特征相关性矩阵
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_correlation_matrix(X, best_features, args.output, timestamp)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_str = format_time(elapsed_time)
    logging.info(f"高级特征选择优化完成，总耗时: {elapsed_str}")
    logging.info(f"最佳特征组合 ({len(best_features)}个特征) 的测试集 R²: {best_candidate['test_r2']:.4f}")
    logging.info(f"模型已保存至: {model_file}")

if __name__ == "__main__":
    main() 