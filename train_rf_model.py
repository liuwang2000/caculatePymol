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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
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

# 堆叠模型实现
class StackingRegressor(BaseEstimator, RegressorMixin):
    """自定义堆叠回归器"""
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.base_models_ = None
        self.meta_model_ = None
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
    def fit(self, X, y):
        self.base_models_ = [list() for _ in range(len(self.base_models))]
        
        # 存储每个模型的预测结果的OOF (Out-of-Fold)
        oof_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        # 训练基础模型
        for i, model in enumerate(self.base_models):
            logging.info(f"训练基础模型 {i+1}/{len(self.base_models)}: {type(model).__name__}")
            for train_idx, valid_idx in self.kfold.split(X):
                X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
                y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]
                
                # 训练模型
                model_fold = model.fit(X_train_fold, y_train_fold)
                self.base_models_[i].append(model_fold)
                
                # 存储预测结果
                oof_predictions[valid_idx, i] = model_fold.predict(X_valid_fold)
        
        # 训练元模型
        logging.info(f"训练元模型: {type(self.meta_model).__name__}")
        self.meta_model_ = self.meta_model.fit(oof_predictions, y)
        
        return self
    
    def predict(self, X):
        # 获取基础模型的预测结果
        meta_features = np.column_stack([
            np.mean([model.predict(X) for model in models], axis=0)
            for models in self.base_models_
        ])
        
        # 使用元模型进行最终预测
        return self.meta_model_.predict(meta_features)

class TemperatureRangeModel:
    """首先判断蛋白质所属温度区间，然后使用对应区间的专用回归模型"""
    
    def __init__(self, temp_thresholds=[30, 70]):
        """初始化温度分段模型
        
        参数:
            temp_thresholds: 温度分段阈值，默认[30, 70]，划分为低温(<30)、中温(30-70)和高温(>70)
        """
        # 温度区间阈值
        self.temp_thresholds = temp_thresholds
        
        # 温度区间分类器
        from sklearn.ensemble import RandomForestClassifier
        self.range_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        
        # 各温度区间的专用回归模型
        self.models = {}
        self.models['low'] = None     # 低温模型
        self.models['medium'] = None  # 中温模型 
        self.models['high'] = None    # 高温模型
        
        # 用于数据转换的缩放器
        self.scaler = RobustScaler()
        
        # 特征名称
        self.feature_names = None
        
    def fit(self, X, y, high_temp_focus=False):
        """训练分段模型
        
        参数:
            X: 特征矩阵
            y: 温度标签
            high_temp_focus: 是否重点优化高温区域
        """
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 应用特征缩放
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建温度区间标签
        bins = [float('-inf')] + self.temp_thresholds + [float('inf')]
        temp_ranges = pd.cut(y, bins=bins, labels=range(len(bins)-1))
        
        # 训练温度区间分类器
        logging.info("训练温度区间分类器...")
        self.range_classifier.fit(X_scaled, temp_ranges)
        
        # 按温度区间分割数据
        low_mask = y < self.temp_thresholds[0]
        medium_mask = (y >= self.temp_thresholds[0]) & (y < self.temp_thresholds[1])
        high_mask = y >= self.temp_thresholds[1]
        
        # 输出各区间样本数量
        logging.info(f"低温区间(<{self.temp_thresholds[0]}°C)样本数: {sum(low_mask)}")
        logging.info(f"中温区间({self.temp_thresholds[0]}-{self.temp_thresholds[1]}°C)样本数: {sum(medium_mask)}")
        logging.info(f"高温区间(>{self.temp_thresholds[1]}°C)样本数: {sum(high_mask)}")
        
        # 训练低温区间模型
        if sum(low_mask) > 10:
            logging.info("训练低温区间模型...")
            self.models['low'] = GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42
            )
            self.models['low'].fit(X_scaled[low_mask], y[low_mask])
        else:
            logging.warning(f"低温样本数量({sum(low_mask)})不足，无法训练专用模型")
        
        # 训练中温区间模型
        if sum(medium_mask) > 10:
            logging.info("训练中温区间模型...")
            self.models['medium'] = RandomForestRegressor(
                n_estimators=200, max_features=0.7, random_state=42
            )
            self.models['medium'].fit(X_scaled[medium_mask], y[medium_mask])
        else:
            logging.warning(f"中温样本数量({sum(medium_mask)})不足，无法训练专用模型")
        
        # 训练高温区间模型 - 特别优化
        if sum(high_mask) > 5:
            logging.info("训练高温区间模型...")
            X_high = X_scaled[high_mask]
            y_high = y[high_mask]
            
            # 如果高温样本太少且高温优化模式，进行数据增强
            if high_temp_focus and len(X_high) < 20 and len(X_high) >= 3:
                try:
                    # 尝试使用SMOTE类算法增加样本
                    sm = SMOTE(random_state=42)
                    X_high_res, y_high_res = sm.fit_resample(X_high, y_high)
                    logging.info(f"使用SMOTE增加高温样本: {len(X_high)} → {len(X_high_res)}")
                except Exception as e:
                    logging.warning(f"SMOTE采样失败: {str(e)}，使用原始样本")
                    X_high_res, y_high_res = X_high, y_high
            else:
                X_high_res, y_high_res = X_high, y_high
            
            # 高温模型使用更保守的配置以避免过拟合
            self.models['high'] = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.01,
                max_depth=3,
                subsample=0.8,
                random_state=42
            )
            self.models['high'].fit(X_high_res, y_high_res)
        else:
            logging.warning(f"高温样本数量({sum(high_mask)})不足，无法训练专用模型")
        
        return self
    
    def predict(self, X):
        """使用分段模型进行预测"""
        if not isinstance(X, pd.DataFrame):
            # 如果输入不是DataFrame，转换为DataFrame
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # 应用特征缩放
        X_scaled = self.scaler.transform(X)
        
        # 预测温度区间
        predicted_ranges = self.range_classifier.predict(X_scaled)
        
        # 根据预测的区间使用对应的回归模型
        predictions = np.zeros(len(X))
        
        for i, temp_range in enumerate(predicted_ranges):
            if temp_range == 0 and self.models['low'] is not None:
                # 低温区域
                predictions[i] = self.models['low'].predict([X_scaled[i]])[0]
            elif temp_range == 1 and self.models['medium'] is not None:
                # 中温区域
                predictions[i] = self.models['medium'].predict([X_scaled[i]])[0]
            elif temp_range == 2 and self.models['high'] is not None:
                # 高温区域
                predictions[i] = self.models['high'].predict([X_scaled[i]])[0]
            else:
                # 如果没有合适的模型，使用最接近的模型
                available_models = [model for model in self.models.values() if model is not None]
                if available_models:
                    # 使用最接近区间的模型
                    if temp_range == 0 and self.models['medium'] is not None:
                        predictions[i] = self.models['medium'].predict([X_scaled[i]])[0]
                    elif temp_range == 1 and self.models['high'] is not None:
                        predictions[i] = self.models['high'].predict([X_scaled[i]])[0]
                    elif temp_range == 2 and self.models['medium'] is not None:
                        predictions[i] = self.models['medium'].predict([X_scaled[i]])[0]
                    else:
                        # 使用任意可用模型
                        predictions[i] = available_models[0].predict([X_scaled[i]])[0]
                else:
                    # 如果没有任何模型可用，返回样本平均值
                    logging.error("没有可用的区间模型，返回样本平均值")
                    predictions[i] = 50.0  # 设置一个合理的默认值
        
        return predictions

class HighTempFocusModel:
    """高温蛋白专精预测模型 - 简化版
    
    该模型采用二阶段策略：
    1. 第一阶段：快速分类模型判断是否为高温蛋白(≥70°C)
    2. 第二阶段：
       - 若判断为高温蛋白，使用专门的回归模型进行精确温度预测
       - 若判断为低温蛋白，使用简化模型给出粗略估计
    """
    
    def __init__(self, temp_threshold=70):
        """初始化高温专精模型
        
        参数:
            temp_threshold: 高温阈值，默认70°C
        """
        self.temp_threshold = temp_threshold
        
        # 温度区间分类器 - 使用随机森林分类器
        from sklearn.ensemble import RandomForestClassifier
        self.classifier = RandomForestClassifier(
            n_estimators=200, 
            max_depth=None,
            class_weight='balanced',  # 处理类别不平衡
            random_state=42
        )
        
        # 高温专精回归器 - 使用更强大的集成学习模型
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.svm import SVR
        
        # 使用核SVM作为高温预测器，处理高温区域的非线性关系
        self.high_temp_regressor = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('svr', SVR(kernel='rbf', C=100, gamma='auto'))
        ])
        
        # 低温简化回归器 - 使用简单的随机森林
        self.low_temp_regressor = RandomForestRegressor(
            n_estimators=50,  # 进一步简化低温模型
            max_depth=10,     # 限制树深度，防止过拟合
            random_state=42
        )
        
        # 特征标准化器
        self.scaler = StandardScaler()
        
        # 特征名列表
        self.feature_names = None
        
        # 热稳定性特征的重要性权重
        self.thermo_features_weight = 2.5  # 增加热稳定性特征的权重，优化高温预测
    
    def fit(self, X, y, synthetic_samples=True, n_synthetic=50):
        """训练高温专精模型
        
        参数:
            X: 特征矩阵
            y: 温度标签
            synthetic_samples: 是否生成合成高温样本
            n_synthetic: 生成的合成样本数量
        """
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 识别热稳定性特征列
        thermo_features = [col for col in self.feature_names if col in [
            'ion_pair_density', 'core_residue_ratio', 'surface_charge_ratio', 
            'ivywrel_index', 'dense_hbond_network', 'compactness_index',
            'helix_sheet_ratio', 'aromatic_interactions', 'glycine_content'
        ]]
        
        # 增强热稳定性特征的权重
        X_weighted = X.copy()
        if thermo_features:
            logging.info(f"找到{len(thermo_features)}个热稳定性特征，增强其权重")
            for feature in thermo_features:
                X_weighted[feature] = X[feature] * self.thermo_features_weight
        
        # 应用特征缩放
        X_scaled = self.scaler.fit_transform(X_weighted)
        
        # 创建温度区间标签 (0表示低温区，1表示高温区)
        temp_classes = (y >= self.temp_threshold).astype(int)
        
        # 输出各温度区间样本数量
        high_count = sum(temp_classes)
        low_count = len(temp_classes) - high_count
        logging.info(f"低温区间(<{self.temp_threshold}°C)样本数: {low_count}")
        logging.info(f"高温区间(≥{self.temp_threshold}°C)样本数: {high_count}")
        
        # 生成合成高温样本（如需要）
        if synthetic_samples and high_count > 2 and high_count < low_count:
            X_aug, y_aug = self._generate_synthetic_samples(
                X_weighted, y, self.temp_threshold, n_synthetic
            )
            X_scaled_aug = self.scaler.transform(X_aug)
            temp_classes_aug = (y_aug >= self.temp_threshold).astype(int)
            
            # 记录增强后的样本数
            high_count_aug = sum(temp_classes_aug)
            logging.info(f"合成后高温样本数: {high_count_aug}")
        else:
            X_scaled_aug = X_scaled
            y_aug = y
            temp_classes_aug = temp_classes
        
        # 训练温度区间分类器
        logging.info("训练温度区间分类器...")
        self.classifier.fit(X_scaled_aug, temp_classes_aug)
        
        # 预测准确率评估
        train_pred_class = self.classifier.predict(X_scaled)
        accuracy = np.mean(train_pred_class == temp_classes)
        logging.info(f"温度区间分类器准确率: {accuracy:.2f}")
        
        # 分别训练高温和低温回归器
        high_mask = y >= self.temp_threshold
        if sum(high_mask) > 5:
            logging.info("训练高温专精回归器...")
            
            # 对高温样本进行加权，关注关键特征
            high_X = X_scaled[high_mask]
            high_y = y[high_mask]
            
            # 使用SVR回归器进行高温专精预测
            self.high_temp_regressor.fit(high_X, high_y)
            
            # 评估高温回归器的训练性能
            high_preds = self.high_temp_regressor.predict(high_X)
            high_rmse = np.sqrt(mean_squared_error(high_y, high_preds))
            high_r2 = r2_score(high_y, high_preds)
            logging.info(f"高温回归器训练性能: RMSE={high_rmse:.2f}, R²={high_r2:.2f}")
            
        else:
            logging.warning(f"高温样本数量({sum(high_mask)})不足，无法训练专用高温回归器")
        
        # 训练低温回归器（这里只是为了兼容性，精度要求不高）
        low_mask = ~high_mask
        if sum(low_mask) > 5:
            logging.info("训练低温简化回归器...")
            self.low_temp_regressor.fit(X_scaled[low_mask], y[low_mask])
        else:
            logging.warning(f"低温样本数量({sum(low_mask)})不足，无法训练低温回归器")
        
        return self
    
    def predict(self, X):
        """使用高温专精模型进行预测"""
        if not isinstance(X, pd.DataFrame):
            # 如果输入不是DataFrame，转换为DataFrame
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # 检查缺失的特征并填充
        for col in self.feature_names:
            if col not in X.columns:
                logging.warning(f"输入数据中缺少特征: {col}，填充为0")
                X[col] = 0.0
        
        # 使用与训练相同的热稳定性特征增强逻辑
        X_weighted = X.copy()
        thermo_features = [col for col in self.feature_names if col in [
            'ion_pair_density', 'core_residue_ratio', 'surface_charge_ratio', 
            'ivywrel_index', 'dense_hbond_network', 'compactness_index',
            'helix_sheet_ratio', 'aromatic_interactions', 'glycine_content'
        ]]
        
        if thermo_features:
            for feature in thermo_features:
                if feature in X.columns:
                    X_weighted[feature] = X[feature] * self.thermo_features_weight
        
        # 应用特征缩放
        X_scaled = self.scaler.transform(X_weighted[self.feature_names])
        
        # 预测温度区间
        is_high_temp = self.classifier.predict(X_scaled)
        
        # 初始化预测结果
        predictions = np.zeros(len(X))
        
        # 对每个样本使用适当的回归器
        for i, is_high in enumerate(is_high_temp):
            if is_high:
                # 高温区域，使用高温专精回归器
                if hasattr(self.high_temp_regressor, 'predict'):
                    predictions[i] = self.high_temp_regressor.predict([X_scaled[i]])[0]
                else:
                    # 降级到简单预测
                    predictions[i] = self.temp_threshold + 10.0
            else:
                # 低温区域，使用简化预测
                if hasattr(self.low_temp_regressor, 'predict'):
                    predictions[i] = self.low_temp_regressor.predict([X_scaled[i]])[0]
                else:
                    # 无可用回归器，使用简单估计值
                    predictions[i] = self.temp_threshold * 0.7
        
        return predictions
    
    def _generate_synthetic_samples(self, X, y, temp_threshold, n_samples):
        """为高温区间生成合成样本"""
        logging.info(f"生成高温合成样本，目标样本数={n_samples}")
        
        # 筛选高温样本
        high_temp_mask = y >= temp_threshold
        high_temp_X = X[high_temp_mask].copy()
        high_temp_y = y[high_temp_mask].copy()
        
        high_temp_count = len(high_temp_X)
        if high_temp_count < 3:
            logging.warning(f"高温样本数量({high_temp_count})不足，无法生成合成样本")
            return X, y
        
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
        
        logging.info(f"合成样本生成完成，高温样本数量: {sum(y_combined >= temp_threshold)}")
        
        return X_combined, y_combined

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

def generate_synthetic_high_temp_samples(X, y, temp_threshold=70, n_samples=50):
    """为高温区间生成合成样本，缓解样本不平衡问题
    
    参数:
        X: 特征矩阵
        y: 温度标签
        temp_threshold: 高温阈值，默认70°C
        n_samples: 要生成的合成样本数量
        
    返回:
        增强后的特征矩阵和标签
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
    """应用非线性特征变换"""
    logging.info("应用非线性特征变换...")
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
    
    # 添加多项式特征（2次）- 只对前5个特征应用以避免维度爆炸
    if len(X.columns) > 5:
        top_cols = X.columns[:5]
        poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(X[top_cols])
        feature_names = poly.get_feature_names_out(top_cols)
        
        # 添加多项式特征
        for i, name in enumerate(feature_names):
            if i >= len(top_cols):  # 跳过原始特征
                X_transformed[name] = poly_features[:, i]
    
    logging.info(f"非线性变换后的特征数量: {X_transformed.shape[1]}")
    return X_transformed

def prepare_features(data, apply_feature_selection=True, use_interaction=False, use_nonlinear=False):
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
    
    # 应用非线性特征变换
    if use_nonlinear:
        X = apply_nonlinear_transform(X)
    
    # 添加特征交互项
    if use_interaction:
        X = create_interaction_features(X)
    
    # 保存原始特征列名
    original_features = X.columns.tolist()
    
    # 执行特征选择
    if apply_feature_selection and len(X) > 20:  # 确保有足够的样本进行特征选择
        try:
            logging.info("开始进行特征选择...")
            # 初始化预选择模型
            pre_selector = RandomForestRegressor(n_estimators=150, random_state=42)
            pre_selector.fit(X, y)
            
            # 计算特征重要性
            importances = pre_selector.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # 记录特征重要性排名
            for i, idx in enumerate(indices[:20]):  # 只记录前20个重要特征
                if i < 10:  # 只显示前10个特征
                    feature_name = X.columns[idx]
                    logging.info(f"特征 {i+1}: {feature_name} - 重要性: {importances[idx]:.4f}")
            
            # 使用更精细的特征选择策略
            # 1. 先使用SelectFromModel初步筛选高重要性特征
            sfm = SelectFromModel(pre_selector, threshold='1.5*mean')
            sfm.fit(X, y)
            selected_features_mask = sfm.get_support()
            selected_features = [f for f, selected in zip(original_features, selected_features_mask) if selected]
            
            # 确保至少保留一定数量的特征
            min_features = max(15, len(original_features) // 4)  # 保留更多特征
            
            if len(selected_features) >= min_features:
                logging.info(f"特征选择后保留{len(selected_features)}个特征")
                
                # 2. 使用交叉验证评估特征选择结果
                X_selected = X[selected_features]
                
                # 创建评估管道
                eval_pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
                ])
                
                # 评估特征选择前后的性能
                cv_before = cross_val_score(eval_pipe, X, y, cv=5, scoring='r2')
                cv_after = cross_val_score(eval_pipe, X_selected, y, cv=5, scoring='r2')
                
                logging.info(f"特征选择前R²: {cv_before.mean():.3f}, 特征选择后R²: {cv_after.mean():.3f}")
                
                # 只有当特征选择后性能不下降时才使用选择后的特征集
                if cv_after.mean() >= cv_before.mean() * 0.95:  # 允许5%的性能损失
                    X = X_selected
                    feature_cols = selected_features
                    logging.info("应用特征选择结果")
                else:
                    logging.warning("特征选择导致性能显著下降，将使用全部特征")
            else:
                logging.warning(f"特征选择后只剩{len(selected_features)}个特征，低于最小阈值{min_features}，使用全部原始特征")
        except Exception as e:
            logging.error(f"特征选择过程出错: {str(e)}")
            logging.warning("将使用全部原始特征")
    
    return X, y, X.columns.tolist()

def train_neural_network(X_train, y_train, X_test, y_test):
    """训练神经网络模型"""
    logging.info("训练神经网络模型...")
    
    # 更复杂的神经网络结构
    hidden_layers = [(100,), (200,), (100, 50), (200, 100), (100, 50, 25)]
    activations = ['relu', 'tanh']
    alphas = [0.0001, 0.001, 0.01, 0.1]
    learning_rates = ['constant', 'adaptive']
    
    # 创建参数网格
    param_grid = {
        'hidden_layer_sizes': hidden_layers,
        'activation': activations,
        'alpha': alphas,
        'learning_rate': learning_rates,
        'max_iter': [2000],  # 增加最大迭代次数
        'early_stopping': [True],
        'random_state': [42]
    }
    
    # 定义基础神经网络
    nn_model = MLPRegressor(learning_rate_init=0.001, tol=1e-5)
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(
        estimator=nn_model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=0
    )
    
    try:
        # 先使用标准归一化处理数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        grid_search.fit(X_train_scaled, y_train)
        logging.info(f"神经网络最佳参数: {grid_search.best_params_}")
        
        # 使用最佳参数训练最终模型
        best_nn = MLPRegressor(
            **grid_search.best_params_,
            learning_rate_init=0.001,
            tol=1e-5,
            n_iter_no_change=20  # 增加无改进次数阈值
        )
        best_nn.fit(X_train_scaled, y_train)
        
        # 评估模型
        y_pred = best_nn.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logging.info(f"神经网络 - 测试集 RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        # 训练一个浅层简化版模型作为备选
        simple_nn = MLPRegressor(
            hidden_layer_sizes=(50,),
            activation='relu',
            alpha=0.01,
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
        simple_nn.fit(X_train_scaled, y_train)
        
        # 评估简化模型
        simple_y_pred = simple_nn.predict(X_test_scaled)
        simple_rmse = np.sqrt(mean_squared_error(y_test, simple_y_pred))
        simple_r2 = r2_score(y_test, simple_y_pred)
        
        logging.info(f"简化神经网络 - 测试集 RMSE: {simple_rmse:.2f}, R²: {simple_r2:.2f}")
        
        # 返回较好的模型
        if simple_r2 > r2:
            logging.info("选择简化神经网络模型")
            return simple_nn, simple_rmse, simple_r2
        else:
            return best_nn, rmse, r2
    except Exception as e:
        logging.error(f"神经网络训练失败: {str(e)}")
        return None, float('inf'), -1.0

def train_stacking_model(X_train, y_train, X_test, y_test):
    """训练堆叠模型"""
    logging.info("训练堆叠模型...")
    
    # 定义基础模型
    base_models = [
        RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42),
        GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=50, random_state=42),
        ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        Lasso(alpha=0.01, random_state=42),
        Ridge(alpha=1.0, random_state=42)
    ]
    
    # 尝试添加神经网络（如果可行）
    try:
        nn_model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', 
                               alpha=0.01, max_iter=1000, random_state=42)
        base_models.append(nn_model)
    except:
        logging.warning("无法添加神经网络到基础模型中")
    
    # 定义元模型 - 使用梯度提升作为元模型
    meta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    
    # 创建堆叠模型
    stacking_model = StackingRegressor(
        base_models=base_models,
        meta_model=meta_model,
        n_folds=5
    )
    
    # 训练模型
    try:
        # 先标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 转换为numpy数组避免索引错误
        X_train_scaled = np.array(X_train_scaled)
        y_train = np.array(y_train)
        
        stacking_model.fit(X_train_scaled, y_train)
        
        # 评估模型
        y_pred = stacking_model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logging.info(f"堆叠模型 - 测试集 RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        return stacking_model, rmse, r2
    except Exception as e:
        logging.error(f"堆叠模型训练失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, float('inf'), -1.0

def cross_validate_model(X, y, n_splits=5):
    """交叉验证评估模型性能"""
    logging.info(f"执行{n_splits}折交叉验证...")
    
    # 创建评估模型
    base_model = RandomForestRegressor(n_estimators=150, max_features='sqrt', random_state=42)
    
    # 创建包含数据标准化的管道
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', base_model)
    ])
    
    # 使用常规K折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 计算多个指标
    rmse_scores = -cross_val_score(model_pipeline, X, y, cv=kf, 
                                  scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(model_pipeline, X, y, cv=kf, 
                                 scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model_pipeline, X, y, cv=kf, 
                               scoring='r2')
    
    # 自定义评分器：温度预测的相对误差
    def temp_relative_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (y_true + 273.15))) * 100  # 返回相对误差百分比
    
    # 创建自定义评分器
    from sklearn.metrics import make_scorer
    rel_error_scorer = make_scorer(temp_relative_error, greater_is_better=False)
    
    # 计算相对误差
    rel_error_scores = -cross_val_score(model_pipeline, X, y, cv=kf, 
                                      scoring=rel_error_scorer)
    
    # 输出结果
    logging.info(f"交叉验证结果 - RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
    logging.info(f"交叉验证结果 - MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
    logging.info(f"交叉验证结果 - R²: {r2_scores.mean():.2f} ± {r2_scores.std():.2f}")
    logging.info(f"交叉验证结果 - 相对误差: {rel_error_scores.mean():.2f}% ± {rel_error_scores.std():.2f}%")
    
    return rmse_scores.mean(), mae_scores.mean(), r2_scores.mean()

def train_model(X, y, output_dir=None, use_advanced_models=False, sample_weights=None):
    """训练模型"""
    # 分割训练集和测试集 - 使用分层抽样
    try:
        # 将温度分组
        y_bins = pd.cut(y, 5, labels=False)
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # 分层抽样
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in split.split(X, y_bins):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if sample_weights is not None:
                # 提取训练样本的权重
                train_weights = sample_weights[train_idx]
                logging.info(f"使用样本权重进行训练，权重范围: {train_weights.min():.2f}-{train_weights.max():.2f}")
            else:
                train_weights = None
        
        logging.info("使用分层抽样分割数据集")
    except Exception as e:
        # 如果分层抽样失败，使用传统方法
        logging.warning(f"分层抽样失败: {e}，使用传统方法")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if sample_weights is not None:
            # 提取训练样本的权重
            train_weights = sample_weights[X_train.index]
        else:
            train_weights = None
    
    logging.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    logging.info(f"训练集温度范围: {y_train.min():.1f}°C - {y_train.max():.1f}°C")
    logging.info(f"测试集温度范围: {y_test.min():.1f}°C - {y_test.max():.1f}°C")
    
    # 执行数据标准化
    scaler = RobustScaler()  # 使用RobustScaler处理离群值
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建并训练基础随机森林模型
    rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
    rf_model.fit(X_train_scaled, y_train, sample_weight=train_weights)
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
    
    # 进行网格搜索优化超参数 - 使用更复杂的参数网格
    logging.info("开始网格搜索优化模型...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.3, 0.5, 0.7]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    # 使用样本权重拟合
    if train_weights is not None:
        grid_search.fit(X_train_scaled, y_train, sample_weight=train_weights)
    else:
        grid_search.fit(X_train_scaled, y_train)
    
    logging.info(f"最佳参数: {grid_search.best_params_}")
    
    # 使用最佳参数训练最终模型
    best_rf_model = RandomForestRegressor(
        **grid_search.best_params_,
        random_state=42
    )
    
    # 使用样本权重拟合最终模型
    if train_weights is not None:
        best_rf_model.fit(X_train_scaled, y_train, sample_weight=train_weights)
    else:
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
    
    # 初始化模型字典，存储所有训练的模型
    models = {
        '随机森林': (best_rf_model, best_test_rmse, best_test_r2)
    }
    
    # 尝试梯度提升模型 - 使用更复杂的配置
    logging.info("训练梯度提升模型...")
    param_grid_gb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    grid_search_gb = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid=param_grid_gb,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    # 使用样本权重拟合梯度提升模型网格搜索
    if train_weights is not None:
        grid_search_gb.fit(X_train_scaled, y_train, sample_weight=train_weights)
    else:
        grid_search_gb.fit(X_train_scaled, y_train)
        
    logging.info(f"梯度提升最佳参数: {grid_search_gb.best_params_}")
    
    gb_model = GradientBoostingRegressor(**grid_search_gb.best_params_, random_state=42)
    
    # 使用样本权重拟合梯度提升模型
    if train_weights is not None:
        gb_model.fit(X_train_scaled, y_train, sample_weight=train_weights)
    else:
        gb_model.fit(X_train_scaled, y_train)
    
    gb_test_pred = gb_model.predict(X_test_scaled)
    gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
    gb_test_r2 = r2_score(y_test, gb_test_pred)
    
    logging.info(f"梯度提升模型 - 测试集 RMSE: {gb_test_rmse:.2f}, R²: {gb_test_r2:.2f}")
    models['梯度提升'] = (gb_model, gb_test_rmse, gb_test_r2)
    
    # 如果使用高级模型
    if use_advanced_models:
        # 训练神经网络模型
        nn_model, nn_rmse, nn_r2 = train_neural_network(X_train, y_train, X_test, y_test)
        if nn_model is not None:
            models['神经网络'] = (nn_model, nn_rmse, nn_r2)
        
        # 训练堆叠模型
        stacking_model, stack_rmse, stack_r2 = train_stacking_model(
            X_train, y_train, X_test, y_test
        )
        if stacking_model is not None:
            models['堆叠模型'] = (stacking_model, stack_rmse, stack_r2)
    
    # 选择性能最好的模型(基于R²) - 改为优先考虑R²而不是RMSE
    best_model_name = max(models.keys(), key=lambda k: models[k][2])
    final_model, final_rmse, final_r2 = models[best_model_name]
    
    logging.info(f"最终选择模型: {best_model_name} (RMSE: {final_rmse:.2f}, R²: {final_r2:.2f})")
    
    # 保存模型和数据转换器
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'temperature_predictor.joblib')
    
    # 创建一个模型包，包含模型、特征名称和缩放器等
    model_package = {
        'model': final_model,
        'feature_names': X.columns.tolist(),
        'scaler': scaler,
        'model_type': best_model_name,
        'performance': {
            'rmse': final_rmse,
            'r2': final_r2
        }
    }
    
    joblib.dump(model_package, model_path)
    logging.info(f"模型包已保存至: {model_path}")
    
    # 生成特征重要性图(如果模型支持)
    try:
        plot_feature_importance(final_model, X, output_dir)
    except:
        logging.warning(f"{best_model_name}模型不支持生成特征重要性图")
    
    # 生成预测值与真实值对比图
    # 使用所选模型的测试集预测
    if best_model_name == '随机森林':
        final_test_pred = best_test_pred
    elif best_model_name == '梯度提升':
        final_test_pred = gb_test_pred
    elif best_model_name == '神经网络':
        final_test_pred = nn_model.predict(X_test_scaled)
    elif best_model_name == '堆叠模型':
        final_test_pred = stacking_model.predict(X_test_scaled)
    
    plot_predictions(y_test, final_test_pred, output_dir)
    
    # 创建残差图
    plot_residuals(y_test, final_test_pred, output_dir)
    
    # 生成模型比较图
    plot_model_comparison(X_test_scaled, y_test, models, output_dir)
    
    return final_model, X_test, y_test, final_test_pred, best_model_name

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
    parser.add_argument('--advanced_models', action='store_true',
                      help='使用高级模型（神经网络和堆叠模型）')
    parser.add_argument('--interactions', action='store_true',
                      help='使用特征交互')
    parser.add_argument('--nonlinear', action='store_true',
                      help='应用非线性特征变换')
    # 添加高温酶优化相关选项
    parser.add_argument('--segmented_model', action='store_true',
                      help='使用分段模型（为不同温度区间训练专用模型）')
    parser.add_argument('--high_temp_focus', action='store_true',
                      help='专注于优化高温区域的预测精度')
    parser.add_argument('--high_temp_specialist', action='store_true',
                      help='使用高温专精模型，对高温蛋白进行精确预测')
    parser.add_argument('--temperature_threshold', type=float, default=70.0,
                      help='定义高温区域的温度阈值（默认70°C）')
    parser.add_argument('--high_temp_weight', type=float, default=3.0,
                      help='高温样本权重系数（默认3.0）')
    parser.add_argument('--synthetic_samples', action='store_true',
                      help='为稀有温度区域生成合成样本')
    return parser.parse_args()

def evaluate_high_temperature_performance(y_true, y_pred, high_temp_threshold=70):
    """评估模型在高温区域的表现"""
    # 筛选高温样本
    high_temp_mask = y_true >= high_temp_threshold
    high_temp_y_true = y_true[high_temp_mask]
    high_temp_y_pred = y_pred[high_temp_mask]
    
    if len(high_temp_y_true) == 0:
        return {"error": "没有高温样本可供评估"}
    
    # 计算高温区域的评估指标
    high_temp_metrics = {
        "高温样本数量": len(high_temp_y_true),
        "高温平均真实值": float(np.mean(high_temp_y_true)),
        "高温平均预测值": float(np.mean(high_temp_y_pred)),
        "高温RMSE": float(np.sqrt(mean_squared_error(high_temp_y_true, high_temp_y_pred))),
        "高温MAE": float(mean_absolute_error(high_temp_y_true, high_temp_y_pred)),
        "高温R²": float(r2_score(high_temp_y_true, high_temp_y_pred)) if len(high_temp_y_true) > 2 else "样本不足",
        "高温相对误差": float(np.mean(np.abs((high_temp_y_true - high_temp_y_pred) / high_temp_y_true)) * 100)
    }
    
    logging.info(f"高温区域(≥{high_temp_threshold}°C)评估结果:")
    for metric, value in high_temp_metrics.items():
        logging.info(f"  {metric}: {value}")
    
    return high_temp_metrics

def plot_high_temp_performance(y_true, y_pred, output_dir, threshold=70):
    """可视化模型在高温区域的表现"""
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
            
            # 如果开启高温优化模式，打印温度分布统计信息
            if args.high_temp_focus or args.high_temp_specialist:
                temp_threshold = args.temperature_threshold
                high_temp_count = sum(data['optimal_temperature'] >= temp_threshold)
                total_count = len(data)
                logging.info(f"数据集温度分布统计:")
                logging.info(f"  总样本数: {total_count}")
                logging.info(f"  高温样本数(≥{temp_threshold}°C): {high_temp_count} ({high_temp_count/total_count*100:.1f}%)")
                logging.info(f"  中低温样本数(<{temp_threshold}°C): {total_count - high_temp_count} ({(total_count - high_temp_count)/total_count*100:.1f}%)")
            
            # 准备特征
            X, y, feature_cols = prepare_features(
                data, 
                apply_feature_selection=args.feature_selection,
                use_interaction=args.interactions,
                use_nonlinear=args.nonlinear
            )
            logging.info(f"使用{len(feature_cols)}个特征进行训练")
            
            # 如果启用合成样本生成
            if args.synthetic_samples and args.high_temp_focus:
                X, y = generate_synthetic_high_temp_samples(
                    X, y, temp_threshold=args.temperature_threshold, n_samples=50
                )
            
            # 执行交叉验证评估
            cv_rmse, cv_mae, cv_r2 = cross_validate_model(X, y)
            
            # 高温专精模型训练
            if args.high_temp_specialist:
                logging.info("使用高温专精模型训练...")
                # 分割训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
                )
                
                # 创建并训练高温专精模型
                specialist_model = HighTempFocusModel(
                    temp_threshold=args.temperature_threshold
                )
                
                # 训练模型
                specialist_model.fit(
                    X_train, y_train, 
                    synthetic_samples=args.synthetic_samples, 
                    n_synthetic=50
                )
                
                # 预测
                y_pred = specialist_model.predict(X_test)
                
                # 整体评估
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                logging.info(f"高温专精模型 - 整体测试集 RMSE: {rmse:.2f}, R²: {r2:.2f}")
                
                # 高温区域评估
                high_temp_metrics = evaluate_high_temperature_performance(
                    y_test, y_pred, high_temp_threshold=args.temperature_threshold
                )
                
                # 绘制高温区域性能图
                if os.path.exists(args.output):
                    plot_high_temp_performance(
                        y_test, y_pred, args.output, threshold=args.temperature_threshold
                    )
                
                # 保存模型
                model_path = os.path.join(args.output, 'temperature_predictor.joblib')
                model_package = {
                    'model': specialist_model,
                    'feature_names': X.columns.tolist(),
                    'model_type': '高温专精模型',
                    'performance': {
                        'rmse': rmse,
                        'r2': r2,
                        'high_temp_metrics': high_temp_metrics
                    }
                }
                joblib.dump(model_package, model_path)
                logging.info(f"高温专精模型已保存至: {model_path}")

                # 生成预测vs实际值对比图
                plot_predictions(y_test, y_pred, args.output)
                
                # 创建残差图
                plot_residuals(y_test, y_pred, args.output)
                
            elif args.segmented_model:
                # 使用原有的分段模型训练逻辑
                # ... 此处保留原有代码 ...
                logging.info("使用分段模型训练...")
                # 创建分段模型
                segmented_model = TemperatureRangeModel(
                    temp_thresholds=[30, args.temperature_threshold]
                )
                
                # 分割训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
                )
                
                # 训练分段模型
                segmented_model.fit(X_train, y_train, high_temp_focus=args.high_temp_focus)
                
                # 预测
                y_pred = segmented_model.predict(X_test)
                
                # 评估
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                logging.info(f"分段模型 - 测试集总体 RMSE: {rmse:.2f}, R²: {r2:.2f}")
                
                # 评估高温区域性能
                high_temp_metrics = evaluate_high_temperature_performance(
                    y_test, y_pred, high_temp_threshold=args.temperature_threshold
                )
                
                # 绘制高温区域性能图
                if os.path.exists(args.output):
                    plot_high_temp_performance(
                        y_test, y_pred, args.output, threshold=args.temperature_threshold
                    )
                
                # 保存模型
                model_path = os.path.join(args.output, 'temperature_predictor.joblib')
                model_package = {
                    'model': segmented_model,
                    'feature_names': X.columns.tolist(),
                    'model_type': '分段模型',
                    'performance': {
                        'rmse': rmse,
                        'r2': r2,
                        'high_temp_metrics': high_temp_metrics
                    }
                }
                joblib.dump(model_package, model_path)
                logging.info(f"分段模型已保存至: {model_path}")
                
            else:
                # 标准模型训练
                # ... 此处保留原有代码 ...
                # 如果开启高温样本加权
                sample_weights = None
                if args.high_temp_focus:
                    sample_weights = temperature_weighted_sampling(
                        X, y, temp_threshold=args.temperature_threshold, weight_factor=args.high_temp_weight
                    )
                
                # 训练模型
                model, X_test, y_test, y_pred, model_type = train_model(
                    X, y, args.output, use_advanced_models=args.advanced_models, sample_weights=sample_weights
                )
                
                # 评估高温区域性能
                high_temp_metrics = evaluate_high_temperature_performance(
                    y_test, y_pred, high_temp_threshold=args.temperature_threshold
                )
                
                # 绘制高温区域性能图
                if os.path.exists(args.output):
                    plot_high_temp_performance(
                        y_test, y_pred, args.output, threshold=args.temperature_threshold
                    )
            
            logging.info("训练完成！模型可用于预测新的PDB文件的最适温度")
            model_type_name = "高温专精模型" if args.high_temp_specialist else ("分段模型" if args.segmented_model else model_type)
            logging.info(f"交叉验证RMSE: {cv_rmse:.2f}, 最终模型类型: {model_type_name}")
            
        except Exception as e:
            logging.error(f"训练过程中出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc()) 