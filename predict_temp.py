#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import joblib
import argparse
import logging
import subprocess
import time
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# 导入HighTempFocusModel类以支持模型的加载
# 这段代码需要与train_rf_model.py中的HighTempFocusModel定义保持一致
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
        
        # 高温专精回归器 - 使用梯度提升回归器
        self.high_temp_regressor = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.01,  # 较小的学习率提高精度
            max_depth=5,
            subsample=0.8,  # 减少过拟合
            random_state=42
        )
        
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
        self.thermo_features_weight = 2.0
        
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

# 配置日志记录
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

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

def predict_temperature(input_path, model_path=None, output_dir=None, extract_features=True):
    """从PDB文件预测蛋白质最适温度"""
    # 默认模型路径
    if model_path is None:
        model_path = os.path.join(os.getcwd(), 'models', 'temperature_predictor.joblib')
    
    # 默认输出目录
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logging.error(f"找不到模型文件: {model_path}")
        return False
    
    # 检查输入路径
    if not os.path.exists(input_path):
        logging.error(f"找不到输入路径: {input_path}")
        return False
    
    # 如果需要提取特征
    if extract_features:
        logging.info(f"从 {input_path} 提取特征...")
        cmd = ['python', 'analyze_pdb.py', input_path, '--thermostability']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"特征提取失败: {result.stderr}")
                return False
            # 等待文件系统更新
            time.sleep(1)
        except Exception as e:
            logging.error(f"执行 analyze_pdb.py 时出错: {str(e)}")
            return False
    
    # 查找最新的分析文件
    analyze_files = glob.glob(os.path.join(output_dir, 'analyze_pdb_*.csv'))
    if not analyze_files:
        logging.error(f"在 {output_dir} 中找不到分析结果文件")
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
    
    # 加载模型
    try:
        model_package = joblib.load(model_path)
        logging.info(f"已加载模型包: {os.path.basename(model_path)}")
        
        model = model_package['model']
        feature_names = model_package['feature_names']
        model_type = model_package.get('model_type', '未知')
        
        # 检查缺失的特征并添加
        for col in feature_names:
            if col not in pdb_data.columns:
                logging.warning(f"输入数据中缺少特征: {col}，填充为0")
                pdb_data[col] = 0
        
        # 确保只选择模型需要的特征列
        X = pdb_data[feature_names]
        
        # 检查并填充NaN值
        if X.isnull().any().any():
            logging.warning("特征数据中存在NaN值，将使用列平均值填充")
            X = X.fillna(X.mean())
        
        # 预测温度
        predictions = model.predict(X)
        
        # 添加预测结果到DataFrame
        pdb_data['predicted_temperature'] = predictions
        
        # 添加高温标记列
        pdb_data['is_high_temp'] = predictions >= 70
        
        # 将pdb_id和预测温度放在前面
        columns = ['pdb_id', 'predicted_temperature', 'is_high_temp'] + [
            col for col in pdb_data.columns if col not in ['pdb_id', 'predicted_temperature', 'is_high_temp']
        ]
        pdb_data = pdb_data[columns]
        
        # 保存预测结果
        prediction_file = os.path.join(os.getcwd(), 'prediction_results.csv')
        pdb_data.to_csv(prediction_file, index=False)
        logging.info(f"预测结果已保存至: {prediction_file}")
        
        # 打印预测结果
        print("\n预测结果汇总表:")
        print(f"{'PDB ID':<15}{'预测温度':<15}{'温度分类'}")
        print("-" * 40)
        for index, row in pdb_data.iterrows():
            temp_class = "高温" if row['is_high_temp'] else "低温"
            print(f"{row['pdb_id']:<15}{row['predicted_temperature']:.2f}°C{'':<5}{temp_class}")
        
        # 生成预测结果图表
        plot_predictions(pdb_data)
        
        return True
    except Exception as e:
        logging.error(f"预测过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def plot_predictions(prediction_data):
    """可视化预测结果"""
    # 提取所需数据
    pdb_ids = prediction_data['pdb_id'].tolist()
    temperatures = prediction_data['predicted_temperature'].tolist()
    is_high_temp = prediction_data['is_high_temp'].tolist()
    
    # 创建颜色列表
    colors = ['red' if ht else 'blue' for ht in is_high_temp]
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(pdb_ids, temperatures, color=colors)
    
    # 添加水平线表示高温阈值
    plt.axhline(y=70, color='black', linestyle='--', alpha=0.7, label='高温阈值 (70°C)')
    
    # 添加标签和标题
    plt.xlabel('PDB ID')
    plt.ylabel('预测最适温度 (°C)')
    plt.title('蛋白质最适温度预测结果')
    
    # 添加数据标签
    for bar, temp in zip(bars, temperatures):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{temp:.1f}°C', ha='center', va='bottom', rotation=0)
    
    # 添加图例
    plt.legend(['高温阈值 (70°C)', '高温蛋白', '低温蛋白'], 
               loc='upper right', 
               handler_map={str: plt.matplotlib.legend_handler.HandlerBase()})
    
    # 旋转x轴标签以防止重叠
    plt.xticks(rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('temperature_prediction_results.png')
    logging.info("预测结果图表已保存至: temperature_prediction_results.png")
    
    plt.close()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='预测蛋白质的最适温度并标记是否为高温蛋白')
    parser.add_argument('input_path', help='输入PDB文件或包含PDB文件的目录路径')
    parser.add_argument('--model', default=None, help='模型文件路径')
    parser.add_argument('--output', default=None, help='输出目录路径')
    parser.add_argument('--no-extract', action='store_true', help='跳过特征提取步骤，直接使用现有分析文件')
    return parser.parse_args()

if __name__ == "__main__":
    # 设置中文字体支持
    setup_chinese_font()
    
    # 解析命令行参数
    args = parse_args()
    
    # 调用预测函数
    predict_temperature(
        args.input_path,
        model_path=args.model,
        output_dir=args.output,
        extract_features=not args.no_extract
    ) 