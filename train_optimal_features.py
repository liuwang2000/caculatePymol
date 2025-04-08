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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from tqdm import tqdm
import time
import datetime
from joblib import Parallel, delayed
import multiprocessing
import sys
import platform
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# 导入train_rf_model.py中的函数
from train_rf_model import setup_chinese_font, load_data, prepare_features, train_model
from train_rf_model import cross_validate_model, plot_feature_importance, plot_predictions

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('feature_selection.log', encoding='utf-8')
    ]
)

# 设置中文字体
def setup_chinese_font():
    """设置中文字体支持"""
    try:
        # 获取matplotlib可用字体列表
        import matplotlib.font_manager as fm
        available_fonts = set([f.name for f in fm.fontManager.ttflist])
        logging.info(f"可用字体数量: {len(available_fonts)}")
        
        # 针对不同系统的字体选项
        if platform.system() == 'Windows':
            # Windows系统字体选项
            chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
        elif platform.system() == 'Darwin':
            # macOS系统字体选项
            chinese_fonts = ['PingFang SC', 'STHeiti', 'Heiti TC', 'Songti SC', 'Songti TC', 'Kaiti SC', 'Kaiti TC']
        else:
            # Linux系统字体选项 - 提供更多选项
            chinese_fonts = [
                'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 
                'Noto Sans CJK JP', 'Noto Sans Mono CJK SC', 'Noto Serif CJK SC', 'Source Han Sans CN',
                'Source Han Sans TW', 'Source Han Serif CN', 'Source Han Serif TW', 'AR PL UMing CN',
                'AR PL UKai CN', 'DejaVu Sans', 'Liberation Sans', 'Droid Sans Fallback'
            ]
        
        # 查找第一个可用的中文字体
        font_found = False
        for font in chinese_fonts:
            if font in available_fonts:
                logging.info(f"找到中文字体: {font}")
                plt.rcParams['font.sans-serif'] = [font]
                font_found = True
                break
        
        # 如果没有找到预定义的中文字体，尝试找到任何包含中文字符的字体
        if not font_found:
            import os
            import subprocess
            from matplotlib.font_manager import fontManager, FontProperties
            
            logging.info("预定义中文字体未找到，尝试通过系统命令查找...")
            try:
                # 在Linux和macOS上尝试使用fc-list查找中文字体
                if platform.system() != 'Windows':
                    # 查询支持中文的字体
                    fc_list_cmd = "fc-list :lang=zh"
                    fonts_output = subprocess.check_output(fc_list_cmd, shell=True, text=True)
                    
                    # 解析字体名称
                    font_paths = fonts_output.split('\n')
                    if font_paths and len(font_paths) > 0:
                        for path in font_paths:
                            if path.strip():
                                # 提取字体文件路径
                                path_parts = path.split(':')
                                if path_parts:
                                    font_path = path_parts[0].strip()
                                    if os.path.exists(font_path):
                                        logging.info(f"添加系统中文字体: {font_path}")
                                        fontManager.addfont(font_path)
                                        font_name = FontProperties(fname=font_path).get_name()
                                        plt.rcParams['font.sans-serif'] = [font_name]
                                        font_found = True
                                        break
            except Exception as e:
                logging.warning(f"系统字体查找失败: {str(e)}")
            
            # 如果仍然找不到中文字体，尝试重置字体管理器并使用任何可能支持的字体
            if not font_found:
                logging.warning("未找到中文字体，将尝试使用系统默认无衬线字体")
                plt.rcParams['font.family'] = 'sans-serif'
                
                # 最后的备选项
                if 'DejaVu Sans' in available_fonts:
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                    logging.info("使用DejaVu Sans字体")
        
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        
        # 验证字体设置是否正确
        current_font = plt.rcParams['font.sans-serif']
        logging.info(f"当前使用字体: {current_font}")
        
        if font_found:
            logging.info("成功配置中文字体支持")
        else:
            logging.warning("无法找到合适的中文字体，图表中的中文可能无法正确显示")
    except Exception as e:
        logging.warning(f"配置中文字体失败: {str(e)}")
        logging.warning("图表中的中文可能无法正确显示")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

# 确保日志输出使用UTF-8编码
def setup_logging():
    """设置日志输出编码"""
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stdout)
            handler.encoding = 'utf-8'

# 在main函数开始处调用
setup_logging()

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

def evaluate_combination(feature_subset, X, y, test_size, random_state, cv):
    """评估一个特征组合"""
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
    
    # 交叉验证
    cv_scores = cross_validate_model(X_train_scaled, y_train, n_splits=cv)
    # cross_validate_model返回一个包含三个元素的元组(rmse_mean, mae_mean, r2_mean)
    if isinstance(cv_scores, tuple) and len(cv_scores) >= 3:
        # 元组的第三个元素是r2_mean
        mean_cv_score = cv_scores[2]
    elif isinstance(cv_scores, dict) and 'r2' in cv_scores:
        # 如果是字典并且包含r2键
        mean_cv_score = np.mean(cv_scores['r2'])
    else:
        # 其他情况，尝试直接取平均值
        mean_cv_score = np.mean(cv_scores)
    
    # 使用train_rf_model.py中的train_model函数进行训练
    # 将X_train_scaled转回DataFrame以便train_model使用
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_subset)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_subset)
    
    # 训练模型
    model, X_test_out, y_test_out, y_pred, model_name = train_model(X_train_df, y_train, use_advanced_models=False)
    
    # 在测试集上评估
    y_test_pred = model.predict(X_test_df)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    return {
        'features': feature_subset,
        'num_features': len(feature_subset),
        'cv_r2': mean_cv_score,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'model': model,
        'scaler': scaler
    }

# 将嵌套函数提升到模块级别，解决序列化问题
def evaluate_with_early_stopping(feature_subset, X, y, test_size, random_state, cv, early_stop_event, current_best_r2_value):
    """评估一个特征组合，支持提前终止"""
    # 如果收到终止信号，立即返回None
    if early_stop_event.is_set():
        return None
    
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
    
    # 交叉验证
    cv_scores = cross_validate_model(X_train_scaled, y_train, n_splits=cv)
    # cross_validate_model返回一个包含三个元素的元组(rmse_mean, mae_mean, r2_mean)
    if isinstance(cv_scores, tuple) and len(cv_scores) >= 3:
        mean_cv_score = cv_scores[2]
    elif isinstance(cv_scores, dict) and 'r2' in cv_scores:
        mean_cv_score = np.mean(cv_scores['r2'])
    else:
        mean_cv_score = np.mean(cv_scores)
    
    # 创建训练和测试数据帧
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_subset)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_subset)
    
    # 直接监控train_model的输出
    try:
        # 导入需要的模块
        import io
        import re
        import subprocess
        import threading
        import tempfile
        
        # 特征集的字符串表示（用于日志）
        features_str = ','.join(feature_subset)[:50] + ('...' if len(feature_subset) > 10 else '')
        
        # 创建一个子进程来运行train_model
        # 首先，将训练数据和测试数据保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as train_file, \
             tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as test_file:
            
            train_data = {'X': X_train_df, 'y': y_train}
            test_data = {'X': X_test_df, 'y': y_test}
            
            joblib.dump(train_data, train_file.name)
            joblib.dump(test_data, test_file.name)
            
            train_file_path = train_file.name
            test_file_path = test_file.name
        
        # 创建一个Python脚本，导入train_rf_model并运行模型训练
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as script_file:
            script_content = f"""
import sys
import joblib
import logging
import numpy as np
from sklearn.metrics import r2_score

# 确保可以导入train_rf_model
sys.path.append('.')
from train_rf_model import train_model

# 加载数据
train_data = joblib.load('{train_file_path}')
test_data = joblib.load('{test_file_path}')

X_train = train_data['X']
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']

# 训练模型
model, X_test_out, y_test_out, y_pred, model_name = train_model(X_train, y_train, use_advanced_models=False)

# 在测试集上评估并输出R2值
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
print(f"测试集 R²: {{test_r2:.4f}}")

# 保存模型
result = {{
    'model': model,
    'test_r2': test_r2
}}
joblib.dump(result, '{train_file_path}.result')
"""
            script_file.write(script_content)
            script_path = script_file.name
        
        # 创建一个标志，用于指示当前模型已被终止
        terminated = threading.Event()
        
        # 启动子进程
        process = subprocess.Popen(
            ['python', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # 行缓冲
        )
        
        # 存储捕获的输出
        output_lines = []
        current_r2 = None
        
        # 监控进程输出并检查R2值
        logging.info(f"开始训练特征组合: {features_str}")
        for line in process.stdout:
            output_line = line.strip()
            output_lines.append(output_line)
            logging.info(f"[{features_str}] {output_line}")
            
            # 检查是否包含测试集R2值的输出
            r2_match = re.search(r'测试集\s+R²:\s+([-+]?\d*\.\d+)', output_line)
            if r2_match:
                try:
                    r2_value = float(r2_match.group(1))
                    current_r2 = r2_value
                    
                    # 如果R2值低于当前最佳值，终止进程
                    if current_r2 < current_best_r2_value.value:
                        logging.info(f"[{features_str}] 中止训练 - 测试集R² ({current_r2:.4f}) 低于当前最佳值 ({current_best_r2_value.value:.4f})")
                        process.terminate()
                        terminated.set()
                        break
                except ValueError:
                    pass
            
            # 检查是否包含"开始网格搜索优化模型..."的输出
            if "开始网格搜索优化模型..." in output_line:
                # 检查是否已经有训练集和测试集R2值
                train_r2_match = re.search(r'训练集.*R²:\s+([-+]?\d*\.\d+)', '\n'.join(output_lines[-5:]))
                test_r2_match = re.search(r'测试集.*R²:\s+([-+]?\d*\.\d+)', '\n'.join(output_lines[-5:]))
                
                if train_r2_match and test_r2_match:
                    try:
                        test_r2_value = float(test_r2_match.group(1))
                        current_r2 = test_r2_value
                        
                        # 如果R2值低于当前最佳值，终止进程
                        if current_r2 < current_best_r2_value.value:
                            logging.info(f"[{features_str}] 中止网格搜索 - 初始测试集R² ({current_r2:.4f}) 低于当前最佳值 ({current_best_r2_value.value:.4f})")
                            process.terminate()
                            terminated.set()
                            break
                    except ValueError:
                        pass
        
        # 如果输出流结束但没有发现终止条件，还需要检查stderr
        if not terminated.is_set():
            for line in process.stderr:
                logging.info(f"[{features_str}] STDERR: {line.strip()}")
        
        # 获取进程返回值
        if not terminated.is_set():
            process.wait()
        
        # 读取训练结果
        if os.path.exists(f"{train_file_path}.result") and not terminated.is_set():
            try:
                result_data = joblib.load(f"{train_file_path}.result")
                model = result_data['model']
                test_r2 = result_data['test_r2']
                
                # 记录最终评估结果
                logging.info(f"特征组合 {features_str} 最终测试集 R²: {test_r2:.4f}")
                
                # 如果测试集R2值低于当前最佳值，提前返回
                if test_r2 < current_best_r2_value.value:
                    logging.info(f"舍弃结果 - 特征组合测试集R² ({test_r2:.4f}) 低于当前最佳值 ({current_best_r2_value.value:.4f})")
                    return {
                        'features': feature_subset,
                        'num_features': len(feature_subset),
                        'test_r2': test_r2,
                        'skip_detailed': True
                    }
                
                # 在测试集上评估
                y_test_pred = model.predict(X_test_df)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                return {
                    'features': feature_subset,
                    'num_features': len(feature_subset),
                    'cv_r2': mean_cv_score,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'model': model,
                    'scaler': scaler,
                    'skip_detailed': False
                }
            except Exception as e:
                logging.error(f"读取训练结果出错: {e}")
        
        # 如果进程被终止或结果文件不存在，返回当前的R2值（如果有的话）
        if current_r2 is not None:
            logging.info(f"使用提前终止的结果 - 特征组合测试集R² ({current_r2:.4f})")
            return {
                'features': feature_subset,
                'num_features': len(feature_subset),
                'test_r2': current_r2,
                'skip_detailed': True
            }
        else:
            # 如果没有获取到任何R2值，返回一个低值
            logging.warning(f"无法获取测试集R²值，返回默认低值")
            return {
                'features': feature_subset,
                'num_features': len(feature_subset),
                'test_r2': -1.0,  # 一个足够低的值，确保不会被选为最佳结果
                'skip_detailed': True
            }
        
    except Exception as e:
        logging.error(f"模型训练失败: {e}")
        return {
            'features': feature_subset,
            'num_features': len(feature_subset),
            'test_r2': -1.0,
            'skip_detailed': True
        }
    finally:
        # 清理临时文件
        try:
            for file_path in [train_file_path, test_file_path, script_path]:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            if os.path.exists(f"{train_file_path}.result"):
                os.unlink(f"{train_file_path}.result")
        except Exception as e:
            logging.warning(f"清理临时文件时出错: {e}")

def feature_selection_by_combination(X, y, feature_cols, min_features=3, max_features=15, 
                                     cv=5, random_state=42, top_k=10, test_size=0.2):
    """通过尝试不同的特征组合来找到最佳特征集，使用双线程比较淘汰"""
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
    
    # 创建所有特征组合列表 - 按特征数量分组
    all_combinations = []
    for num_features in range(min_features, min(max_features + 1, n_features + 1)):
        for feature_subset in combinations(feature_cols, num_features):
            all_combinations.append(feature_subset)
    
    total_combinations = len(all_combinations)
    logging.info(f"需要测试{total_combinations}种特征组合")
    
    # 预估所需时间
    avg_time_per_combo = 1.0  # 估计每个组合1秒（考虑到串行方式）
    estimated_time = total_combinations * avg_time_per_combo / 2  # 考虑双线程加速
    
    estimated_str = format_time(estimated_time)
    logging.info(f"预计完成特征组合测试需要约{estimated_str}，使用双线程比较策略")
    
    # 创建进度条
    pbar = tqdm(total=total_combinations, desc="测试特征组合")
    
    # 记录开始时间
    combo_start_time = time.time()
    
    # 导入需要的模块
    import concurrent.futures
    import queue
    
    # 创建一个线程安全的队列来存储最佳结果
    best_results = []
    best_results_lock = threading.Lock()
    
    # 添加一个全局最佳结果变量，用于跟踪所有组合中最佳的结果
    global_best_result = {
        'test_r2': -float('inf'),
        'result': None
    }
    global_best_lock = threading.Lock()
    
    def update_global_best(result):
        """更新全局最佳结果并输出日志"""
        with global_best_lock:
            if result['test_r2'] > global_best_result['test_r2']:
                global_best_result['test_r2'] = result['test_r2']
                global_best_result['result'] = result
                logging.info(f"发现新的全局最佳模型 - 测试集R²: {result['test_r2']:.4f}")
                logging.info(f"特征组合: {', '.join(result['features'])}")
                logging.info(f"当前全局最高测试集R²: {global_best_result['test_r2']:.4f}")
    
    # 运行双线程比较评估的函数
    def run_dual_comparison():
        completed = 0
        skipped = 0
        
        # 遍历所有特征组合
        combinations_queue = queue.Queue()
        for combo in all_combinations:
            combinations_queue.put(combo)
        
        # 当前最佳特征组合的信息
        current_best = {
            'features': None,
            'r2': -float('inf'),
            'process': None,
            'future': None,
            'output_buffer': [],
            'terminated': False
        }
        
        # 保持运行直到所有组合都被评估
        while not combinations_queue.empty() or current_best['process'] is not None:
            # 如果当前没有正在运行的最佳组合，从队列中取出一个
            if current_best['process'] is None and not combinations_queue.empty():
                next_features = combinations_queue.get()
                current_best = run_evaluation(next_features, current_best=None)
                completed += 1
                pbar.update(1)
                continue
            
            # 如果队列不为空，取下一个特征组合进行评估
            if not combinations_queue.empty():
                next_features = combinations_queue.get()
                
                # 运行下一个特征组合的评估
                challenger = run_evaluation(next_features, current_best)
                completed += 1
                pbar.update(1)
                
                # 比较当前最佳和挑战者
                if challenger['r2'] > current_best['r2']:
                    # 挑战者更好，终止当前最佳
                    logging.info(f"发现更好的特征组合 - 新R²: {challenger['r2']:.4f} > 旧R²: {current_best['r2']:.4f}")
                    
                    if current_best['process'] and not current_best['terminated']:
                        logging.info(f"终止测试集R²较低的特征组合: {','.join(current_best['features'])[:50]}...")
                        current_best['process'].terminate()
                        current_best['terminated'] = True
                        try:
                            if current_best['future'] and not current_best['future'].done():
                                current_best['future'].cancel()
                        except Exception as e:
                            logging.warning(f"取消Future时出错: {e}")
                        skipped += 1
                    
                    # 更新当前最佳
                    current_best = challenger
                else:
                    # 当前最佳更好，终止挑战者
                    logging.info(f"保留当前最佳特征组合 - 当前R²: {current_best['r2']:.4f} > 新R²: {challenger['r2']:.4f}")
                    
                    if challenger['process'] and not challenger['terminated']:
                        logging.info(f"终止测试集R²较低的特征组合: {','.join(challenger['features'])[:50]}...")
                        challenger['process'].terminate()
                        challenger['terminated'] = True
                        try:
                            if challenger['future'] and not challenger['future'].done():
                                challenger['future'].cancel()
                        except Exception as e:
                            logging.warning(f"取消Future时出错: {e}")
                        skipped += 1
            
            # 如果当前最佳的进程已完成，处理其结果
            if current_best['future'] and current_best['future'].done():
                try:
                    # 使用安全的方式获取Future结果
                    try:
                        result = current_best['future'].result(timeout=0.1)  # 设置超时避免阻塞
                    except concurrent.futures.TimeoutError:
                        # 如果超时，继续等待
                        continue
                    except Exception as e:
                        logging.error(f"获取Future结果时出错: {e}")
                        result = None
                    
                    if result and not result.get('skip_detailed', False):
                        # 添加到最佳结果列表
                        with best_results_lock:
                            best_results.append(result)
                            best_results.sort(key=lambda x: x['test_r2'], reverse=True)
                            if len(best_results) > top_k:
                                best_results.pop()
                        
                        # 更新全局最佳结果
                        update_global_best(result)
                except Exception as e:
                    logging.error(f"处理特征组合结果时出错: {e}")
                
                # 重置当前最佳
                current_best = {
                    'features': None,
                    'r2': -float('inf'),
                    'process': None,
                    'future': None,
                    'output_buffer': [],
                    'terminated': False
                }
            
            # 稍作等待以减少CPU使用
            time.sleep(0.1)
        
        return completed, skipped
    
    # 运行单个特征组合评估的函数
    def run_evaluation(feature_subset, current_best):
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
        
        # 创建训练和测试数据帧
        X_train_df = pd.DataFrame(X_train_scaled, columns=feature_subset)
        X_test_df = pd.DataFrame(X_test_scaled, columns=feature_subset)
        
        # 导入需要的模块
        import io
        import re
        import subprocess
        import tempfile
        
        # 特征集的字符串表示
        features_str = ','.join(feature_subset)[:50] + ('...' if len(feature_subset) > 10 else '')
        
        # 创建临时文件
        train_file_path = None
        test_file_path = None
        script_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as train_file, \
                 tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as test_file:
                
                train_data = {'X': X_train_df, 'y': y_train}
                test_data = {'X': X_test_df, 'y': y_test}
                
                joblib.dump(train_data, train_file.name)
                joblib.dump(test_data, test_file.name)
                
                train_file_path = train_file.name
                test_file_path = test_file.name
            
            # 创建Python脚本
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as script_file:
                script_content = f"""
import sys
import joblib
import logging
import numpy as np
from sklearn.metrics import r2_score

# 确保可以导入train_rf_model
sys.path.append('.')
from train_rf_model import train_model

# 加载数据
train_data = joblib.load('{train_file_path}')
test_data = joblib.load('{test_file_path}')

X_train = train_data['X']
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']

# 训练模型
model, X_test_out, y_test_out, y_pred, model_name = train_model(X_train, y_train, use_advanced_models=False)

# 在测试集上评估并输出R2值
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
print(f"测试集 R²: {{test_r2:.4f}}")

# 保存模型
result = {{
    'model': model,
    'test_r2': test_r2,
    'scaler': None  # 稍后在主进程中设置
}}
joblib.dump(result, '{train_file_path}.result')
"""
                script_file.write(script_content)
                script_path = script_file.name
            
            # 创建输出缓冲区和R2值变量
            output_buffer = []
            r2_value = -float('inf')
            rf_r2_found = threading.Event()
            
            # 启动子进程
            logging.info(f"开始评估特征组合: {features_str}")
            process = subprocess.Popen(
                ['python', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # 定义结果处理函数
            def process_output():
                nonlocal r2_value
                
                # 监控进程输出
                for line in process.stdout:
                    line = line.strip()
                    output_buffer.append(line)
                    logging.info(f"[{features_str}] {line}")
                    
                    # 检查是否包含测试集R2值
                    r2_match = re.search(r'测试集\s+R²:\s+([-+]?\d*\.\d+)', line)
                    if r2_match:
                        try:
                            r2_value = float(r2_match.group(1))
                            rf_r2_found.set()  # 设置事件，表示找到了随机森林的R2值
                        except ValueError:
                            pass
                    
                    # 检查是否包含"开始网格搜索优化模型..."
                    if "开始网格搜索优化模型..." in line and current_best and current_best['r2'] > r2_value:
                        logging.info(f"[{features_str}] 网格搜索前发现R²值({r2_value:.4f})低于当前最佳({current_best['r2']:.4f})，准备终止")
                        return
                
                # 检查stderr
                for line in process.stderr:
                    logging.info(f"[{features_str}] STDERR: {line.strip()}")
            
            # 使用线程池提交任务
            future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(process_output)
            
            # 等待随机森林的R2值或进程结束
            while not rf_r2_found.is_set() and process.poll() is None:
                time.sleep(0.1)
            
            # 如果找到了R2值，检查是否需要停止
            if rf_r2_found.is_set() and current_best and current_best['r2'] > r2_value:
                logging.info(f"[{features_str}] R²值({r2_value:.4f})低于当前最佳({current_best['r2']:.4f})，准备终止")
            
            # 准备返回结果
            result = {
                'features': feature_subset,
                'r2': r2_value,
                'process': process,
                'future': future,
                'output_buffer': output_buffer,
                'terminated': False
            }
            
            # 如果R2值已找到，并且这是一个完整的评估（没有当前最佳进行比较）
            if rf_r2_found.is_set() and current_best is None:
                # 等待进程完成
                process.wait()
                
                # 尝试加载结果
                if os.path.exists(f"{train_file_path}.result"):
                    try:
                        result_data = joblib.load(f"{train_file_path}.result")
                        
                        # 创建完整结果
                        cv_scores = cross_validate_model(X_train_scaled, y_train, n_splits=cv)
                        if isinstance(cv_scores, tuple) and len(cv_scores) >= 3:
                            mean_cv_score = cv_scores[2]
                        elif isinstance(cv_scores, dict) and 'r2' in cv_scores:
                            mean_cv_score = np.mean(cv_scores['r2'])
                        else:
                            mean_cv_score = np.mean(cv_scores)
                        
                        # 计算其他指标
                        y_test_pred = result_data['model'].predict(X_test_df)
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                        test_mae = mean_absolute_error(y_test, y_test_pred)
                        
                        # 完整的结果对象
                        complete_result = {
                            'features': feature_subset,
                            'num_features': len(feature_subset),
                            'cv_r2': mean_cv_score,
                            'test_r2': result_data['test_r2'],
                            'test_rmse': test_rmse,
                            'test_mae': test_mae,
                            'model': result_data['model'],
                            'scaler': scaler,
                            'skip_detailed': False
                        }
                        
                        # 创建一个自定义Future并设置结果
                        custom_future = concurrent.futures.Future()
                        custom_future.set_result(complete_result)
                        result['future'] = custom_future
                    except Exception as e:
                        logging.error(f"读取训练结果出错: {e}")
                        # 不再尝试设置结果，而是记录错误并继续
            
            return result
            
        finally:
            # 清理临时文件
            try:
                for file_path in [train_file_path, test_file_path, script_path]:
                    if file_path and os.path.exists(file_path):
                        os.unlink(file_path)
                if train_file_path and os.path.exists(f"{train_file_path}.result"):
                    os.unlink(f"{train_file_path}.result")
            except Exception as e:
                logging.warning(f"清理临时文件时出错: {e}")
    
    # 运行双线程比较策略
    completed, skipped = run_dual_comparison()
    
    # 关闭进度条
    pbar.close()
    
    # 计算实际耗时
    combo_end_time = time.time()
    combo_elapsed = combo_end_time - combo_start_time
    combo_elapsed_str = format_time(combo_elapsed)
    logging.info(f"特征组合测试完成，实际耗时: {combo_elapsed_str}")
    logging.info(f"总共评估了 {completed}/{total_combinations} 个组合，提前跳过了 {skipped} 个表现不佳的组合")
    
    # 确保结果按测试集R2排序
    best_results.sort(key=lambda x: x['test_r2'], reverse=True)
    
    # 如果全局最佳结果不在best_results中，添加它
    if global_best_result['result'] is not None:
        global_best_found = False
        for result in best_results:
            if set(result['features']) == set(global_best_result['result']['features']):
                global_best_found = True
                break
        
        if not global_best_found:
            logging.info(f"全局最佳结果不在top-{top_k}列表中，将其添加到结果中")
            best_results.append(global_best_result['result'])
            best_results.sort(key=lambda x: x['test_r2'], reverse=True)
            if len(best_results) > top_k:
                best_results = best_results[:top_k]
    
    logging.info(f"全局最佳模型 - 测试集R²: {global_best_result['test_r2']:.4f}")
    logging.info(f"特征: {','.join(global_best_result['result']['features']) if global_best_result['result'] else '无'}")
    
    return best_results[:top_k], global_best_result['result']

def save_model_and_results(final_results, output_dir, global_best_result=None):
    """保存最佳模型和结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确定要保存的最佳结果
    if isinstance(final_results, tuple) and len(final_results) == 2:
        top_results, global_best = final_results
    else:
        top_results = final_results
        global_best = global_best_result
    
    # 如果有全局最佳结果，优先使用它
    if global_best is not None:
        best_result = global_best
        logging.info(f"使用全局最佳模型 - 测试集R²: {best_result['test_r2']:.4f}")
    else:
        # 否则使用top_results中测试集R^2最高的
        best_result = max(top_results, key=lambda x: x['test_r2'])
        logging.info(f"使用top-k结果中的最佳模型 - 测试集R²: {best_result['test_r2']:.4f}")
    
    # 保存最佳模型
    model_file = os.path.join(output_dir, f"best_model_{timestamp}.pkl")
    joblib.dump({
        'model': best_result['model'],
        'scaler': best_result['scaler'],
        'features': best_result['features']
    }, model_file)
    logging.info(f"最佳模型已保存至 {model_file}")
    logging.info(f"最佳模型 - 测试集 R²: {best_result['test_r2']:.4f}")
    
    # 保存所有结果
    if isinstance(top_results, list):
        results_df = pd.DataFrame([
            {
                'rank': i + 1,
                'num_features': r['num_features'],
                'cv_r2': r['cv_r2'] if 'cv_r2' in r else None,
                'test_r2': r['test_r2'],
                'test_rmse': r['test_rmse'] if 'test_rmse' in r else None,
                'test_mae': r['test_mae'] if 'test_mae' in r else None,
                'features': ','.join(r['features']),
                'is_global_best': 1 if r is best_result else 0
            }
            for i, r in enumerate(top_results)
        ])
        
        results_file = os.path.join(output_dir, f"feature_selection_results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        logging.info(f"结果已保存至 {results_file}")
    
    # 绘制最佳模型的特征重要性
    plot_file = os.path.join(output_dir, f"feature_importance_{timestamp}.png")
    plot_feature_importance(best_result['model'], pd.DataFrame(columns=best_result['features']), plot_file)
    
    return model_file, results_file, best_result

def prepare_feature_data(data_file, include_aa=False, use_interaction=False, use_nonlinear=False):
    """准备用于特征选择的数据"""
    # 加载数据
    data = load_data(data_file)
    
    # 准备特征
    X, y, feature_cols = prepare_features(
        data, 
        apply_feature_selection=False,  # 我们自己将进行特征选择
        use_interaction=use_interaction,
        use_nonlinear=use_nonlinear
    )
    
    # 如果不包含氨基酸比例特征，手动过滤它们
    if not include_aa:
        # 找出所有氨基酸比例特征（以'aa_'开头的列）
        aa_cols = [col for col in feature_cols if 'aa_' in col.lower()]
        if aa_cols:
            logging.info(f"排除了{len(aa_cols)}个氨基酸比例特征")
            # 过滤掉氨基酸比例特征
            non_aa_cols = [col for col in feature_cols if col not in aa_cols]
            X = X[non_aa_cols]
            feature_cols = non_aa_cols
    else:
        logging.info("保留氨基酸比例特征用于分析")
    
    return X, y, feature_cols

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用train_rf_model中的功能进行特征选择和模型训练')
    parser.add_argument('--data', type=str, required=True, help='训练数据文件路径 (CSV格式)')
    parser.add_argument('--output', type=str, default='./optimal_models', help='输出目录，用于保存模型和结果')
    parser.add_argument('--min_features', type=int, default=3, help='最小特征数量')
    parser.add_argument('--max_features', type=int, default=15, help='最大特征数量')
    parser.add_argument('--top_k', type=int, default=10, help='返回的顶部结果数量')
    parser.add_argument('--cv', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--include_aa', action='store_true', help='包含氨基酸比例特征')
    parser.add_argument('--interactions', action='store_true', help='创建特征交互项')
    parser.add_argument('--nonlinear', action='store_true', help='应用非线性变换')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    start_time = time.time()
    
    # 获取CPU核心数并显示
    cpu_count = multiprocessing.cpu_count()
    logging.info(f"检测到{cpu_count}个CPU核心，将使用{cpu_count-1}个核心进行并行处理")
    logging.info("已配置多线程支持以加速特征选择过程")
    logging.info("已配置图表中文支持，生成的图表可显示中文标题和标签")
    
    logging.info(f"开始特征选择优化过程...")
    logging.info(f"将选择并保存所有特征组合中测试集R²最高的模型")
    
    # 准备数据
    X, y, feature_cols = prepare_feature_data(
        args.data, 
        include_aa=args.include_aa,
        use_interaction=args.interactions,
        use_nonlinear=args.nonlinear
    )
    logging.info(f"原始特征数量: {len(feature_cols)}")
    
    # 特征选择
    feature_selection_results = feature_selection_by_combination(
        X, y, feature_cols,
        min_features=args.min_features,
        max_features=args.max_features,
        cv=args.cv,
        top_k=args.top_k,
        random_state=args.random_state,
        test_size=args.test_size
    )
    
    # 检查是否有有效的特征组合
    if not feature_selection_results or (isinstance(feature_selection_results, tuple) and not feature_selection_results[0]):
        logging.error("未找到有效的特征组合，请尝试包含更多特征或修改参数。")
        return
    
    # 保存模型和结果
    model_file, results_file, best_result = save_model_and_results(feature_selection_results, args.output)
    
    # 输出最佳特征组合
    logging.info(f"全局最佳特征组合: {', '.join(best_result['features'])}")
    logging.info(f"特征数量: {len(best_result['features'])}")
    logging.info(f"最终全局最高测试集R²: {best_result['test_r2']:.4f}")
    if 'test_rmse' in best_result:
        logging.info(f"测试集 RMSE: {best_result['test_rmse']:.4f}")
    if 'test_mae' in best_result:
        logging.info(f"测试集 MAE: {best_result['test_mae']:.4f}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_str = format_time(elapsed_time)
    logging.info(f"特征选择优化完成，总耗时: {elapsed_str}")
    logging.info(f"最佳模型已保存至: {model_file}")
    logging.info(f"结果摘要已保存至: {results_file}")

if __name__ == "__main__":
    main() 