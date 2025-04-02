import os
import sys
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser
import warnings

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdb2temp_dl.log', mode='a')
    ]
)

# 方法1：使用系统中已有的中文字体(以微软雅黑为例)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 方法2：使用自定义中文字体
# font_path = '您的字体文件路径，例如 C:/Windows/Fonts/simhei.ttf'
# prop = fm.FontProperties(fname=font_path)
# 在需要显示中文的地方使用：plt.title('中文标题', fontproperties=prop)

# 在PDBDataset类之前添加此函数
def extract_additional_features(pdb_path):
    """从analyze_pdb.py提取热稳定性特征"""
    try:
        from analyze_pdb import extract_thermostability_features
        features = extract_thermostability_features(pdb_path)
        # 由于thermostability_features是全局特征，为每个残基赋相同值
        feature_list = []
        for _ in range(1000):  # 最多支持1000个残基
            feature_list.append(np.array(list(features.values())))
        return feature_list
    except Exception as e:
        logging.warning(f"提取额外特征失败: {str(e)}")
        return []

# 定义PDB蛋白质数据集
class PDBDataset(Dataset):
    def __init__(self, pdb_files, temperatures=None, max_atoms=3000, transform=None):
        """
        创建PDB数据集
        
        参数:
            pdb_files: PDB文件路径列表
            temperatures: 对应的最适温度列表 (训练需要，预测时可为None)
            max_atoms: 每个蛋白质最大原子数
            transform: 数据转换函数
        """
        self.pdb_files = pdb_files
        self.temperatures = temperatures
        self.max_atoms = max_atoms
        self.transform = transform
        self.parser = PDBParser(QUIET=True)
        
        # 原子特征映射
        self.atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'H': 4, 'P': 5, 
            'ZN': 6, 'CA': 7, 'MG': 8, 'FE': 9, 'OTHER': 10
        }
        
        # 氨基酸特征映射
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
            'OTHER': 20
        }
    
    def __len__(self):
        return len(self.pdb_files)
    
    def __getitem__(self, idx):
        pdb_path = self.pdb_files[idx]
        try:
            # 获取PDB ID
            pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
            
            # 解析PDB文件
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = self.parser.get_structure(pdb_id, pdb_path)
            
            # 提取残基级特征
            residue_features = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_name = residue.get_resname()
                        if res_name in self.residue_types:
                            # 创建残基特征向量
                            feature_vec = np.zeros(len(self.residue_types) + 5)  # 5个额外特征
                            
                            # 残基类型的one-hot编码
                            feature_vec[self.residue_types[res_name]] = 1
                            
                            # 添加残基中心坐标
                            center = np.zeros(3)
                            atom_count = 0
                            
                            # 计算残基中心坐标
                            for atom in residue:
                                center += atom.get_coord()
                                atom_count += 1
                            
                            if atom_count > 0:
                                center /= atom_count
                                
                                # 残基特征: [残基类型, x, y, z, 原子数]
                                feature_vec[len(self.residue_types):len(self.residue_types)+3] = center
                                feature_vec[len(self.residue_types)+3] = atom_count
                                
                                # 计算残基体积(用原子数近似)
                                feature_vec[len(self.residue_types)+4] = atom_count / 10.0
                                
                                residue_features.append(feature_vec)
            
            # 添加额外特征
            try:
                # 使用analyze_pdb.py中的函数提取更多特征
                thermo_features = extract_additional_features(pdb_path)
                
                # 合并到现有特征中
                for i, feature_vec in enumerate(residue_features):
                    if i < len(thermo_features):
                        feature_vec = np.concatenate([feature_vec, thermo_features[i]])
                    residue_features[i] = feature_vec
            except:
                pass
            
            # 转换为NumPy数组
            features = np.array(residue_features)
            
            # 限制最大残基数
            if len(features) > self.max_atoms:
                features = features[:self.max_atoms]
            elif len(features) == 0:
                # 处理空蛋白质情况
                features = np.zeros((1, len(self.residue_types) + 5))
            
            # 创建特征张量
            features_tensor = torch.tensor(features, dtype=torch.float)
            
            # 使用CNN模型需要添加通道维度
            data = {
                'features': features_tensor,
                'pdb_id': pdb_id
            }
            
            # 添加温度标签（如果有）
            if self.temperatures is not None:
                data['temperature'] = torch.tensor([self.temperatures[idx]], dtype=torch.float)
            
            # 应用转换
            if self.transform:
                data = self.transform(data)
            
            return data
            
        except Exception as e:
            logging.error(f"处理PDB文件 {pdb_path} 时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
            # 返回空数据
            empty_data = {
                'features': torch.zeros((1, len(self.residue_types) + 5), dtype=torch.float),
                'pdb_id': pdb_id
            }
            
            if self.temperatures is not None:
                empty_data['temperature'] = torch.tensor([0.0], dtype=torch.float)
                
            return empty_data

# 定义深度学习模型
class Protein3DCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(Protein3DCNN, self).__init__()
        
        # 增加卷积层数和通道数
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.bn3 = nn.BatchNorm1d(hidden_dim*2)
        
        # 残差连接
        self.residual = nn.Conv1d(input_dim, hidden_dim*2, kernel_size=1)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, output_dim)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # 调整输入维度
        x_in = x.permute(0, 2, 1)
        residual = self.residual(x_in)
        
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.max_pool1d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, residual.shape[2])  # 上采样以匹配残差连接
        
        # 残差连接
        x = x + residual
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 全局池化
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# 加载蛋白质数据和温度
def load_protein_data(pdb_dir, cazy_file=None):
    """
    加载PDB文件和温度数据
    
    参数:
        pdb_dir: PDB文件目录
        cazy_file: 包含温度数据的CSV文件
        
    返回:
        pdb_files: PDB文件路径列表
        temperatures: 对应的温度列表
        pdb_ids: PDB ID列表
    """
    # 收集PDB文件
    if os.path.isdir(pdb_dir):
        pdb_files = []
        for root, dirs, files in os.walk(pdb_dir):
            for file in files:
                if file.lower().endswith(('.pdb', '.ent')):
                    pdb_files.append(os.path.join(root, file))
    elif os.path.isfile(pdb_dir) and pdb_dir.lower().endswith(('.pdb', '.ent')):
        pdb_files = [pdb_dir]
    else:
        raise ValueError(f"无效的PDB路径: {pdb_dir}")
    
    logging.info(f"找到 {len(pdb_files)} 个PDB文件")
    
    # 提取PDB ID
    pdb_ids = [os.path.splitext(os.path.basename(f))[0] for f in pdb_files]
    
    # 加载温度数据（如果有）
    temperatures = None
    if cazy_file and os.path.exists(cazy_file):
        try:
            # 尝试加载温度数据
            temp_data = pd.read_csv(cazy_file)
            
            # 检查是否有Alphafold GenBank Accession列
            if 'Alphafold GenBank Accession' in temp_data.columns and 'Optimal Temperature (°C)' in temp_data.columns:
                # 先用外部脚本合并数据
                logging.info("检测到CAZy数据格式，需要先运行merge_data.py合并数据")
                
                import subprocess
                merge_cmd = ['python', 'merge_data.py', '--cazy', cazy_file]
                logging.info(f"执行命令: {' '.join(merge_cmd)}")
                
                result = subprocess.run(merge_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"合并数据失败: {result.stderr}")
                    raise ValueError("无法合并PDB和温度数据")
                
                # 加载合并后的数据
                merged_files = glob.glob(os.path.join('trainData', 'analyze_pdb_merged_*.csv'))
                if not merged_files:
                    logging.error("未找到merge_data.py生成的合并文件")
                    raise ValueError("未找到合并后的数据文件")
                
                merged_file = max(merged_files, key=os.path.getmtime)
                logging.info(f"使用合并数据文件: {merged_file}")
                
                merged_data = pd.read_csv(merged_file)
                
                # 创建PDB ID到温度的映射
                temp_map = {}
                for _, row in merged_data.iterrows():
                    pdb_id = row['pdb_id']
                    if 'optimal_temperature' in row and pd.notnull(row['optimal_temperature']):
                        try:
                            temp = float(row['optimal_temperature'])
                            temp_map[pdb_id] = temp
                        except:
                            pass
                
                # 获取每个PDB文件的温度
                temperatures = []
                valid_pdb_files = []
                valid_pdb_ids = []
                
                for i, (pdb_file, pdb_id) in enumerate(zip(pdb_files, pdb_ids)):
                    if pdb_id in temp_map:
                        temperatures.append(temp_map[pdb_id])
                        valid_pdb_files.append(pdb_file)
                        valid_pdb_ids.append(pdb_id)
                    else:
                        logging.warning(f"PDB ID {pdb_id} 没有找到对应的温度数据")
                
                # 更新PDB文件列表
                pdb_files = valid_pdb_files
                pdb_ids = valid_pdb_ids
                
                logging.info(f"成功加载 {len(temperatures)} 个带温度数据的PDB文件")
            
            # 常规格式温度数据
            elif 'pdb_id' in temp_data.columns and 'optimal_temperature' in temp_data.columns:
                # 创建PDB ID到温度的映射
                temp_map = {}
                for _, row in temp_data.iterrows():
                    pdb_id = row['pdb_id']
                    if pd.notnull(row['optimal_temperature']):
                        try:
                            temp = float(row['optimal_temperature'])
                            temp_map[pdb_id] = temp
                        except:
                            pass
                
                # 获取每个PDB文件的温度
                temperatures = []
                valid_pdb_files = []
                valid_pdb_ids = []
                
                for i, (pdb_file, pdb_id) in enumerate(zip(pdb_files, pdb_ids)):
                    if pdb_id in temp_map:
                        temperatures.append(temp_map[pdb_id])
                        valid_pdb_files.append(pdb_file)
                        valid_pdb_ids.append(pdb_id)
                    else:
                        logging.warning(f"PDB ID {pdb_id} 没有找到对应的温度数据")
                
                # 更新PDB文件列表
                pdb_files = valid_pdb_files
                pdb_ids = valid_pdb_ids
                
                logging.info(f"成功加载 {len(temperatures)} 个带温度数据的PDB文件")
            else:
                logging.error(f"温度数据文件格式无效，需要包含 'pdb_id' 和 'optimal_temperature' 列")
                raise ValueError("温度数据文件格式无效")
                
        except Exception as e:
            logging.error(f"加载温度数据时出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise ValueError("无法加载温度数据")
    
    return pdb_files, temperatures, pdb_ids

# 训练模型
def train_model(pdb_dir, cazy_file, output_dir='./models_dl', epochs=50, batch_size=8, 
                learning_rate=0.001, hidden_dim=128, test_size=0.2, device='cuda'):
    """
    训练端到端的深度学习模型
    
    参数:
        pdb_dir: PDB文件目录
        cazy_file: 温度数据文件
        output_dir: 模型输出目录
        epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        hidden_dim: 隐藏层大小
        test_size: 测试集比例
        device: 训练设备 ('cuda'/'cpu')
    
    返回:
        训练后的模型和训练历史
    """
    # 设置设备
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA不可用，使用CPU代替")
        device = 'cpu'
    
    device = torch.device(device)
    logging.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载数据
        pdb_files, temperatures, pdb_ids = load_protein_data(pdb_dir, cazy_file)
        
        if not temperatures:
            logging.error("训练需要温度数据")
            return None, None
        
        # 添加这些代码来记录温度统计信息
        temp_min = min(temperatures)
        temp_max = max(temperatures)
        temp_mean = np.mean(temperatures)
        temp_std = np.std(temperatures)

        logging.info(f"原始温度统计: 最小值={temp_min:.2f}, 最大值={temp_max:.2f}, 平均值={temp_mean:.2f}, 标准差={temp_std:.2f}")

        # 检查是否需要规范化（温度范围很大时）
        if temp_max - temp_min > 100:
            # 记录是否进行了温度规范化
            do_normalize = True
            # 保存规范化参数
            temp_scale = 100.0  # 规范化因子
            
            # 规范化温度数据
            temperatures = [t / temp_scale for t in temperatures]
            
            logging.info(f"温度数据已规范化，缩放因子: {temp_scale}")
        else:
            do_normalize = False
            temp_scale = 1.0
            logging.info("温度数据未规范化，使用原始值")
        
        # 训练测试分割
        n_samples = len(pdb_files)
        indices = np.random.permutation(n_samples)
        test_size_n = int(n_samples * test_size)
        test_indices = indices[:test_size_n]
        train_indices = indices[test_size_n:]
        
        train_files = [pdb_files[i] for i in train_indices]
        train_temps = [temperatures[i] for i in train_indices]
        
        test_files = [pdb_files[i] for i in test_indices]
        test_temps = [temperatures[i] for i in test_indices]
        
        logging.info(f"训练集大小: {len(train_files)}，测试集大小: {len(test_files)}")
        
        # 创建数据集
        train_dataset = PDBDataset(train_files, train_temps)
        test_dataset = PDBDataset(test_files, test_temps)
        
        # 使用数据加载器加速训练
        def collate_fn(batch):
            """自定义collate函数处理不同长度的序列"""
            features = [item['features'] for item in batch]
            pdb_ids = [item['pdb_id'] for item in batch]
            
            # 找到最大长度
            max_len = max(feat.shape[0] for feat in features)
            
            # 使用填充使所有序列相同长度
            padded_features = []
            for feat in features:
                if feat.shape[0] < max_len:
                    padding = torch.zeros((max_len - feat.shape[0], feat.shape[1]), dtype=feat.dtype)
                    padded_feat = torch.cat([feat, padding], dim=0)
                else:
                    padded_feat = feat
                padded_features.append(padded_feat)
            
            # 堆叠为批次
            stacked_features = torch.stack(padded_features)
            
            result = {
                'features': stacked_features,
                'pdb_ids': pdb_ids
            }
            
            # 添加温度标签（如果有）
            if 'temperature' in batch[0]:
                temperatures = torch.stack([item['temperature'] for item in batch])
                result['temperature'] = temperatures
            
            return result
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # 获取第一个样本以确定实际特征维度
        sample_data = train_dataset[0]
        actual_input_dim = sample_data['features'].shape[1]
        
        # 使用实际特征维度创建模型
        model = Protein3DCNN(actual_input_dim, hidden_dim).to(device)
        logging.info(f"创建模型: Protein3DCNN，输入维度: {actual_input_dim}，隐藏维度: {hidden_dim}")
        
        # 定义优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        criterion = nn.MSELoss()
        
        # 使用余弦退火学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # 训练模型
        train_losses = []
        val_losses = []
        val_rmses = []
        best_val_rmse = float('inf')
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for batch in train_loader:
                # 将数据移到设备
                features = batch['features'].to(device)
                targets = batch['temperature'].to(device)
                
                # 前向传播
                pred = model(features)
                loss = criterion(pred, targets)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_train_loss = epoch_loss / batch_count
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            batch_count = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in test_loader:
                    features = batch['features'].to(device)
                    targets = batch['temperature'].to(device)
                    
                    pred = model(features)
                    loss = criterion(pred, targets)
                    
                    val_loss += loss.item()
                    batch_count += 1
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            avg_val_loss = val_loss / batch_count
            val_losses.append(avg_val_loss)
            
            # 计算RMSE
            rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets))**2))
            val_rmses.append(rmse)
            
            # 更新学习率
            scheduler.step()
            
            # 保存最佳模型
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
                logging.info(f"保存最佳模型，RMSE: {rmse:.2f}")
            
            logging.info(f"Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, RMSE: {rmse:.2f}")
        
        # 加载最佳模型
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_rmses, label='验证RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        
        # 评估最终模型
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(device)
                targets = batch['temperature'].to(device)
                
                pred = model(features)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算评估指标
        rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets))**2))
        mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
        
        logging.info(f"最终测试结果 - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        # 绘制预测vs实际图
        plt.figure(figsize=(8, 6))
        plt.scatter(all_targets, all_preds, alpha=0.6)
        
        # 添加拟合线
        min_val = min(min(all_targets), min(all_preds))
        max_val = max(max(all_targets), max(all_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('实际温度 (°C)')
        plt.ylabel('预测温度 (°C)')
        plt.title(f'预测vs实际温度 (RMSE: {rmse:.2f}, MAE: {mae:.2f})')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predictions.png'))
        
        # 保存训练历史
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_rmses': val_rmses,
            'final_rmse': rmse,
            'final_mae': mae
        }
        
        # 将规范化信息保存到模型中
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': actual_input_dim,
            'hidden_dim': hidden_dim,
            'history': history,
            'temp_normalization': {
                'applied': do_normalize,
                'scale_factor': temp_scale,
                'min': temp_min,
                'max': temp_max,
                'mean': temp_mean,
                'std': temp_std
            }
        }, os.path.join(output_dir, 'complete_model.pt'))
        
        logging.info(f"模型和训练历史已保存到: {output_dir}")
        
        return model, history
        
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

# 预测函数
def predict(pdb_dir, model_path, output_dir='./results'):
    """
    使用训练好的模型进行预测
    
    参数:
        pdb_dir: PDB文件或目录
        model_path: 模型文件路径
        output_dir: 输出目录
    
    返回:
        预测结果
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用设备: {device}")
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint['hidden_dim']
        
        # 创建数据集并获取样本特征大小
        pdb_files, _, pdb_ids = load_protein_data(pdb_dir)
        logging.info(f"加载了 {len(pdb_files)} 个PDB文件")
        
        dataset = PDBDataset(pdb_files)
        sample_data = dataset[0]
        actual_input_dim = sample_data['features'].shape[1]
        
        # 检查输入维度是否匹配
        if input_dim != actual_input_dim:
            logging.warning(f"模型期望的输入维度({input_dim})与实际数据维度({actual_input_dim})不匹配")
            logging.warning("将使用实际数据维度创建新模型并尝试适配权重")
            
            # 如果维度不匹配，创建新模型并尝试加载兼容的权重
            model = Protein3DCNN(actual_input_dim, hidden_dim).to(device)
            
            # 尝试加载兼容的权重
            try:
                # 创建新的状态字典，保留匹配的层
                state_dict = checkpoint['model_state_dict']
                model_dict = model.state_dict()
                
                # 过滤掉不匹配的权重
                filtered_dict = {k: v for k, v in state_dict.items() 
                                if k in model_dict and v.shape == model_dict[k].shape}
                
                # 加载过滤后的权重
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict, strict=False)
                logging.info("成功加载部分权重")
            except Exception as e:
                logging.error(f"加载权重失败: {e}")
                logging.warning("将使用随机初始化的模型")
        else:
            # 维度匹配时直接使用原模型
            model = Protein3DCNN(input_dim, hidden_dim).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        
        logging.info(f"模型已加载: {model_path}")
        
        # 定义collate函数
        def collate_fn(batch):
            features = [item['features'] for item in batch]
            pdb_ids = [item['pdb_id'] for item in batch]
            
            # 找到最大长度
            max_len = max(feat.shape[0] for feat in features)
            
            # 使用填充
            padded_features = []
            for feat in features:
                if feat.shape[0] < max_len:
                    padding = torch.zeros((max_len - feat.shape[0], feat.shape[1]), dtype=feat.dtype)
                    padded_feat = torch.cat([feat, padding], dim=0)
                else:
                    padded_feat = feat
                padded_features.append(padded_feat)
            
            return {
                'features': torch.stack(padded_features),
                'pdb_ids': pdb_ids
            }
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # 进行预测
        predictions = []
        pred_pdb_ids = []
        
        # 加载温度规范化信息
        temp_norm_info = checkpoint.get('temp_normalization', {'applied': False, 'scale_factor': 1.0})
        logging.info(f"温度规范化信息: {temp_norm_info}")
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(device)
                pdb_ids = batch['pdb_ids']
                
                pred = model(features)
                
                for i, p in enumerate(pred.cpu().numpy()):
                    p_value = float(p.item())  # 正确提取标量值
                    
                    # 如果训练时应用了规范化，现在反向应用
                    if temp_norm_info['applied']:
                        p_value = p_value * temp_norm_info['scale_factor']
                    
                    predictions.append(p_value)
                    pred_pdb_ids.append(pdb_ids[i])
                    logging.info(f"预测 {pdb_ids[i]}: {p_value:.2f}°C")
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'pdb_id': pred_pdb_ids,
            'predicted_temperature': predictions
        })
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f'predictions_{timestamp}.csv')
        results.to_csv(csv_path, index=False)
        logging.info(f"预测结果已保存到: {csv_path}")
        
        # 绘制预测结果
        plt.figure(figsize=(12, 6))
        
        # 按温度排序
        sorted_indices = np.argsort(predictions)
        sorted_ids = [pred_pdb_ids[i] for i in sorted_indices]
        sorted_temps = [predictions[i] for i in sorted_indices]
        
        # 使用不同颜色标记高温和低温
        colors = ['red' if temp >= 60 else 'blue' for temp in sorted_temps]
        
        # 绘制温度条形图
        plt.bar(range(len(sorted_ids)), sorted_temps, color=colors, alpha=0.7)
        
        # 添加高温阈值线
        plt.axhline(y=60, color='r', linestyle='--', alpha=0.7)
        
        # 添加标签和标题
        plt.xlabel('蛋白质索引')
        plt.ylabel('预测温度 (°C)')
        plt.title('蛋白质温度预测结果')
        
        # x轴标签太多时，只显示部分
        if len(sorted_ids) > 30:
            plt.xticks(range(0, len(sorted_ids), len(sorted_ids)//20), 
                     [sorted_ids[i] for i in range(0, len(sorted_ids), len(sorted_ids)//20)], 
                     rotation=90)
        else:
            plt.xticks(range(len(sorted_ids)), sorted_ids, rotation=90)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='高温蛋白 (≥60°C)'),
            Patch(facecolor='blue', alpha=0.7, label='低温蛋白 (<60°C)')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'temperature_predictions_{timestamp}.png'))
        
        return results
        
    except Exception as e:
        logging.error(f"预测过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

# 命令行解析
def parse_args():
    parser = argparse.ArgumentParser(description='端到端PDB蛋白质温度预测')
    
    # 模式选择
    parser.add_argument('--train', action='store_true', help='训练模式')
    parser.add_argument('--predict', action='store_true', help='预测模式')
    
    # 训练参数
    parser.add_argument('--pdb_dir', type=str, help='PDB文件目录')
    parser.add_argument('--cazy_file', type=str, help='温度数据文件')
    parser.add_argument('--output', type=str, default='./models_dl', help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层大小')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备 (cuda/cpu)')
    
    # 预测参数
    parser.add_argument('--model', type=str, help='预测模式下的模型路径')
    parser.add_argument('--output_dir', type=str, default='./results', help='预测结果输出目录')
    
    return parser.parse_args()

# 主函数
def main():
    args = parse_args()
    
    if args.train:
        if not args.pdb_dir:
            logging.error("训练模式需要指定 --pdb_dir")
            return
        
        if not args.cazy_file:
            logging.error("未指定温度数据文件 (--cazy_file)，训练可能会失败")
        
        logging.info("开始训练模式")
        train_model(
            args.pdb_dir, 
            args.cazy_file, 
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_dim=args.hidden_dim,
            test_size=args.test_size,
            device=args.device
        )
    
    elif args.predict:
        if not args.pdb_dir:
            logging.error("预测模式需要指定 --pdb_dir")
            return
        
        if not args.model:
            logging.error("预测模式需要指定 --model")
            return
        
        logging.info("开始预测模式")
        predict(args.pdb_dir, args.model, output_dir=args.output_dir)
    
    else:
        logging.error("请指定 --train 或 --predict 模式")

def train_ensemble(pdb_dir, cazy_file, n_models=5, **kwargs):
    models = []
    for i in range(n_models):
        logging.info(f"训练模型 {i+1}/{n_models}")
        model, _ = train_model(pdb_dir, cazy_file, **kwargs)
        models.append(model)
    return models

def ensemble_predict(models, data):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(data)
            predictions.append(pred)
    # 平均所有模型的预测结果
    return torch.mean(torch.stack(predictions), dim=0)

if __name__ == "__main__":
    main()