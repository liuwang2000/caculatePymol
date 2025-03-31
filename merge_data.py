import os
import csv
import glob
import shutil
import re
import logging
import argparse
from datetime import datetime

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def find_latest_analyze_file(directory, specific_file=None):
    """查找目录中最新的analyze_pdb文件或指定文件"""
    if specific_file and os.path.exists(os.path.join(directory, specific_file)):
        return os.path.join(directory, specific_file)
    
    pattern = os.path.join(directory, 'analyze_pdb_*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"在{directory}中未找到analyze_pdb文件")
    
    # 根据文件名中的时间戳排序
    try:
        # 提取出日期和时间，并处理匹配失败的情况
        def extract_timestamp(filename):
            match = re.search(r'analyze_pdb_(\d+)_(\d+)\.csv', filename)
            if match:
                return match.group(1) + match.group(2)  # 合并日期和时间部分
            else:
                # 如果匹配失败，返回一个默认的低值
                return "00000000000000"
                
        latest_file = max(files, key=extract_timestamp)
        return latest_file
    except Exception as e:
        logging.warning(f"按时间戳排序失败: {str(e)}，使用最后修改时间排序")
        # 如果正则匹配失败，使用文件的修改时间作为备选方案
        latest_file = max(files, key=os.path.getmtime)
        return latest_file

def find_cazy_file(directory, specific_file=None):
    """查找目录中的cazy数据文件"""
    if specific_file:
        # 如果指定了具体文件
        file_path = os.path.join(directory, specific_file)
        if os.path.exists(file_path):
            return file_path
            
        # 如果指定的是完整路径
        if os.path.exists(specific_file):
            return specific_file
    
    # 模糊匹配以cazy开头的文件
    pattern = os.path.join(directory, 'cazy*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"在{directory}中未找到cazy*文件")
    
    # 优先选择cazy-cleandata-compare.csv，如果存在的话
    for file in files:
        if 'cleandata-compare' in file:
            return file
    
    # 否则返回第一个匹配的文件
    return files[0]

def clean_old_files(directory, pattern='analyze_pdb_*.csv'):
    """删除目录中指定模式的旧文件"""
    files = glob.glob(os.path.join(directory, pattern))
    for file in files:
        try:
            os.remove(file)
            logging.info(f"已删除: {file}")
        except Exception as e:
            logging.error(f"删除{file}失败: {str(e)}")

def extract_accession_from_pdb_id(pdb_id):
    """尝试从PDB ID中提取可能的GenBank Accession"""
    # 示例：尝试几种不同的格式匹配
    # 1. 直接匹配
    possible_accessions = [pdb_id]
    
    # 2. 如果PDB ID有特定前缀/后缀模式，尝试去除
    if "_" in pdb_id:
        parts = pdb_id.split("_")
        possible_accessions.append(parts[0])  # 取第一部分
        possible_accessions.append("_".join(parts[1:]))  # 取余下部分
    
    # 3. 如果是字母+数字的格式，尝试提取
    match = re.match(r'([A-Za-z]+)(\d+.*)', pdb_id)
    if match:
        possible_accessions.append(match.group(2))  # 数字部分
    
    return possible_accessions

def create_accession_to_pdb_mapping(cazy_file, analyze_file=None):
    """创建GenBank Accession到PDB ID的映射"""
    mapping = {}
    pdb_patterns = {}
    
    try:
        # 首先读取analyze_pdb文件中的所有PDB ID
        output_dir = os.path.join(os.getcwd(), 'output')
        latest_analyze_file = find_latest_analyze_file(output_dir, analyze_file)
        
        pdb_ids = set()
        with open(latest_analyze_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pdb_id = row['pdb_id'].strip()
                pdb_ids.add(pdb_id)
                
                # 为每个PDB ID创建可能的匹配模式
                for pattern in extract_accession_from_pdb_id(pdb_id):
                    if pattern:
                        pdb_patterns[pattern] = pdb_id
        
        # 然后读取CAZY数据，尝试匹配
        with open(cazy_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                accession = row.get('Alphafold GenBank Accession', '').strip()
                if not accession:
                    continue
                
                # 尝试直接匹配
                if accession in pdb_ids:
                    mapping[accession] = accession
                    continue
                
                # 尝试使用之前生成的模式匹配
                for pattern, pdb_id in pdb_patterns.items():
                    if accession == pattern or accession in pattern or pattern in accession:
                        mapping[accession] = pdb_id
                        break
                
                # 默认情况：假设没有匹配成功，将Accession映射到自身
                # 这样merge_data函数中的直接匹配逻辑仍然可以工作
                if accession not in mapping:
                    mapping[accession] = accession
                    
    except Exception as e:
        logging.error(f"创建映射关系时出错: {str(e)}")
    
    return mapping

def merge_data(cazy_file=None, analyze_file=None, output_dir=None, train_data_dir=None):
    """合并CAZY数据和蛋白质分析数据"""
    try:
        # 设置默认目录
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'output')
        if train_data_dir is None:
            train_data_dir = os.path.join(os.getcwd(), 'trainData')
            
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(train_data_dir, exist_ok=True)
        
        # 查找CAZY文件
        actual_cazy_file = find_cazy_file(train_data_dir, cazy_file)
        logging.info(f"使用CAZY数据文件: {actual_cazy_file}")
        
        # 获取analyze_pdb文件
        latest_analyze_file = find_latest_analyze_file(output_dir, analyze_file)
        logging.info(f"使用分析文件: {latest_analyze_file}")
        
        # 创建Accession到PDB ID的映射
        accession_to_pdb = create_accession_to_pdb_mapping(actual_cazy_file, analyze_file)
        logging.info(f"创建了{len(accession_to_pdb)}个Accession到PDB的映射")
        
        # 读取CAZY数据
        cazy_data = {}
        with open(actual_cazy_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                accession = row.get('Alphafold GenBank Accession', '').strip()
                if accession:
                    cazy_data[accession] = {
                        'sequence': row.get('Sequence', ''),
                        'optimal_temperature': row.get('Optimal Temperature (°C)', '')
                    }
        logging.info(f"已读取{len(cazy_data)}条CAZY数据")
        
        # 反向创建PDB ID到Accession的映射
        pdb_to_accession = {}
        for acc, pdb in accession_to_pdb.items():
            if pdb not in pdb_to_accession:
                pdb_to_accession[pdb] = acc
            # 如果有多个Accession映射到同一个PDB ID，保留最后一个
        
        # 读取并更新analyze_pdb数据
        analyze_data = []  # 存储匹配成功的数据
        failed_data = []  # 存储匹配失败的数据
        with open(latest_analyze_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames + ['sequence', 'optimal_temperature']
            
            for row in reader:
                pdb_id = row['pdb_id'].strip()
                found_match = False
                row_with_fields = row.copy()  # 创建行的副本用于添加字段
                row_with_fields['sequence'] = ''
                row_with_fields['optimal_temperature'] = ''
                
                # 方法1: 使用映射查找对应的Accession
                accession = pdb_to_accession.get(pdb_id)
                if accession and accession in cazy_data:
                    row_with_fields['sequence'] = cazy_data[accession]['sequence']
                    row_with_fields['optimal_temperature'] = cazy_data[accession]['optimal_temperature']
                    found_match = True
                    logging.info(f"通过映射匹配: PDB={pdb_id} -> Accession={accession}")
                
                # 方法2: 直接匹配尝试
                if not found_match and pdb_id in cazy_data:
                    row_with_fields['sequence'] = cazy_data[pdb_id]['sequence']
                    row_with_fields['optimal_temperature'] = cazy_data[pdb_id]['optimal_temperature']
                    found_match = True
                    logging.info(f"直接匹配成功: PDB={pdb_id}")
                
                # 方法3: 尝试提取Accession的各种变体
                if not found_match:
                    possible_accessions = extract_accession_from_pdb_id(pdb_id)
                    for possible_acc in possible_accessions:
                        if possible_acc in cazy_data:
                            row_with_fields['sequence'] = cazy_data[possible_acc]['sequence']
                            row_with_fields['optimal_temperature'] = cazy_data[possible_acc]['optimal_temperature']
                            found_match = True
                            logging.info(f"变体匹配成功: PDB={pdb_id} -> 可能的Accession={possible_acc}")
                            break
                
                # 根据匹配结果将行添加到相应的列表
                if found_match:
                    # 只有匹配成功的行才添加到成功数据列表
                    analyze_data.append(row_with_fields)
                else:
                    # 匹配失败的行只添加到失败数据列表
                    logging.warning(f"未找到匹配: PDB={pdb_id}")
                    failed_data.append(row_with_fields)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 写入合并后的成功数据
        output_file = os.path.join(output_dir, f'analyze_pdb_merged_{timestamp}.csv')
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analyze_data)
        logging.info(f"合并数据已保存至: {output_file}")
        logging.info(f"成功文件包含{len(analyze_data)}条匹配成功的记录")
        
        # 写入匹配失败的数据
        if failed_data:
            failed_file = os.path.join(output_dir, f'analyze_pdb_failed_{timestamp}.csv')
            with open(failed_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(failed_data)
            logging.info(f"匹配失败数据已保存至: {failed_file}")
            logging.info(f"失败文件包含{len(failed_data)}条匹配失败的记录")
        else:
            failed_file = None
            logging.info("所有记录均匹配成功，未生成失败文件")
        
        # 清理trainData中的旧文件
        clean_old_files(train_data_dir, 'analyze_pdb_merged_*.csv')
        clean_old_files(train_data_dir, 'analyze_pdb_failed_*.csv')
        
        # 复制最新的merged文件到trainData
        dest_file = os.path.join(train_data_dir, os.path.basename(output_file))
        shutil.copy2(output_file, dest_file)
        logging.info(f"已复制最新成功数据到: {dest_file}")
        
        # 如果存在失败文件，也复制到trainData
        if failed_file:
            dest_failed_file = os.path.join(train_data_dir, os.path.basename(failed_file))
            shutil.copy2(failed_file, dest_failed_file)
            logging.info(f"已复制失败数据到: {dest_failed_file}")
        
        return True
    except Exception as e:
        import traceback
        logging.error(f"合并数据失败: {str(e)}")
        logging.error(f"错误详情: {traceback.format_exc()}")
        return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='合并CAZY数据和蛋白质分析数据')
    parser.add_argument('--cazy', default=None,
                        help='CAZY数据文件路径 (默认: 自动查找trainData目录下的cazy*.csv文件)')
    parser.add_argument('--analyze', default=None,
                        help='特定的analyze_pdb文件名 (默认: 自动查找最新)')
    parser.add_argument('--output', default=None,
                        help='输出目录 (默认: ./output/)')
    parser.add_argument('--train', default=None,
                        help='训练数据目录 (默认: ./trainData/)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.info("开始处理数据...")
    if merge_data(args.cazy, args.analyze, args.output, args.train):
        logging.info("数据处理完成")
    else:
        logging.error("数据处理失败") 