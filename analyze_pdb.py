import os
import sys
import csv
import re
import logging
import time
from datetime import datetime
from collections import defaultdict
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.SASA import ShrakeRupley
import pymol
from pymol import cmd

# 配置日志记录
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def parse_residue(res_str):
    """解析残基字符串为(名称, 编号)"""
    match = re.match(r'^([A-Za-z]+)(\d+)$', res_str)
    if not match:
        raise ValueError(f"Invalid residue format: {res_str}")
    return (match.group(1).upper(), match.group(2))

# 删除整个 calculate_catalytic_distance 函数

def analyze_pdb_features(pdb_path):  # 移除 catalytic_residues 参数
    """PyMOL特征分析"""
    features = {}
    try:
        # 初始化PyMOL环境
        cmd.reinitialize()
        obj_name = os.path.splitext(os.path.basename(pdb_path))[0]
        cmd.load(pdb_path, obj_name)
        cmd.set('solvent_radius', 1.4)
      
        # 计算溶剂可及性
        cmd.get_area(obj_name, load_b=1)
      
        # 二硫键
        cysteines = cmd.find_pairs('(resn cys & name sg)', '(resn cys & name sg)', 
                                 mode=1, cutoff=2.2)
        features['disulfide_bonds'] = len(cysteines)
      
        # 表面极性比例
        cmd.set('dot_solvent', 1)  # 开启溶剂点计算
        cmd.set('dot_density', 3)  # 提高点密度
        cmd.select('surface', 'b > 10')
        polar = cmd.count_atoms('surface & (resn arg+lys+asp+glu+asn+gln+his+ser+thr+tyr)')
        total = cmd.count_atoms('surface')
        features['surface_polar_ratio'] = polar/total if total >0 else 0
      
        # 氢键网络（优化版）
        donors = '(resn arg+lys+his+asn+gln+ser+thr+tyr+trp & name n+od1+od2+oe1+oe2)'
        acceptors = '(resn asp+glu+asn+gln+ser+thr+his+tyr & name o+od1+od2+oe1+oe2)'
        hbonds = cmd.find_pairs(donors, acceptors, mode=1, cutoff=3.2)
        features['hydrogen_bonds'] = len([p for p in hbonds if p[0][1] != p[1][1]])
      
        # 疏水核心
        cmd.select('hydrophobic_core', 'resn ala+val+ile+leu+phe+trp+pro+met & b < 10')
        contacts = cmd.find_pairs('hydrophobic_core', 'hydrophobic_core', cutoff=5.0)
        features['hydrophobic_contacts'] = len(contacts)
      
        # 盐桥
        positive = 'resn arg+lys'
        negative = 'resn asp+glu'
        salt_bridges = cmd.find_pairs(positive, negative, mode=2, cutoff=4.0)
        features['salt_bridges'] = len(salt_bridges)
      
            
 
        cmd.delete(obj_name)
    except Exception as e:
        logging.error(f"PyMOL分析失败: {str(e)}")
    return features

def extract_biopython_features(pdb_path):
    """Biopython特征提取（移除DSSP依赖）"""
    features = {}
    parser = PDBParser()
    
    # 添加警告过滤
    import warnings
    from Bio import BiopythonWarning
  
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', BiopythonWarning)
            structure = parser.get_structure('protein', pdb_path)
      
        # 氨基酸组成
        aa_count = defaultdict(int)
        for residue in structure.get_residues():
            resname = residue.get_resname()
            if resname not in ['HOH', 'WAT']:
                aa_count[resname] += 1
        total = sum(aa_count.values()) or 1
        features['aa_composition'] = {k: v/total for k, v in aa_count.items()}
      
        # SASA计算
        sasa_calculator = ShrakeRupley()
        sasa_calculator.compute(structure, level='R')
        hydrophobic = ['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'PRO', 'MET', 'TRP']
        sasa_data = {'total': 0.0, 'count': 0}
        for residue in structure.get_residues():
            if residue.get_resname() in hydrophobic:
                sasa_data['total'] += residue.sasa
                sasa_data['count'] += 1
        features['hydrophobic_sasa'] = sasa_data['total']
        features['mean_sasa'] = sasa_data['total']/sasa_data['count'] if sasa_data['count']>0 else 0
      
        # 二级结构（改用PyMOL分析）
        features.update(calculate_secondary_structure(pdb_path))
      
    except Exception as e:
        logging.error(f"Biopython分析失败: {str(e)}")
      
    return features

def calculate_secondary_structure(pdb_path):
    """使用PyMOL进行二级结构分析（按残基比例）"""
    result = {'helix':0.0, 'sheet':0.0, 'loop':0.0}
    obj_name = os.path.splitext(os.path.basename(pdb_path))[0]
    
    try:
        cmd.reinitialize()
        cmd.load(pdb_path, obj_name)
        cmd.dss()  # PyMOL内置二级结构分析
        
        # 按残基统计二级结构
        ss_counter = {'helix': set(), 'sheet': set(), 'loop': set()}
        cmd.iterate(f"{obj_name} and ss h", 
                   'ss_counter["helix"].add(f"{model}/{segi}/{chain}/{resi}")', 
                   space={'ss_counter': ss_counter})
        cmd.iterate(f"{obj_name} and ss s", 
                   'ss_counter["sheet"].add(f"{model}/{segi}/{chain}/{resi}")', 
                   space={'ss_counter': ss_counter})
        cmd.iterate(f"{obj_name} and ss l", 
                   'ss_counter["loop"].add(f"{model}/{segi}/{chain}/{resi}")', 
                   space={'ss_counter': ss_counter})
        logging.info(f"helix: {len(ss_counter['helix'])}")
        logging.info(f"sheet: {len(ss_counter['sheet'])}")
        logging.info(f"loop: {len(ss_counter['loop'])}")

        # 计算残基总数（排除水分子）
        total_resi = set()
        cmd.iterate(f"{obj_name} and not resn HOH,WAT", 
                   'total_resi.add(f"{model}/{segi}/{chain}/{resi}")',
                   space={'total_resi': total_resi})
        total = len(total_resi) or 1
        
        logging.info(f"total: {total}")
        # 计算比例
        result.update({
            'helix': len(ss_counter['helix']) / total,
            'sheet': len(ss_counter['sheet']) / total,
            'loop' : len(ss_counter['loop']) / total
        })
        
        cmd.delete(obj_name)
    except Exception as e:
        logging.error(f"PyMOL二级结构分析失败: {str(e)}")
    
    return result

def process_directory(input_dir):  # 移除 catalytic_residues 参数
    """批量处理PDB目录"""
    # 创建输出目录
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f'analyze_pdb_{timestamp}.csv')  # 修改输出路径
    
    # 动态收集所有可能的氨基酸字段
    all_aa_fields = set()
    reports = []

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.pdb'):
            continue
          
        pdb_path = os.path.join(input_dir, filename)
        pdb_id = os.path.splitext(filename)[0]
        logging.info(f"正在处理: {pdb_id}")
      
        # 提取特征
        report = {'pdb_id': pdb_id}
        report.update(analyze_pdb_features(pdb_path))
        report.update(extract_biopython_features(pdb_path))
        
        # 计算蛋白质中的氨基酸总数
        try:
            cmd.reinitialize()
            obj_name = os.path.splitext(os.path.basename(pdb_path))[0]
            cmd.load(pdb_path, obj_name)
            total_residues = cmd.count_atoms(f"{obj_name} and name ca")
            
            # 归一化相关指标
            if total_residues > 0:
                report['hydrogen_bonds_norm'] = report['hydrogen_bonds'] / total_residues
                report['hydrophobic_contacts_norm'] = report['hydrophobic_contacts'] / total_residues
                report['salt_bridges_norm'] = report['salt_bridges'] / total_residues
                report['hydrophobic_sasa_norm'] = report['hydrophobic_sasa'] / total_residues
                # mean_sasa已经是平均值，但如果需要按所有氨基酸平均：
                # report['mean_sasa_all'] = report['hydrophobic_sasa'] / total_residues
            
            cmd.delete(obj_name)
        except Exception as e:
            logging.error(f"归一化计算失败: {str(e)}")
      
        # 展平氨基酸组成并收集字段
        aa_comp = report.pop('aa_composition', {})
        for aa in aa_comp.keys():
            all_aa_fields.add(f'aa_{aa}')
        for aa, val in aa_comp.items():
            report[f'aa_{aa}'] = round(val, 4)
          
        reports.append(report)

    # 合并字段到fieldnames，添加新的归一化字段
    fieldnames = [
        'pdb_id', 'disulfide_bonds', 'surface_polar_ratio', 'hydrogen_bonds', 'hydrogen_bonds_norm',
        'hydrophobic_contacts', 'hydrophobic_contacts_norm', 'salt_bridges', 'salt_bridges_norm',
        'hydrophobic_sasa', 'hydrophobic_sasa_norm', 'mean_sasa', 'helix', 'sheet', 'loop'
    ] + sorted(all_aa_fields)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            # 填充缺失的氨基酸字段为0
            for aa_field in all_aa_fields:
                if aa_field not in report:
                    report[aa_field] = 0.0
            writer.writerow(report)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python protein_analyzer.py <input_dir>")  # 简化使用说明
        sys.exit(1)
      
    process_directory(sys.argv[1])  # 移除第二个参数
    logging.info(f"分析完成，结果已保存至 output/ 目录")  # 修改日志输出