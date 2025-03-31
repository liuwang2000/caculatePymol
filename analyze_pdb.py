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
import argparse
import numpy as np

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

def extract_thermostability_features(pdb_path):
    """提取热稳定性相关特征 - 简化优化版本，专注于最关键的高温蛋白特征"""
    features = {}
    obj_name = os.path.splitext(os.path.basename(pdb_path))[0]
    
    try:
        cmd.reinitialize()
        cmd.load(pdb_path, obj_name)
        
        # 获取蛋白质的残基总数
        total_residues = cmd.count_atoms(f"{obj_name} and name ca")
        if total_residues == 0:
            logging.error(f"无法在{pdb_path}中找到氨基酸残基")
            return features
        
        # 1. 离子对网络密度（重要高温特征）
        positive = 'resn arg+lys+his'
        negative = 'resn asp+glu'
        ion_pairs = cmd.find_pairs(positive, negative, mode=2, cutoff=6.0)
        features['ion_pair_density'] = len(ion_pairs) / total_residues
        
        # 2. 核心残基比例
        cmd.select('core', f"{obj_name} and b < 10")
        core_residues = cmd.count_atoms('core and name ca')
        features['core_residue_ratio'] = core_residues / total_residues
        
        # 3. 表面电荷分布
        cmd.select('surface_charged', f"{obj_name} and b > 30 and (resn arg+lys+asp+glu)")
        charged_residues = cmd.count_atoms('surface_charged and name ca')
        surface_residues = cmd.count_atoms(f"{obj_name} and b > 30 and name ca")
        features['surface_charge_ratio'] = charged_residues / (surface_residues or 1)
        
        # 4. IVYWREL指数 - 高温蛋白质的关键指标
        ivywrel = 'resn ile+val+tyr+trp+arg+glu+leu'
        ivywrel_count = cmd.count_atoms(f"{obj_name} and {ivywrel} and name ca")
        features['ivywrel_index'] = ivywrel_count / total_residues
        
        # 5. 紧密氢键网络
        donors = '(resn arg+lys+his+asn+gln+ser+thr+tyr+trp & name n+od1+od2+oe1+oe2)'
        acceptors = '(resn asp+glu+asn+gln+ser+thr+his+tyr & name o+od1+od2+oe1+oe2)'
        dense_hbonds = cmd.find_pairs(donors, acceptors, mode=1, cutoff=3.4)
        features['dense_hbond_network'] = len([p for p in dense_hbonds if p[0][1] != p[1][1]]) / total_residues
        
        # 6. 紧凑性指数
        ca_coordinates = []
        cmd.iterate_state(1, f"{obj_name} and name ca", 
                         "ca_coordinates.append([x,y,z])", space={"ca_coordinates": ca_coordinates})
        if len(ca_coordinates) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(ca_coordinates)
            features['compactness_index'] = 1.0 / np.mean(distances)
        else:
            features['compactness_index'] = 0.0
        
        # 7. 螺旋和β片层稳定性
        ss_helix = cmd.count_atoms(f"{obj_name} and ss h and name ca")
        ss_sheet = cmd.count_atoms(f"{obj_name} and ss s and name ca")
        features['helix_sheet_ratio'] = (ss_helix + ss_sheet) / total_residues
        
        # 8. 芳香族相互作用 - 对热稳定性有重要贡献
        aromatic = 'resn phe+tyr+trp+his'
        aromatic_pairs = cmd.find_pairs(f"{obj_name} and {aromatic} and name ca", 
                                      f"{obj_name} and {aromatic} and name ca", 
                                      cutoff=7.0)
        features['aromatic_interactions'] = len(aromatic_pairs) / total_residues
        
        # 9. 甘氨酸含量 - 低甘氨酸含量与高温稳定性相关
        gly_count = cmd.count_atoms(f"{obj_name} and resn gly and name ca")
        features['glycine_content'] = gly_count / total_residues
        
        cmd.delete(obj_name)
        
        logging.info(f"成功提取 {pdb_path} 的热稳定性特征")
    except Exception as e:
        logging.error(f"热稳定性特征提取失败: {str(e)}")
    
    return features

def process_directory(input_path, extract_thermo_features=True):  # 添加提取热稳定性特征选项
    """批量处理PDB目录或单个PDB文件"""
    # 创建输出目录
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f'analyze_pdb_{timestamp}.csv')  # 修改输出路径
    
    # 动态收集所有可能的氨基酸字段
    all_aa_fields = set()
    reports = []

    # 判断输入是文件还是目录
    if os.path.isfile(input_path) and input_path.lower().endswith('.pdb'):
        # 如果是单个PDB文件
        pdb_path = input_path
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
        logging.info(f"正在处理单个文件: {pdb_id}")
        
        # 提取特征
        report = {'pdb_id': pdb_id}
        report.update(analyze_pdb_features(pdb_path))
        report.update(extract_biopython_features(pdb_path))
        
        # 添加热稳定性特征提取
        if extract_thermo_features:
            logging.info(f"为 {pdb_id} 提取热稳定性特征")
            report.update(extract_thermostability_features(pdb_path))
        
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
    elif os.path.isdir(input_path):
        # 如果是目录，处理所有PDB文件
        for filename in os.listdir(input_path):
            if not filename.lower().endswith('.pdb'):
                continue
              
            pdb_path = os.path.join(input_path, filename)
            pdb_id = os.path.splitext(filename)[0]
            logging.info(f"正在处理: {pdb_id}")
          
            # 提取特征
            report = {'pdb_id': pdb_id}
            report.update(analyze_pdb_features(pdb_path))
            report.update(extract_biopython_features(pdb_path))
            
            # 添加热稳定性特征提取
            if extract_thermo_features:
                logging.info(f"为 {pdb_id} 提取热稳定性特征")
                report.update(extract_thermostability_features(pdb_path))
            
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
    else:
        logging.error(f"输入路径既不是PDB文件也不是目录: {input_path}")
        return

    # 合并字段到fieldnames，添加新的归一化字段和热稳定性字段
    fieldnames = [
        'pdb_id', 'disulfide_bonds', 'surface_polar_ratio', 'hydrogen_bonds', 'hydrogen_bonds_norm',
        'hydrophobic_contacts', 'hydrophobic_contacts_norm', 'salt_bridges', 'salt_bridges_norm',
        'hydrophobic_sasa', 'hydrophobic_sasa_norm', 'mean_sasa', 'helix', 'sheet', 'loop'
    ]
    
    # 添加热稳定性特征字段
    if extract_thermo_features:
        thermo_fields = [
            'ion_pair_density', 'core_residue_ratio', 'surface_charge_ratio', 
            'ivywrel_index', 'dense_hbond_network', 'compactness_index',
            'helix_sheet_ratio', 'aromatic_interactions', 'glycine_content'
        ]
        fieldnames.extend(thermo_fields)
    
    # 添加氨基酸组成字段
    fieldnames.extend(sorted(all_aa_fields))

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            # 填充缺失的氨基酸字段为0
            for aa_field in all_aa_fields:
                if aa_field not in report:
                    report[aa_field] = 0.0
            writer.writerow(report)
            
    logging.info(f"分析完成，结果已保存至 {output_csv}")

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="分析PDB文件提取特征")
    parser.add_argument("input_path", help="输入PDB文件或包含PDB文件的目录路径")
    parser.add_argument("--thermostability", action="store_true", help="是否提取热稳定性相关特征")
    parser.add_argument("--output_dir", default="output", help="输出目录路径，默认为'output'")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理PDB文件/目录
    process_directory(args.input_path, extract_thermo_features=args.thermostability)
    
    logging.info(f"分析完成，结果已保存至 {args.output_dir}/ 目录")