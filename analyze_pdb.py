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

# 确保result目录存在
os.makedirs('result', exist_ok=True)
# 确保output目录存在
os.makedirs('output', exist_ok=True)

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

def analyze_pdb_features(pdb_path):
    """PyMOL特征分析"""
    features = {}
    try:
        # 初始化PyMOL环境
        cmd.reinitialize()
        obj_name = os.path.splitext(os.path.basename(pdb_path))[0]
        cmd.load(pdb_path, obj_name)
        cmd.set('solvent_radius', 1.4)
      
        # 计算蛋白质总氨基酸数
        total_residues = cmd.count_atoms(f"{obj_name} and name ca")
        if total_residues == 0:
            logging.error(f"无法在{pdb_path}中找到氨基酸残基")
            return features
      
        # 计算溶剂可及性
        cmd.get_area(obj_name, load_b=1)
      
        # 二硫键
        cysteines = cmd.find_pairs('(resn cys & name sg)', '(resn cys & name sg)', 
                                 mode=1, cutoff=2.2)
        features['disulfide_bonds_ratio'] = len(cysteines) / total_residues
      
        # 表面极性比例
        cmd.set('dot_solvent', 1)  # 开启溶剂点计算
        cmd.set('dot_density', 3)  # 提高点密度
        cmd.select('surface', 'b > 10')
        polar = cmd.count_atoms('surface & (resn arg+lys+asp+glu+asn+gln+his+ser+thr+tyr)')
        total_surface = cmd.count_atoms('surface')
        features['surface_polar_ratio'] = polar/total_surface if total_surface >0 else 0
      
        # 氢键网络（优化版）
        donors = '(resn arg+lys+his+asn+gln+ser+thr+tyr+trp & name n+od1+od2+oe1+oe2)'
        acceptors = '(resn asp+glu+asn+gln+ser+thr+his+tyr & name o+od1+od2+oe1+oe2)'
        hbonds = cmd.find_pairs(donors, acceptors, mode=1, cutoff=3.2)
        features['hydrogen_bonds_ratio'] = len([p for p in hbonds if p[0][1] != p[1][1]]) / total_residues
      
        # 疏水核心
        cmd.select('hydrophobic_core', 'resn ala+val+ile+leu+phe+trp+pro+met & b < 10')
        contacts = cmd.find_pairs('hydrophobic_core', 'hydrophobic_core', cutoff=5.0)
        features['hydrophobic_contacts_ratio'] = len(contacts) / total_residues
      
        # 盐桥
        positive = 'resn arg+lys'
        negative = 'resn asp+glu'
        salt_bridges = cmd.find_pairs(positive, negative, mode=2, cutoff=4.0)
        features['salt_bridges_ratio'] = len(salt_bridges) / total_residues
      
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
      
        # 计算总残基数
        total_residues = sum(1 for _ in structure.get_residues() if _.get_resname() not in ['HOH', 'WAT'])
        if total_residues == 0:
            return features
      
        # 氨基酸组成
        aa_count = defaultdict(int)
        for residue in structure.get_residues():
            resname = residue.get_resname()
            if resname not in ['HOH', 'WAT']:
                aa_count[resname] += 1
        features['aa_composition'] = {k: v/total_residues for k, v in aa_count.items()}
      
        # SASA计算
        sasa_calculator = ShrakeRupley()
        sasa_calculator.compute(structure, level='R')
        hydrophobic = ['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'PRO', 'MET', 'TRP']
        hydrophobic_sasa = 0.0
        hydrophobic_count = 0
        total_sasa = 0.0
      
        for residue in structure.get_residues():
            if residue.get_resname() in hydrophobic:
                hydrophobic_sasa += residue.sasa
                hydrophobic_count += 1
            if residue.get_resname() not in ['HOH', 'WAT']:
                total_sasa += residue.sasa
      
        features['hydrophobic_sasa_ratio'] = hydrophobic_sasa / total_sasa if total_sasa > 0 else 0
        features['mean_sasa_per_residue'] = total_sasa / total_residues if total_residues > 0 else 0
      
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
    """提取热稳定性相关特征 - 优化增强版本，增加更全面的高温蛋白特征"""
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
        
        # 4. IVYWREL指数 (Val, Ile, Tyr, Trp, Arg, Glu, Leu的比例)
        cmd.select('ivywrel', f"{obj_name} and resn val+ile+tyr+trp+arg+glu+leu")
        ivywrel_count = cmd.count_atoms('ivywrel and name ca')
        features['ivywrel_index'] = ivywrel_count / total_residues
        
        # 5. 密集氢键网络
        donors = '(resn arg+lys+his+asn+gln+ser+thr+tyr+trp)'
        acceptors = '(resn asp+glu+asn+gln+ser+thr+his+tyr)'
        hbonds = cmd.find_pairs(f"{donors} and name n+nd1+ne2+og+og1+oh", 
                             f"{acceptors} and name o+od1+od2+oe1+oe2+nd1+oe1+oe2", 
                             mode=1, cutoff=3.5)
        features['dense_hbond_network'] = len([p for p in hbonds if abs(p[0][1] - p[1][1]) > 5]) / total_residues
        
        # 6. 紧凑度指数 (环路残基比例的倒数，越低越紧凑)
        cmd.dss()  # 计算二级结构
        loop_residues = cmd.count_atoms(f"{obj_name} and ss l and name ca")
        features['compactness_index'] = 1 - (loop_residues / total_residues)
        
        # 7. 二级结构比例 - 对热稳定性有重要影响
        helix_residues = cmd.count_atoms(f"{obj_name} and ss h and name ca")
        sheet_residues = cmd.count_atoms(f"{obj_name} and ss s and name ca")
        features['helix_ratio'] = helix_residues / total_residues
        features['sheet_ratio'] = sheet_residues / total_residues
        
        # 8. 螺旋/片层比例 - 高温蛋白通常具有更高的螺旋结构比例
        features['helix_sheet_ratio'] = (helix_residues / sheet_residues) if sheet_residues > 0 else 0
        
        # 9. 芳香族氨基酸相互作用 - 增强蛋白质稳定性
        aromatic = 'resn phe+tyr+trp+his'
        arom_interactions = cmd.find_pairs(f"{aromatic} and not name c+n+o+ca", 
                                         f"{aromatic} and not name c+n+o+ca", 
                                         mode=1, cutoff=6.0)
        # 过滤出不同残基之间的相互作用
        filtered_arom = [p for p in arom_interactions if p[0][1] != p[1][1]]
        features['aromatic_interactions'] = len(filtered_arom) / total_residues
        
        # 10. 疏水核心联系密度 - 高温蛋白的疏水核心通常更为紧密
        cmd.select('hydrophobic', f"{obj_name} and resn ala+val+leu+ile+met+phe+trp")
        hydrophobic_contacts = cmd.find_pairs('hydrophobic and b < 10', 
                                           'hydrophobic and b < 10', 
                                           mode=1, cutoff=5.0)
        hydrophobic_core = cmd.count_atoms('hydrophobic and b < 10 and name ca')
        features['hydrophobic_core_density'] = len(hydrophobic_contacts) / (hydrophobic_core or 1)
        
        # 11. 荷电氨基酸分布 - 计算各类荷电氨基酸比例
        for aa in ['ARG', 'LYS', 'ASP', 'GLU', 'HIS']:
            cmd.select(f'aa_{aa.lower()}', f"{obj_name} and resn {aa}")
            aa_count = cmd.count_atoms(f'aa_{aa.lower()} and name ca')
            features[f'aa_{aa}_ratio'] = aa_count / total_residues
        
        # 12. 脯氨酸含量 - 可能影响蛋白质刚性
        cmd.select('proline', f"{obj_name} and resn pro")
        proline_count = cmd.count_atoms('proline and name ca')
        features['proline_content'] = proline_count / total_residues
        
        # 13. 甘氨酸含量 - 通常在高温蛋白质中含量较低
        cmd.select('glycine', f"{obj_name} and resn gly")
        glycine_count = cmd.count_atoms('glycine and name ca')
        features['glycine_content'] = glycine_count / total_residues
        
        # 14. 疏水性氨基酸总含量
        cmd.select('all_hydrophobic', f"{obj_name} and resn ala+val+leu+ile+met+phe+trp+pro")
        hydrophobic_total = cmd.count_atoms('all_hydrophobic and name ca')
        features['hydrophobic_content'] = hydrophobic_total / total_residues
        
        # 15. 极性氨基酸总含量
        cmd.select('polar', f"{obj_name} and resn ser+thr+asn+gln+tyr")
        polar_total = cmd.count_atoms('polar and name ca')
        features['polar_content'] = polar_total / total_residues
        
        # 16. 计算荷电网络的连通性 - 荷电网络中每个氨基酸平均接触数
        if len(ion_pairs) > 0:
            # 提取所有参与离子对的残基
            ion_residues = set()
            for pair in ion_pairs:
                ion_residues.add((pair[0][0], pair[0][1]))  # 格式化为(chain, resid)
                ion_residues.add((pair[1][0], pair[1][1]))
            
            # 计算每个荷电残基的平均接触数
            contacts_per_residue = len(ion_pairs) * 2 / len(ion_residues)
            features['ion_network_connectivity'] = contacts_per_residue
        else:
            features['ion_network_connectivity'] = 0.0
            
        # 17. 表面-体积比
        # 计算蛋白质表面积和体积
        cmd.set('solvent_radius', 1.4)
        surface_area = cmd.get_area(obj_name)
        
        # 使用PyMOL计算体积的近似值
        # 注意：这是一个近似方法
        cmd.create('surf_obj', f"{obj_name}")
        cmd.set('surface_quality', 2)
        cmd.show('surface', 'surf_obj')
        # 获取体积 (单位：立方埃)
        results = []
        cmd.iterate_state(1, 'surf_obj', 'results.append((x,y,z))', space={'results': results})
        if results:
            # 使用凸包体积作为近似
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(results)
                volume = hull.volume
                features['surface_volume_ratio'] = surface_area / (volume or 1)
            except:
                features['surface_volume_ratio'] = 0.0
        else:
            features['surface_volume_ratio'] = 0.0
        
        cmd.delete('surf_obj')
        
        # 清理选择器
        cmd.delete(obj_name)
    except Exception as e:
        logging.error(f"热稳定性特征提取失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    
    return features

def process_pdb_file(pdb_path, extract_thermo_features=True, optimized=False):
    """处理单个PDB文件并提取特征"""
    try:
        # 确保PDB文件路径是绝对路径
        pdb_path = os.path.abspath(pdb_path)
        
        # 检查文件是否存在
        if not os.path.exists(pdb_path):
            logging.error(f"PDB文件不存在: {pdb_path}")
            return {'pdb_id': os.path.splitext(os.path.basename(pdb_path))[0], 'error': 'PDB文件不存在'}
            
        # 检查文件大小
        if os.path.getsize(pdb_path) == 0:
            logging.error(f"PDB文件为空: {pdb_path}")
            return {'pdb_id': os.path.splitext(os.path.basename(pdb_path))[0], 'error': 'PDB文件为空'}
        
        features = {'pdb_id': os.path.splitext(os.path.basename(pdb_path))[0]}
        
        # PyMOL基础特征
        try:
            logging.info(f"为{features['pdb_id']}使用PyMOL提取基础特征...")
            pymol_features = analyze_pdb_features(pdb_path)
            if not pymol_features:
                logging.warning(f"PyMOL未能为{features['pdb_id']}提取特征")
            features.update(pymol_features)
        except Exception as e:
            logging.error(f"PyMOL特征提取失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
        
        # Biopython特征
        try:
            logging.info(f"为{features['pdb_id']}使用Biopython提取特征...")
            bio_features = extract_biopython_features(pdb_path)
            if not bio_features:
                logging.warning(f"Biopython未能为{features['pdb_id']}提取特征")
            features.update(bio_features)
        except Exception as e:
            logging.error(f"Biopython特征提取失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
        
        # 热稳定性特征提取
        if extract_thermo_features:
            try:
                logging.info(f"为{features['pdb_id']}提取热稳定性特征...")
                thermo_features = extract_thermostability_features(pdb_path)
                if not thermo_features:
                    logging.warning(f"未能为{features['pdb_id']}提取热稳定性特征")
                features.update(thermo_features)
            except Exception as e:
                logging.error(f"热稳定性特征提取失败: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        # 确保至少提取了一些特征
        if len(features) <= 1:  # 只有pdb_id
            logging.error(f"无法为{features['pdb_id']}提取任何特征")
            features['error'] = '无法提取特征'
            
        return features
    except Exception as e:
        logging.error(f"处理PDB文件 {pdb_path} 时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {'pdb_id': os.path.splitext(os.path.basename(pdb_path))[0], 'error': str(e)}

def process_directory(directory, output_dir='output', extract_thermo_features=True, optimized=False):
    """处理目录中的所有PDB文件"""
    if not os.path.exists(directory):
        logging.error(f"目录不存在: {directory}")
        return None
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有PDB文件
    pdb_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.pdb', '.ent')):
                pdb_files.append(os.path.join(root, file))
    
    if not pdb_files:
        logging.error(f"目录 {directory} 中未找到PDB文件")
        return None
    
    logging.info(f"在 {directory} 中找到 {len(pdb_files)} 个PDB文件")
    
    # 处理所有PDB文件
    results = []
    processed_count = 0
    error_count = 0
    start_time = time.time()
    last_progress_update = start_time
    
    for i, pdb_file in enumerate(pdb_files):
        try:
            # 每10个文件或30秒更新一次进度报告
            current_time = time.time()
            if i % 10 == 0 or current_time - last_progress_update > 30:
                progress = (i + 1) / len(pdb_files) * 100
                elapsed_time = current_time - start_time
                
                # 估计剩余时间
                if i > 0:
                    avg_time_per_file = elapsed_time / (i + 1)
                    remaining_files = len(pdb_files) - (i + 1)
                    remaining_time = avg_time_per_file * remaining_files
                    
                    # 格式化时间显示
                    if remaining_time > 3600:
                        time_str = f"{remaining_time/3600:.1f}小时"
                    else:
                        time_str = f"{remaining_time/60:.1f}分钟"
                    
                    logging.info(f"进度: {progress:.1f}% ({i+1}/{len(pdb_files)}), 预计剩余时间: {time_str}")
                else:
                    logging.info(f"进度: {progress:.1f}% ({i+1}/{len(pdb_files)})")
                
                last_progress_update = current_time
            
            logging.info(f"处理文件: {os.path.basename(pdb_file)}")
            features = process_pdb_file(pdb_file, extract_thermo_features=extract_thermo_features, optimized=optimized)
            
            if features:
                if 'error' in features:
                    logging.warning(f"处理 {os.path.basename(pdb_file)} 失败: {features['error']}")
                    error_count += 1
                else:
                    results.append(features)
                    processed_count += 1
        except Exception as e:
            logging.error(f"处理 {pdb_file} 时发生异常: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            error_count += 1
    
    # 报告处理结果
    total_time = time.time() - start_time
    logging.info(f"PDB文件处理完成. 总计: {len(pdb_files)}, 成功: {processed_count}, 失败: {error_count}")
    logging.info(f"总处理时间: {total_time/60:.1f}分钟 (平均每个文件 {total_time/len(pdb_files):.1f}秒)")
    
    # 保存到CSV
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"analyze_pdb_{timestamp}.csv")
        
        # 获取所有字段
        fieldnames = set()
        for result in results:
            for key in result.keys():
                if key != 'aa_composition':  # 排除氨基酸组成字典
                    fieldnames.add(key)
        
        # 添加氨基酸组成字段
        aa_types = set()
        for result in results:
            if 'aa_composition' in result:
                aa_types.update(result['aa_composition'].keys())
        
        for aa in aa_types:
            fieldnames.add(f"aa_{aa}")
        
        # 将pdb_id放在最前面
        fieldnames_list = sorted(fieldnames)
        if 'pdb_id' in fieldnames_list:
            fieldnames_list.remove('pdb_id')
            fieldnames_list.insert(0, 'pdb_id')
        
        # 写入CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames_list)
            writer.writeheader()
            for result in results:
                row = {}
                for key, value in result.items():
                    if key != 'aa_composition':
                        row[key] = value
                
                # 展开氨基酸组成
                if 'aa_composition' in result:
                    for aa, fraction in result['aa_composition'].items():
                        row[f"aa_{aa}"] = fraction
                
                writer.writerow(row)
        
        logging.info(f"分析结果已保存到: {output_file}")
        return output_file
    else:
        logging.warning("没有生成任何分析结果")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析PDB文件的特征')
    parser.add_argument('input_path', help='输入PDB文件或包含PDB文件的目录路径')
    parser.add_argument('--output_dir', default='output', help='输出目录路径')
    parser.add_argument('--thermostability', action='store_true', help='是否提取热稳定性特征')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理输入
    if os.path.isdir(args.input_path):
        process_directory(args.input_path, args.output_dir, 
                          extract_thermo_features=args.thermostability)
    elif os.path.isfile(args.input_path) and args.input_path.lower().endswith(('.pdb', '.ent')):
        # 单个PDB文件处理
        features = process_pdb_file(args.input_path, 
                                   extract_thermo_features=args.thermostability)
        
        if features:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(args.output_dir, f"analyze_pdb_{timestamp}.csv")
            
            # 获取所有字段
            fieldnames = set()
            for key in features.keys():
                if key != 'aa_composition':  # 排除氨基酸组成字典
                    fieldnames.add(key)
            
            # 添加氨基酸组成字段
            if 'aa_composition' in features:
                for aa in features['aa_composition'].keys():
                    fieldnames.add(f"aa_{aa}")
            
            # 将pdb_id放在最前面
            fieldnames_list = sorted(fieldnames)
            if 'pdb_id' in fieldnames_list:
                fieldnames_list.remove('pdb_id')
                fieldnames_list.insert(0, 'pdb_id')
            
            # 写入CSV
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames_list)
                writer.writeheader()
                
                row = {}
                for key, value in features.items():
                    if key != 'aa_composition':
                        row[key] = value
                
                # 展开氨基酸组成
                if 'aa_composition' in features:
                    for aa, fraction in features['aa_composition'].items():
                        row[f"aa_{aa}"] = fraction
                
                writer.writerow(row)
            
            logging.info(f"分析结果已保存到: {output_file}")
        else:
            logging.warning("没有生成分析结果")
    else:
        logging.error(f"无效的输入路径: {args.input_path}")

if __name__ == "__main__":
    main()