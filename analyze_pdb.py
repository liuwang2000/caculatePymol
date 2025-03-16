import pymol
from pymol import cmd
import sys
import os
import csv
import re
import time
from datetime import datetime

def parse_residue(res_str):
    """解析残基字符串格式为(名称, 编号)，例如'ARG15' -> ('ARG', '15')"""
    match = re.match(r'^([A-Za-z]+)(\d+)$', res_str)
    if not match:
        raise ValueError(f"Invalid residue format: {res_str}")
    return (match.group(1).upper(), match.group(2))

def calculate_catalytic_distance(catalytic_residues):
    """计算催化残基间距"""
    res1, res2 = catalytic_residues
    resn1, resi1 = parse_residue(res1)
    resn2, resi2 = parse_residue(res2)
    sel1 = f'resn {resn1} and resi {resi1}'
    sel2 = f'resn {resn2} and resi {resi2}'
    cmd.distance('catalytic_dist', sel1, sel2)
    return cmd.get_distance('catalytic_dist')

def count_disulfide_bonds():
    """统计二硫键数量（SG原子间距≤2.2Å）"""
    cysteines = cmd.find_pairs('(resn cys & name sg)', '(resn cys & name sg)', 
                              mode=1, cutoff=2.2)
    return len(cysteines)

def surface_polar_ratio():
    """计算表面极性残基占比（溶剂可及性>20）"""
    cmd.select('surface', 'b > 20')
    polar = cmd.count_atoms('surface & (resn arg+lys+asp+glu+asn+gln+his+ser+thr+tyr)')
    total = cmd.count_atoms('surface')
    return polar/total if total >0 else 0

def hbond_network_strength():
    """氢键网络强度（3.2Å距离判据，优化版）"""
    # 供体：主链N (name n) + 侧链N/O (asn/gln的 od/oe)
    donors = '(resn arg+lys+his+asn+gln+ser+thr+tyr+trp & name n+od1+od2+oe1+oe2)'
  
    # 受体：主链O (name o) + 侧链O/N (asp/glu的 od/oe)
    acceptors = '(resn asp+glu+asn+gln+ser+thr+his+tyr & name o+od1+od2+oe1+oe2)'
  
    hbonds = cmd.find_pairs(donors, acceptors, mode=1, cutoff=3.2)
    valid_hbonds = [pair for pair in hbonds if pair[0][1] != pair[1][1]]
  
    return len(valid_hbonds)

def hydrophobic_core_density():
    """疏水核心接触点（溶剂可及性<10且距离<5Å）"""
    cmd.select('hydrophobic_core', 'resn ala+val+ile+leu+phe+trp+pro+met & b < 10')
    contacts = cmd.find_pairs('hydrophobic_core', 'hydrophobic_core', cutoff=5.0)
    return len(contacts)

def salt_bridge_density():
    """盐桥密度（正负残基间距≤4Å）"""
    positive = 'resn arg+lys'
    negative = 'resn asp+glu'
    salt_bridges = cmd.find_pairs(positive, negative, mode=2, cutoff=4.0)
    return len(salt_bridges)

def main(pdb_file, catalytic_residues=None):
    if not cmd.get_names():
        cmd.reinitialize()
    obj_name = None
    try:
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f'PDB文件未找到: {pdb_file}')
          
        print(f'\n正在加载PDB文件: {pdb_file}')
        cmd.feedback("disable", "all", "actions")
        cmd.reinitialize()
        obj_name = os.path.splitext(os.path.basename(pdb_file))[0]
        cmd.load(pdb_file, obj_name)
        cmd.set('solvent_radius', 1.4)
        print('PDB文件加载成功，开始分析...')
      
        # PyMOL 3.x 兼容预处理
        cmd.set("ignore_case", 1)
        # 计算溶剂可及性
        cmd.get_area(obj_name, load_b=1)
        # 移除is_processing检查，改用固定延时确保计算完成
        time.sleep(2)  # 等待2秒确保溶剂可及性计算完成
        
      
    except Exception as e:
        print(f'\n错误发生: {str(e)}')
        sys.exit(1)
      
    try:
        sys.stderr.write("\n开始计算结构特征...\n")
        sys.stdout.flush()
      
        catalytic_distance = None
        if catalytic_residues:
            catalytic_distance = calculate_catalytic_distance(catalytic_residues)
            print(f"\n催化残基间距: {catalytic_distance:.2f} Å")
        else:
            print("\n未提供催化残基对，跳过距离计算")

        print("\n[1/6] 计算二硫键...")
        disulfide_count = count_disulfide_bonds()
        print("[2/6] 计算表面极性...")
        polar_ratio = surface_polar_ratio()
        print("[3/6] 计算氢键网络...")
        hbond_strength = hbond_network_strength()
        print("[4/6] 计算疏水核心...")
        core_density = hydrophobic_core_density()
        print("[5/6] 计算盐桥密度...")
        salt_density = salt_bridge_density()
      
        # 输出结果
        print(f"二硫键数量: {disulfide_count}")
        print(f"表面极性残基占比: {polar_ratio:.2%}")
        print(f"氢键网络强度: {hbond_strength} 个氢键")
        print(f"疏水核心接触点: {core_density}")
        print(f"盐桥密度: {salt_density}")
  
        csv_filename = f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            if catalytic_distance is not None:
                writer.writerow(['Catalytic Distance (Å)', f'{catalytic_distance:.2f}'])
            writer.writerow(['Disulfide Bonds', disulfide_count])
            writer.writerow(['Surface Polar Ratio', f'{polar_ratio:.2%}'])
            writer.writerow(['HBond Network Strength', hbond_strength])
            writer.writerow(['Hydrophobic Core Density', core_density])
            writer.writerow(['Salt Bridge Density', salt_density])
        print(f'\n报告已生成: {csv_filename}')
  
    finally:
        if obj_name:
            cmd.delete(obj_name)
        if catalytic_residues:
            cmd.delete('catalytic_dist')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_pdb.py <pdb_file> [residue1 residue2]")
        print("Example: python analyze_pdb.py enzyme.pdb ARG15 HIS57")
        sys.exit(1)
      
    catalytic_args = sys.argv[2:4] if len(sys.argv)>=4 else None
    main(sys.argv[1], catalytic_args)