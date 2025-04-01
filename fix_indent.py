#!/usr/bin/env python
"""
修复train_rf_model.py中的缩进问题
"""
import re

def fix_indentation(file_path):
    """修复文件中的缩进问题"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复问题1: 在321行左右的"for col in original_features:"的缩进问题
    content = re.sub(
        r'(\s{8})for col in original_features:',
        r'    for col in original_features:',
        content
    )
    
    # 修复问题2: 在690行左右的"os.makedirs(output_dir, exist_ok=True)"的缩进问题
    content = re.sub(
        r'(\s{8})os\.makedirs\(output_dir, exist_ok=True\)',
        r'    os.makedirs(output_dir, exist_ok=True)',
        content
    )
    
    # 修复问题3: 在888行左右的"plt.xlabel('实际温度 (°C)')"的缩进问题
    content = re.sub(
        r'(\s{8})plt\.xlabel\(\'实际温度 \(°C\)\'\)',
        r'    plt.xlabel(\'实际温度 (°C)\')',
        content
    )
    
    # 修复问题4: 在889行左右的"plt.ylabel('预测温度 (°C)')"的缩进问题
    content = re.sub(
        r'(\s{8})plt\.ylabel\(\'预测温度 \(°C\)\'\)',
        r'    plt.ylabel(\'预测温度 (°C)\')',
        content
    )
    
    # 修复问题5: 在892行左右的"plt.legend()"的缩进问题
    content = re.sub(
        r'(\s{8})plt\.legend\(\)',
        r'    plt.legend()',
        content
    )
    
    # 修复问题6: 在908行左右的"plt.close()"的缩进问题
    content = re.sub(
        r'(\s{8})plt\.close\(\)',
        r'    plt.close()',
        content
    )
    
    # 修复问题7: 在921行左右的"plt.xlabel('预测误差 (°C)')"的缩进问题
    content = re.sub(
        r'(\s{8})plt\.xlabel\(\'预测误差 \(°C\)\'\)',
        r'    plt.xlabel(\'预测误差 (°C)\')',
        content
    )
    
    # 修复问题8: 在922行左右的"plt.ylabel('频率')"的缩进问题
    content = re.sub(
        r'(\s{8})plt\.ylabel\(\'频率\'\)',
        r'    plt.ylabel(\'频率\')',
        content
    )
    
    # 修复问题9: 在932行左右的"plt.close()"的缩进问题
    content = re.sub(
        r'(\s{8})plt\.close\(\)',
        r'    plt.close()',
        content
    )
    
    # 将修复后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复{file_path}中的缩进问题")

if __name__ == "__main__":
    fix_indentation("train_rf_model.py") 