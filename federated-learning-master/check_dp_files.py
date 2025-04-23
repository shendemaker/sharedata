#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于检查差分隐私对比相关文件是否存在的工具
"""

import os
import glob
import numpy as np

def check_file_exists(file_path):
    """检查文件是否存在"""
    exists = os.path.exists(file_path)
    print(f"检查文件: {file_path} - {'存在' if exists else '不存在'}")
    return exists

def check_dir_exists(dir_path):
    """检查目录是否存在"""
    exists = os.path.exists(dir_path) and os.path.isdir(dir_path)
    print(f"检查目录: {dir_path} - {'存在' if exists else '不存在'}")
    if not exists and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")
        except Exception as e:
            print(f"创建目录失败: {str(e)}")
    return exists

def check_dataset_dp_files(dataset, results_dir='results'):
    """检查特定数据集的差分隐私对比文件"""
    print(f"\n正在检查数据集 {dataset} 的差分隐私对比文件...")
    
    # 检查数据集目录
    dataset_dir = os.path.join(results_dir, dataset)
    if not check_dir_exists(dataset_dir):
        return False
    
    # 检查comparison子目录
    comparison_dir = os.path.join(dataset_dir, 'comparison')
    if not check_dir_exists(comparison_dir):
        return False
    
    # 检查数据文件
    data_file = os.path.join(comparison_dir, 'dp_comparison_data.npz')
    if check_file_exists(data_file):
        try:
            data = np.load(data_file, allow_pickle=True)
            print(f"数据文件包含以下键: {list(data.keys())}")
        except Exception as e:
            print(f"读取数据文件失败: {str(e)}")
    
    # 检查图片文件
    image_files = [
        os.path.join(comparison_dir, 'privacy_accuracy_tradeoff.png'),
        os.path.join(comparison_dir, 'accuracy_comparison.png'),
        os.path.join(comparison_dir, 'loss_comparison.png')
    ]
    
    missing_images = []
    for img_file in image_files:
        if not check_file_exists(img_file):
            missing_images.append(os.path.basename(img_file))
    
    if missing_images:
        print(f"缺少以下图片文件: {', '.join(missing_images)}")
        return False
    
    print(f"数据集 {dataset} 的差分隐私对比文件检查完成，所有文件存在")
    return True

def list_all_files_in_results(results_dir='results'):
    """列出results目录中的所有文件"""
    print("\n列出results目录中的所有文件...")
    
    for root, dirs, files in os.walk(results_dir):
        level = root.replace(results_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")

def main():
    """主函数"""
    results_dir = 'results'
    
    # 获取所有数据集
    datasets = []
    for dir_path in glob.glob(os.path.join(results_dir, '*')):
        if os.path.isdir(dir_path):
            datasets.append(os.path.basename(dir_path))
    
    print(f"发现以下数据集: {', '.join(datasets)}")
    
    # 检查每个数据集
    all_passed = True
    for dataset in datasets:
        if not check_dataset_dp_files(dataset, results_dir):
            all_passed = False
    
    # 列出所有文件
    list_all_files_in_results(results_dir)
    
    if all_passed:
        print("\n所有数据集的差分隐私对比文件检查通过")
    else:
        print("\n部分数据集的差分隐私对比文件检查失败")
        print("您可以运行以下命令生成差分隐私对比数据:")
        print("python fix_dp_compare.py")

if __name__ == "__main__":
    main() 