#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob

def main():
    """检查差分隐私对比相关文件"""
    results_dir = "results"
    
    # 检查结果目录是否存在
    if not os.path.exists(results_dir):
        print(f"结果目录 '{results_dir}' 不存在")
        return
    
    # 获取所有数据集
    datasets = []
    for dir_path in glob.glob(os.path.join(results_dir, "*")):
        if os.path.isdir(dir_path):
            datasets.append(os.path.basename(dir_path))
    
    print(f"找到以下数据集: {', '.join(datasets)}")
    
    # 检查每个数据集的comparison目录和图片文件
    for dataset in datasets:
        print(f"\n检查数据集 {dataset}:")
        
        # 检查comparison目录
        comparison_dir = os.path.join(results_dir, dataset, "comparison")
        if not os.path.exists(comparison_dir):
            print(f"  comparison目录不存在: {comparison_dir}")
            continue
        
        # 检查数据文件
        data_file = os.path.join(comparison_dir, "dp_comparison_data.npz")
        if os.path.exists(data_file):
            print(f"  数据文件存在: {data_file}")
        else:
            print(f"  数据文件不存在: {data_file}")
        
        # 检查图片文件
        image_files = [
            "privacy_accuracy_tradeoff.png",
            "accuracy_comparison.png",
            "loss_comparison.png"
        ]
        
        for img_file in image_files:
            full_path = os.path.join(comparison_dir, img_file)
            if os.path.exists(full_path):
                print(f"  图片文件存在: {img_file}")
            else:
                print(f"  图片文件不存在: {img_file}")
    
    # 列出所有文件
    print("\n结果目录的文件结构:")
    for root, dirs, files in os.walk(results_dir):
        level = root.replace(results_dir, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")

if __name__ == "__main__":
    main() 