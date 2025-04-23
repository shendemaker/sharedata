#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于修复数据结构以确保Web可视化器能正确显示指标数据
"""

import os
import numpy as np
import glob
import shutil
import traceback

def fix_experiment_structure(file_path):
    """修复实验数据文件的结构"""
    print(f"\n处理文件: {file_path}")
    try:
        # 加载原始数据
        data = np.load(file_path, allow_pickle=True)
        print(f"原始文件中的键: {list(data.keys())}")
        
        # 创建新的数据结构
        new_data = {}
        
        # 如果results和hyperparameters已经在根级别，则不需要修改
        if 'results' in data and 'hyperparameters' in data:
            print("数据结构已经符合要求，无需修复")
            return True
            
        # 创建新的results字典
        results = {}
        
        # 保存所有与结果相关的键
        for key in data.keys():
            if key != 'hyperparameters' and key != 'finished':
                results[key] = data[key]
                
        # 创建新的数据结构
        new_data['results'] = results
        
        if 'hyperparameters' in data:
            new_data['hyperparameters'] = data['hyperparameters']
        else:
            # 如果没有hyperparameters，创建一个基本的hyperparameters字典
            hyperparameters = {
                'n_clients': 50,
                'batch_size': 10,
                'test_batch_size': 1000,
                'rounds': 100,
                'lr': 0.01,
                'log_interval': 10
            }
            new_data['hyperparameters'] = hyperparameters
            
        # 设置finished标记
        new_data['finished'] = True
            
        # 备份原始文件
        backup_path = file_path + '.backup'
        if not os.path.exists(backup_path):
            shutil.copy2(file_path, backup_path)
            print(f"已备份原始文件到: {backup_path}")
            
        # 保存新的数据结构
        np.savez(file_path, **new_data)
        print(f"已更新文件结构: {file_path}")
        
        return True
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        traceback.print_exc()
        return False

def fix_all_experiments():
    """修复所有数据集的实验数据"""
    # 获取所有数据集目录
    datasets = [d for d in os.listdir('results') if os.path.isdir(os.path.join('results', d)) 
               and d not in ['plots', 'trash']]  # 排除非数据集目录
               
    print(f"发现数据集: {datasets}")
    
    for dataset in datasets:
        dataset_path = os.path.join('results', dataset)
        npz_files = glob.glob(os.path.join(dataset_path, '*.npz'))
        
        if npz_files:
            print(f"\n=== 处理数据集: {dataset} ===")
            print(f"发现 {len(npz_files)} 个.npz文件")
            
            # 处理所有npz文件
            for file_path in npz_files:
                fix_experiment_structure(file_path)
        else:
            print(f"\n=== 数据集: {dataset} ===")
            print("未找到.npz文件")
            
    print("\n所有数据集处理完成")
    print("请重新启动Web可视化服务器以查看更新后的结果")

if __name__ == "__main__":
    fix_all_experiments() 