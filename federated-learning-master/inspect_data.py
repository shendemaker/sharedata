#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于检查.npz数据文件内容的简单脚本
"""

import os
import numpy as np
import glob
import traceback

def inspect_file(file_path):
    """检查.npz文件的内容"""
    print(f"\n检查文件: {file_path}")
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"文件中的键: {list(data.keys())}")
        
        # 检查结果字典
        if 'results' in data:
            results = data['results'].item()
            print(f"results键中的内容: {list(results.keys())}")
            
            # 打印关键指标的值
            for key in ['accuracy_test', 'loss_test', 'communication_round']:
                if key in results:
                    if isinstance(results[key], np.ndarray) and len(results[key]) > 0:
                        if key.startswith('accuracy'):
                            print(f"  {key}: 最终值 = {results[key][-1]:.4f}")
                        elif key.startswith('loss'):
                            print(f"  {key}: 最小值 = {min(results[key]):.4f}")
                        else:
                            print(f"  {key}: 长度 = {len(results[key])}")
                    else:
                        print(f"  {key}: 非数组或为空")
                else:
                    print(f"  {key}: 不存在")
        
        # 检查超参数字典
        if 'hyperparameters' in data:
            hyperparameters = data['hyperparameters'].item()
            print(f"hyperparameters键中的内容: {list(hyperparameters.keys())}")
            
            # 打印一些关键超参数
            for key in ['n_clients', 'batch_size', 'rounds']:
                if key in hyperparameters:
                    print(f"  {key}: {hyperparameters[key]}")
                    
        # 检查是否有finished标记
        if 'finished' in data:
            print(f"finished: {data['finished']}")
        
    except Exception as e:
        print(f"检查文件时出错: {str(e)}")
        traceback.print_exc()

def main():
    # 获取所有数据集目录
    datasets = [d for d in os.listdir('results') if os.path.isdir(os.path.join('results', d))]
    print(f"发现数据集: {datasets}")
    
    for dataset in datasets:
        dataset_path = os.path.join('results', dataset)
        npz_files = glob.glob(os.path.join(dataset_path, '*.npz'))
        
        if npz_files:
            print(f"\n=== 数据集: {dataset} ===")
            print(f"发现 {len(npz_files)} 个.npz文件")
            
            # 只检查第一个文件
            inspect_file(npz_files[0])
        else:
            print(f"\n=== 数据集: {dataset} ===")
            print("未找到.npz文件")

if __name__ == "__main__":
    main() 