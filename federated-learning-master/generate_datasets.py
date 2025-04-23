#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于生成各数据集实验结果的脚本，以便在Web可视化工具中显示
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import traceback
from plot_results import ResultsVisualizer

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成示例实验数据用于Web可视化')
    
    parser.add_argument('--datasets', nargs='+', default=['mnist', 'fashionmnist', 'cifar10'],
                      help='要生成数据的数据集，例如：--datasets mnist fashionmnist cifar10')
    
    return parser.parse_args()

def generate_sample_experiment_data(dataset, rounds=100, clients=50):
    """
    为指定数据集生成示例实验数据
    
    参数:
        dataset: 数据集名称
        rounds: 模拟的训练轮数
        clients: 模拟的客户端数量
    """
    try:
        print(f"正在为 {dataset} 生成示例实验数据...")
        
        # 创建数据集目录（如果不存在）
        dataset_dir = os.path.join('results', dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        # 模拟实验数据
        # 使用不同模式生成不同数据集的数据，使它们看起来不同
        if dataset == 'mnist':
            acc_test = 0.5 + 0.45 * (1 - np.exp(-np.arange(rounds)/30))
            loss_test = 0.7 - 0.65 * (1 - np.exp(-np.arange(rounds)/25))
            factor = 1.0
        elif dataset == 'fashionmnist':
            acc_test = 0.4 + 0.45 * (1 - np.exp(-np.arange(rounds)/40))
            loss_test = 0.8 - 0.60 * (1 - np.exp(-np.arange(rounds)/35))
            factor = 0.95
        else:  # cifar10
            acc_test = 0.3 + 0.50 * (1 - np.exp(-np.arange(rounds)/50))
            loss_test = 0.9 - 0.70 * (1 - np.exp(-np.arange(rounds)/45))
            factor = 0.9
        
        acc_train = acc_test * factor
        loss_train = loss_test / factor
        
        # 添加一些噪声使曲线看起来更真实
        noise = np.random.normal(0, 0.01, rounds)
        acc_test += noise
        acc_test = np.clip(acc_test, 0, 1)  # 确保准确率在[0,1]范围内
        
        noise = np.random.normal(0, 0.02, rounds)
        loss_test += noise
        loss_test = np.clip(loss_test, 0, 1)  # 确保损失是正数
        
        # 创建结果字典
        results = {
            'accuracy_test': acc_test,
            'loss_test': loss_test,
            'accuracy_train': acc_train,
            'loss_train': loss_train,
            'communication_round': np.arange(rounds)
        }
        
        # 添加客户端数据
        for i in range(clients):
            # 为每个客户端生成略有不同的损失曲线
            client_noise = np.random.normal(0, 0.05, rounds)
            client_loss = loss_train + client_noise
            client_loss = np.clip(client_loss, 0, 1)
            results[f'client{i}_loss'] = client_loss
        
        # 创建超参数字典
        hyperparameters = {
            'n_clients': clients,
            'batch_size': 10,
            'test_batch_size': 1000,
            'rounds': rounds,
            'lr': 0.01,
            'log_interval': 10
        }
        
        # 创建保存文件名
        exp_id = f"xp_{np.random.randint(10000, 99999)}"
        results_file = os.path.join(dataset_dir, f"{exp_id}.npz")
        
        # 保存数据
        np.savez(
            results_file,
            results=results,
            hyperparameters=hyperparameters,
            finished=True
        )
        
        print(f"已生成 {dataset} 的示例实验数据，保存为 {results_file}")
        
        # 确保plots目录存在
        plots_dir = os.path.join(dataset_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        return True
    except Exception as e:
        print(f"生成 {dataset} 数据时出错: {str(e)}")
        traceback.print_exc()
        return False

def ensure_results_directory():
    """确保results目录存在"""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("已创建results目录")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 确保results目录存在
    ensure_results_directory()
    
    # 记录成功生成的数据集
    successful_datasets = []
    failed_datasets = []
    
    # 为每个指定的数据集生成示例数据
    for dataset in args.datasets:
        success = generate_sample_experiment_data(dataset)
        if success:
            successful_datasets.append(dataset)
        else:
            failed_datasets.append(dataset)
    
    # 打印结果摘要
    print("\n=== 数据生成结果摘要 ===")
    if successful_datasets:
        print(f"成功生成的数据集: {', '.join(successful_datasets)}")
    if failed_datasets:
        print(f"生成失败的数据集: {', '.join(failed_datasets)}")
        
    print("\n所有数据集的示例实验数据已处理完成")
    print("现在可以使用web_visualizer.py查看这些数据")
    print("如需启动可视化服务器，请运行:")
    print("python web_visualizer.py")

if __name__ == "__main__":
    main() 