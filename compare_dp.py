#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成差分隐私对比数据和图表
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 防止需要GUI
import matplotlib.pyplot as plt
import shutil
import random

def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_example_data(dataset):
    """创建示例数据"""
    print(f"为 {dataset} 创建示例差分隐私对比数据")
    
    # 创建数据目录
    data_dir = os.path.join('results', dataset)
    comparison_dir = os.path.join(data_dir, 'comparison')
    ensure_dir(data_dir)
    ensure_dir(comparison_dir)
    
    # 创建示例差分隐私数据
    # 模拟不同ε值下的准确率和损失
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
    accuracy_values = []
    loss_values = []
    
    # 基准精度和损失（无差分隐私）
    base_accuracy = 0.85 if dataset == 'mnist' else 0.75 if dataset == 'fashionmnist' else 0.65
    base_loss = 0.2 if dataset == 'mnist' else 0.3 if dataset == 'fashionmnist' else 0.4
    
    # 为每个epsilon值生成模拟数据
    for eps in epsilon_values:
        if eps == float('inf'):  # 无差分隐私
            accuracy = base_accuracy
            loss = base_loss
        else:
            # 模拟精度随ε减小而降低
            noise_factor = 1.0 / (eps + 0.1)
            accuracy = base_accuracy * (1 - 0.4 * noise_factor) + random.uniform(-0.02, 0.02)
            accuracy = max(0.4, min(0.95, accuracy))  # 限制范围
            
            # 模拟损失随ε减小而增加
            loss = base_loss * (1 + 0.5 * noise_factor) + random.uniform(-0.02, 0.02)
            loss = max(0.1, min(0.9, loss))  # 限制范围
        
        accuracy_values.append(accuracy)
        loss_values.append(loss)
    
    # 模拟训练轮次上的精度曲线
    rounds = np.arange(100)
    
    # 无差分隐私的精度曲线
    no_dp_accuracy = base_accuracy * (1 - np.exp(-rounds/30)) + np.random.normal(0, 0.01, 100)
    no_dp_accuracy = np.clip(no_dp_accuracy, 0.3, 0.95)
    
    # 有差分隐私的精度曲线(ε=1.0)
    dp_accuracy = (base_accuracy - 0.15) * (1 - np.exp(-rounds/40)) + np.random.normal(0, 0.015, 100)
    dp_accuracy = np.clip(dp_accuracy, 0.25, 0.9)
    
    # 无差分隐私的损失曲线
    no_dp_loss = base_loss + 0.5 * np.exp(-rounds/25) + np.random.normal(0, 0.01, 100)
    no_dp_loss = np.clip(no_dp_loss, 0.05, 0.7)
    
    # 有差分隐私的损失曲线(ε=1.0)
    dp_loss = (base_loss + 0.2) + 0.5 * np.exp(-rounds/35) + np.random.normal(0, 0.015, 100)
    dp_loss = np.clip(dp_loss, 0.1, 0.8)
    
    # 保存数据
    np.savez(
        os.path.join(comparison_dir, 'dp_comparison_data.npz'),
        epsilon_values=epsilon_values,
        accuracy_values=accuracy_values,
        loss_values=loss_values,
        rounds=rounds,
        no_dp_accuracy=no_dp_accuracy,
        dp_accuracy=dp_accuracy,
        no_dp_loss=no_dp_loss,
        dp_loss=dp_loss
    )
    
    print(f"数据已保存至 {os.path.join(comparison_dir, 'dp_comparison_data.npz')}")
    
    # 创建图表
    create_plots(dataset, comparison_dir, epsilon_values, accuracy_values, 
                loss_values, rounds, no_dp_accuracy, dp_accuracy, 
                no_dp_loss, dp_loss)

def create_plots(dataset, comparison_dir, epsilon_values, accuracy_values, 
                loss_values, rounds, no_dp_accuracy, dp_accuracy, 
                no_dp_loss, dp_loss):
    """创建差分隐私比较图表"""
    plt.style.use('ggplot')
    
    # 图1: 隐私-准确率权衡
    plt.figure(figsize=(10, 6))
    # 使用有限的ε值
    finite_indices = [i for i, eps in enumerate(epsilon_values) if eps != float('inf')]
    finite_eps = [epsilon_values[i] for i in finite_indices]
    finite_acc = [accuracy_values[i] for i in finite_indices]
    
    plt.plot(finite_eps, finite_acc, 'o-', linewidth=2)
    plt.xscale('log')
    plt.xlabel('隐私预算 (ε)', fontsize=12)
    plt.ylabel('测试准确率', fontsize=12)
    plt.title(f'{dataset.upper()} 数据集的隐私-准确率权衡', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'privacy_accuracy_tradeoff.png'), dpi=100)
    print(f"已保存隐私-准确率权衡图表")
    plt.close()
    
    # 图2: 准确率比较
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, no_dp_accuracy, '-', label='标准联邦学习', linewidth=2)
    plt.plot(rounds, dp_accuracy, '-', label='差分隐私联邦学习 (ε=1.0)', linewidth=2)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('测试准确率', fontsize=12)
    plt.title(f'{dataset.upper()} 数据集的准确率对比', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'accuracy_comparison.png'), dpi=100)
    print(f"已保存准确率对比图表")
    plt.close()
    
    # 图3: 损失比较
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, no_dp_loss, '-', label='标准联邦学习', linewidth=2)
    plt.plot(rounds, dp_loss, '-', label='差分隐私联邦学习 (ε=1.0)', linewidth=2)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('测试损失', fontsize=12)
    plt.title(f'{dataset.upper()} 数据集的损失对比', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'loss_comparison.png'), dpi=100)
    print(f"已保存损失对比图表")
    plt.close()

def main():
    # 确保结果目录存在
    ensure_dir('results')
    
    # 创建示例数据和图表
    for dataset in ['mnist', 'fashionmnist', 'cifar10']:
        create_example_data(dataset)
    
    print("所有差分隐私对比数据和图表已生成")

if __name__ == "__main__":
    main() 