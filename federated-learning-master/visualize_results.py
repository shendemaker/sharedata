#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于加载训练结果并生成可视化图表的独立脚本
"""

import os
import sys
import argparse
from plot_results import ResultsVisualizer

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练结果可视化工具')
    
    parser.add_argument('--dataset', type=str, default='quick_test',
                        help='要可视化的数据集名称 (默认: quick_test)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='结果文件的根目录 (默认: results)')
    parser.add_argument('--save', action='store_true',
                        help='是否保存生成的图表')
    parser.add_argument('--save_dir', type=str, default='plots',
                        help='保存图表的目录 (默认: plots)')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查结果目录是否存在
    dataset_dir = os.path.normpath(os.path.join(args.results_dir, args.dataset))
    if not os.path.exists(dataset_dir):
        print(f"错误: 结果目录 '{dataset_dir}' 不存在")
        print("请确保已经运行了训练脚本并生成了结果文件")
        return
    
    # 检查是否有结果文件
    result_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]
    if not result_files:
        print(f"错误: 在 '{dataset_dir}' 中未找到结果文件")
        print("请确保已经运行了训练脚本并生成了结果文件")
        return
    
    print(f"找到 {len(result_files)} 个结果文件: {', '.join(result_files)}")
    
    # 创建可视化器
    print("创建可视化器...")
    try:
        visualizer = ResultsVisualizer(datasets=[args.dataset], base_path=args.results_dir)
        
        # 打印指标表格
        print("\n=== 训练结果指标 ===")
        visualizer.print_metrics_table()
        
        # 创建保存图表的目录
        # 确保保存图表的目录使用正确的路径分隔符
        save_dir = os.path.normpath(os.path.join(os.path.abspath(dataset_dir), args.save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 绘制详细指标
        print("\n正在生成可视化图表...")
        visualizer.plot_dataset_metrics(
            args.dataset, 
            show=False,  # 不显示，避免阻塞
            save=True,   # 总是保存
            save_path=save_dir
        )
        
        print(f"图表已保存到 {save_dir} 目录")
        metrics_file = os.path.normpath(os.path.join(save_dir, args.dataset + '_metrics.png'))
        print(f"查看文件: {metrics_file}")
        
        print("\n可视化完成!")
    except Exception as e:
        print(f"可视化过程出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc() 