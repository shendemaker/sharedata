#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速启动联邦学习项目的脚本
使用精简配置进行训练，便于快速测试和验证
"""

import os
import sys
import time
import subprocess
import argparse
import json
import shutil

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='联邦学习快速启动工具')
    
    parser.add_argument('--iterations', type=int, default=100,
                        help='训练迭代次数 (默认: 100)')
    parser.add_argument('--clients', type=int, default=50,
                        help='客户端数量 (默认: 50)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashionmnist', 'cifar10'],
                        help='要使用的数据集 (默认: mnist)')
    parser.add_argument('--visualize', action='store_true',
                        help='训练后是否可视化结果')
    
    return parser.parse_args()

def update_quick_config(args):
    """
    根据命令行参数更新快速配置文件
    
    参数:
        args: 命令行参数
    """
    # 读取快速配置文件
    config_path = 'federated_learning_quick.json'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 更新配置
    config['quick'][0]['dataset'] = [args.dataset]
    config['quick'][0]['iterations'] = [args.iterations]
    config['quick'][0]['n_clients'] = [args.clients]
    
    # 写回配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"已更新快速配置:")
    print(f" - 数据集: {args.dataset}")
    print(f" - 客户端数量: {args.clients}")
    print(f" - 迭代次数: {args.iterations}")

def run_quick_training():
    """运行快速训练过程"""
    print("\n" + "="*80)
    print("启动快速训练...")
    print("="*80 + "\n")
    
    # 备份原始配置文件
    shutil.copy('federated_learning.json', 'federated_learning.json.backup')
    
    try:
        # 将快速配置复制到主配置文件
        with open('federated_learning_quick.json', 'r', encoding='utf-8') as source:
            quick_config = json.load(source)
            
        # 读取原始配置文件，保留其他设置
        with open('federated_learning.json', 'r', encoding='utf-8') as target_file:
            main_config = json.load(target_file)
            
        # 添加或替换快速配置
        main_config['quick'] = quick_config['quick']
        
        # 写入修改后的配置到主配置文件
        with open('federated_learning.json', 'w', encoding='utf-8') as target_file:
            json.dump(main_config, target_file, indent=2)
            
        print("临时更新主配置文件以支持快速启动...")
        
        # 设置环境变量，强制Python使用UTF-8编码
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # 使用subprocess运行训练脚本
        start_time = time.time()
        
        try:
            # 注意使用--schedule quick参数，并设置encoding
            process = subprocess.Popen(
                [sys.executable, 'federated_learning.py', '--schedule', 'quick'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',  # 明确指定编码
                errors='replace',  # 当遇到无法解码的字节时替换为占位符
                bufsize=1,
                env=env
            )
            
            # 实时输出进程的标准输出
            for line in iter(process.stdout.readline, ''):
                try:
                    print(line, end='')
                except UnicodeEncodeError:
                    # 处理控制台输出错误
                    print("[无法显示的字符]", end='')
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code != 0:
                print(f"训练过程异常退出，返回代码: {return_code}")
            else:
                elapsed_time = time.time() - start_time
                print(f"\n快速训练完成！总用时: {elapsed_time:.2f} 秒")
                
        except Exception as e:
            print(f"运行训练过程时出错: {str(e)}")
    
    finally:
        # 恢复原始配置文件
        try:
            shutil.copy('federated_learning.json.backup', 'federated_learning.json')
            if os.path.exists('federated_learning.json.backup'):
                os.remove('federated_learning.json.backup')
            print("已恢复原始配置文件")
        except Exception as e:
            print(f"恢复配置文件时出错: {str(e)}")

def visualize_results():
    """可视化训练结果"""
    try:
        # 确保结果目录存在
        results_path = os.path.join('results', 'quick_test')
        if not os.path.exists(results_path):
            print(f"警告: 结果目录 '{results_path}' 不存在，可能是训练未完成或结果未保存")
            print("跳过可视化步骤")
            return
            
        # 检查目录中是否有结果文件
        result_files = [f for f in os.listdir(results_path) if f.endswith('.npz')]
        if not result_files:
            print(f"警告: 在 '{results_path}' 中未找到结果文件")
            print("跳过可视化步骤")
            return
            
        # 导入可视化器
        try:
            from plot_results import ResultsVisualizer
        except ImportError as e:
            print(f"导入可视化模块时出错: {str(e)}")
            print("请确保plot_results.py文件在当前目录中")
            return
        
        # 创建可视化器实例，指定结果路径
        visualizer = ResultsVisualizer(datasets=['quick_test'], base_path='results')
        
        # 打印指标表格
        visualizer.print_metrics_table()
        
        # 为数据集绘制详细指标
        visualizer.plot_dataset_metrics('quick_test', show=True, save=False)
            
    except ModuleNotFoundError as e:
        print(f"缺少必要的Python模块: {str(e)}")
        print("请确保安装了所有必要的依赖，如matplotlib和numpy")
    except Exception as e:
        print(f"可视化过程出错: {str(e)}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 更新快速配置
    update_quick_config(args)
    
    # 运行快速训练
    run_quick_training()
    
    # 如果需要，可视化结果
    if args.visualize:
        print("\n" + "="*80)
        print("开始可视化实验结果...")
        print("="*80 + "\n")
        visualize_results()
    
    print("\n快速启动完成！")

if __name__ == "__main__":
    main() 