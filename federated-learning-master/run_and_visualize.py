import os
import sys
import argparse
import subprocess
import time
from plot_results import ResultsVisualizer

def parse_arguments():
    """
    解析命令行参数
    
    返回:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='联邦学习实验运行与可视化工具')
    
    parser.add_argument('--train', action='store_true',
                        help='运行训练过程')
    parser.add_argument('--visualize', action='store_true',
                        help='运行可视化过程')
    parser.add_argument('--datasets', nargs='+', default=['mnist', 'fashionmnist', 'cifar10'],
                        help='要处理的数据集，例如：--datasets mnist fashionmnist cifar10')
    parser.add_argument('--save_plots', action='store_true',
                        help='是否保存生成的图表')
    parser.add_argument('--show_plots', action='store_true',
                        help='是否显示生成的图表')
    parser.add_argument('--plot_path', type=str, default='plots',
                        help='保存图表的路径')
    
    return parser.parse_args()

def update_json_config(datasets):
    """
    更新JSON配置文件，只包含指定的数据集
    
    参数:
        datasets: 要包含的数据集列表
    """
    import json
    
    # 构建对应的结果路径
    log_paths = [f"results/{dataset}/" for dataset in datasets]
    
    # 读取现有配置
    with open('federated_learning.json', 'r') as f:
        config = json.load(f)
    
    # 更新配置
    config['main'][0]['dataset'] = datasets
    config['main'][0]['log_path'] = log_paths
    
    # 写回配置文件
    with open('federated_learning.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"已更新配置文件，包含数据集: {', '.join(datasets)}")

def run_training():
    """
    运行联邦学习训练过程
    """
    print("\n" + "="*80)
    print("启动联邦学习训练...")
    print("="*80 + "\n")
    
    # 使用subprocess运行训练脚本
    start_time = time.time()
    
    try:
        process = subprocess.Popen([sys.executable, 'federated_learning.py'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT,
                                  text=True,
                                  bufsize=1)
        
        # 实时输出进程的标准输出
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"训练过程异常退出，返回代码: {return_code}")
        else:
            elapsed_time = time.time() - start_time
            print(f"\n训练完成！总用时: {elapsed_time:.2f} 秒")
            
    except Exception as e:
        print(f"运行训练过程时出错: {str(e)}")

def run_visualization(args):
    """
    运行结果可视化过程
    
    参数:
        args: 命令行参数
    """
    print("\n" + "="*80)
    print("开始可视化实验结果...")
    print("="*80 + "\n")
    
    try:
        # 创建可视化器
        visualizer = ResultsVisualizer(datasets=args.datasets)
        
        # 打印指标表格
        visualizer.print_metrics_table()
        
        # 绘制准确率对比
        visualizer.plot_accuracy_comparison(
            show=args.show_plots, 
            save=args.save_plots, 
            save_path=args.plot_path
        )
        
        # 绘制损失对比
        visualizer.plot_loss_comparison(
            show=args.show_plots, 
            save=args.save_plots, 
            save_path=args.plot_path
        )
        
        # 绘制每个数据集的详细指标
        visualizer.plot_all_datasets_detailed(
            show=args.show_plots, 
            save=args.save_plots, 
            save_path=args.plot_path
        )
        
        if args.save_plots:
            print(f"已将图表保存到 {args.plot_path} 目录")
            
    except Exception as e:
        print(f"可视化过程出错: {str(e)}")

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果未指定操作，则默认执行训练和可视化
    if not args.train and not args.visualize:
        args.train = True
        args.visualize = True
    
    # 确保至少有一个数据集
    if not args.datasets:
        print("错误：至少需要指定一个数据集")
        return
    
    # 如果要训练，更新配置并运行训练
    if args.train:
        update_json_config(args.datasets)
        run_training()
    
    # 如果要可视化，运行可视化
    if args.visualize:
        # 如果保存图表但未指定是否显示，则默认不显示
        if args.save_plots and not args.show_plots:
            args.show_plots = False
        # 如果未指定是否保存也未指定是否显示，则默认显示
        elif not args.save_plots and not args.show_plots:
            args.show_plots = True
            
        run_visualization(args)
    
    print("\n所有操作已完成！")

if __name__ == "__main__":
    main() 