import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import experiment_manager as xpm

class ResultsVisualizer:
    """
    用于可视化联邦学习实验结果的类
    """
    def __init__(self, datasets=None, base_path="results"):
        """
        初始化可视化器
        
        参数:
            datasets: 要可视化的数据集列表
            base_path: 结果文件的基础路径
        """
        if datasets is None:
            self.datasets = ["mnist", "fashionmnist", "cifar10"]
        else:
            self.datasets = datasets
        
        self.base_path = base_path
        self.experiments = {}
        
        # 设置中文字体
        self._setup_font()
        
        # 加载实验结果
        self._load_experiments()
        
    def _setup_font(self):
        """设置适合中文显示的字体"""
        # 尝试加载常见的中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi', 'Arial Unicode MS']
        for font_name in chinese_fonts:
            try:
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                if os.path.exists(font_path):
                    self.font = FontProperties(family=font_name)
                    print(f"使用中文字体: {font_name}")
                    return
            except:
                continue
                
        # 如果找不到中文字体，使用默认字体但在使用时避免中文
        print("警告: 未找到支持中文的字体，将使用默认字体并避免使用中文标签")
        self.font = FontProperties()
        self.use_english = True
        
    def _load_experiments(self):
        """
        加载所有数据集的实验结果
        """
        for dataset in self.datasets:
            # 使用os.path.join构建路径，然后用normpath标准化
            path = os.path.normpath(os.path.join(self.base_path, dataset))
            
            # 检查路径是否存在
            if not os.path.exists(path):
                print(f"路径 {path} 不存在，跳过...")
                continue
                
            try:
                # 获取该数据集的所有实验
                experiments = xpm.get_list_of_experiments(path, only_finished=False, verbose=False)
                if experiments:
                    self.experiments[dataset] = experiments
                    print(f"已加载 {dataset} 的 {len(experiments)} 个实验结果")
                else:
                    print(f"未找到 {dataset} 的已完成实验")
            except Exception as e:
                print(f"加载 {dataset} 的实验时出错: {str(e)}")
    
    def print_metrics_table(self):
        """打印指标表格"""
        if not self.experiments:
            print("没有可用的实验结果可供显示")
            return
            
        for dataset, exps in self.experiments.items():
            if exps:
                print(f"\n--- {self._get_dataset_display_name(dataset)} 数据集 ---")
                exp = exps[0]  # 使用第一个实验
                
                print(f"测试准确率: {exp.results.get('accuracy_test', ['N/A'])[-1]:.4f}")
                print(f"测试损失: {exp.results.get('loss_test', ['N/A'])[-1]:.4f}")
                print(f"训练轮数: {len(exp.results.get('communication_round', []))}")
                
                # 提取客户端数量
                client_keys = [k for k in exp.results.keys() if k.startswith('client') and k.endswith('_loss')]
                print(f"客户端数量: {len(client_keys)}")
                print("----------------------------")
    
    def _get_dataset_display_name(self, dataset):
        """获取数据集的显示名称，处理特殊名称"""
        dataset_map = {
            "mnist": "MNIST",
            "fashionmnist": "Fashion-MNIST",
            "cifar10": "CIFAR-10",
            "quick_test": "Quick Test",
            "trash": "Trash"
        }
        return dataset_map.get(dataset, dataset)
    
    def plot_accuracy_comparison(self, show=True, save=False, save_path="plots"):
        """
        比较不同数据集的测试准确率
        
        参数:
            show: 是否显示图形
            save: 是否保存图形
            save_path: 保存图形的路径
        """
        if not self.experiments:
            print("没有可用的实验结果可供可视化")
            return
            
        plt.figure(figsize=(12, 8))
        
        for dataset, exps in self.experiments.items():
            for exp in exps:
                if 'accuracy_test' in exp.results:
                    rounds = exp.results.get('communication_round', 
                                           np.arange(len(exp.results['accuracy_test'])))
                    plt.plot(rounds, exp.results['accuracy_test'], 
                             label=f"{self._get_dataset_display_name(dataset)}", linewidth=2)
                    break  # 每个数据集只取一个实验
        
        # 设置图表标题和标签，根据字体支持情况使用中文或英文
        if hasattr(self, 'use_english') and self.use_english:
            plt.title("Test Accuracy Comparison", fontsize=16)
            plt.xlabel("Communication Rounds", fontsize=14)
            plt.ylabel("Accuracy", fontsize=14)
        else:
            plt.title("不同数据集的测试准确率对比", fontproperties=self.font, fontsize=16)
            plt.xlabel("通信轮数", fontproperties=self.font, fontsize=14)
            plt.ylabel("准确率", fontproperties=self.font, fontsize=14)
            
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            # 标准化路径
            save_file = os.path.join(save_path, "accuracy_comparison.png")
            save_file = os.path.normpath(save_file)
            
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"已保存图表到: {save_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_comparison(self, show=True, save=False, save_path="plots"):
        """
        比较不同数据集的测试损失
        
        参数:
            show: 是否显示图形
            save: 是否保存图形
            save_path: 保存图形的路径
        """
        if not self.experiments:
            print("没有可用的实验结果可供可视化")
            return
            
        plt.figure(figsize=(12, 8))
        
        for dataset, exps in self.experiments.items():
            for exp in exps:
                if 'loss_test' in exp.results:
                    rounds = exp.results.get('communication_round', 
                                           np.arange(len(exp.results['loss_test'])))
                    plt.plot(rounds, exp.results['loss_test'], 
                             label=f"{self._get_dataset_display_name(dataset)}", linewidth=2)
                    break  # 每个数据集只取一个实验
        
        # 设置图表标题和标签，根据字体支持情况使用中文或英文
        if hasattr(self, 'use_english') and self.use_english:
            plt.title("Test Loss Comparison", fontsize=16)
            plt.xlabel("Communication Rounds", fontsize=14)
            plt.ylabel("Loss", fontsize=14)
        else:
            plt.title("不同数据集的测试损失对比", fontproperties=self.font, fontsize=16)
            plt.xlabel("通信轮数", fontproperties=self.font, fontsize=14)
            plt.ylabel("损失", fontproperties=self.font, fontsize=14)
            
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            # 标准化路径
            save_file = os.path.join(save_path, "loss_comparison.png")
            save_file = os.path.normpath(save_file)
            
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"已保存图表到: {save_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_all_datasets_detailed(self, show=True, save=False, save_path="plots"):
        """
        为所有数据集绘制详细指标
        
        参数:
            show: 是否显示图形
            save: 是否保存图形
            save_path: 保存图形的路径
        """
        for dataset in self.datasets:
            if dataset in self.experiments and self.experiments[dataset]:
                self.plot_dataset_metrics(dataset, show=show, save=save, save_path=save_path)
    
    def plot_dataset_metrics(self, dataset, show=True, save=False, save_path="plots"):
        """
        为特定数据集绘制详细的指标图表
        
        参数:
            dataset: 数据集名称
            show: 是否显示图表
            save: 是否保存图表
            save_path: 保存图表的路径
        """
        if dataset not in self.experiments or not self.experiments[dataset]:
            print(f"没有找到 {dataset} 的实验结果")
            return
            
        # 获取实验结果
        exp = self.experiments[dataset][0]
        
        # 创建一个包含4个子图的大图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 设置标题
        if hasattr(self, 'use_english') and self.use_english:
            fig.suptitle(f"{self._get_dataset_display_name(dataset)} Dataset Metrics", fontsize=18)
        else:
            fig.suptitle(f"{self._get_dataset_display_name(dataset)} 数据集训练指标", fontproperties=self.font, fontsize=18)
        
        # 测试准确率
        if 'accuracy_test' in exp.results:
            rounds = exp.results.get('communication_round', 
                                   np.arange(len(exp.results['accuracy_test'])))
            axes[0, 0].plot(rounds, exp.results['accuracy_test'], 'bo-', linewidth=2)
            
            # 设置标题和标签，根据字体支持情况使用中文或英文
            if hasattr(self, 'use_english') and self.use_english:
                axes[0, 0].set_title("Test Accuracy", fontsize=14)
                axes[0, 0].set_xlabel("Communication Rounds", fontsize=12)
                axes[0, 0].set_ylabel("Accuracy", fontsize=12)
            else:
                axes[0, 0].set_title("测试准确率", fontproperties=self.font, fontsize=14)
                axes[0, 0].set_xlabel("通信轮数", fontproperties=self.font, fontsize=12)
                axes[0, 0].set_ylabel("准确率", fontproperties=self.font, fontsize=12)
                
            axes[0, 0].grid(True, linestyle='--', alpha=0.7)
            
        # 测试损失
        if 'loss_test' in exp.results:
            rounds = exp.results.get('communication_round', 
                                   np.arange(len(exp.results['loss_test'])))
            axes[0, 1].plot(rounds, exp.results['loss_test'], 'ro-', linewidth=2)
            
            # 设置标题和标签，根据字体支持情况使用中文或英文
            if hasattr(self, 'use_english') and self.use_english:
                axes[0, 1].set_title("Test Loss", fontsize=14)
                axes[0, 1].set_xlabel("Communication Rounds", fontsize=12)
                axes[0, 1].set_ylabel("Loss", fontsize=12)
            else:
                axes[0, 1].set_title("测试损失", fontproperties=self.font, fontsize=14)
                axes[0, 1].set_xlabel("通信轮数", fontproperties=self.font, fontsize=12)
                axes[0, 1].set_ylabel("损失", fontproperties=self.font, fontsize=12)
                
            axes[0, 1].grid(True, linestyle='--', alpha=0.7)
            
        # 训练准确率
        if 'accuracy_train' in exp.results:
            rounds = exp.results.get('communication_round', 
                                   np.arange(len(exp.results['accuracy_train'])))
            axes[1, 0].plot(rounds, exp.results['accuracy_train'], 'go-', linewidth=2)
            
            # 设置标题和标签，根据字体支持情况使用中文或英文
            if hasattr(self, 'use_english') and self.use_english:
                axes[1, 0].set_title("Train Accuracy", fontsize=14)
                axes[1, 0].set_xlabel("Communication Rounds", fontsize=12)
                axes[1, 0].set_ylabel("Accuracy", fontsize=12)
            else:
                axes[1, 0].set_title("训练准确率", fontproperties=self.font, fontsize=14)
                axes[1, 0].set_xlabel("通信轮数", fontproperties=self.font, fontsize=12)
                axes[1, 0].set_ylabel("准确率", fontproperties=self.font, fontsize=12)
                
            axes[1, 0].grid(True, linestyle='--', alpha=0.7)
            
        # 训练损失
        if 'loss_train' in exp.results:
            rounds = exp.results.get('communication_round', 
                                   np.arange(len(exp.results['loss_train'])))
            axes[1, 1].plot(rounds, exp.results['loss_train'], 'mo-', linewidth=2)
            
            # 设置标题和标签，根据字体支持情况使用中文或英文
            if hasattr(self, 'use_english') and self.use_english:
                axes[1, 1].set_title("Train Loss", fontsize=14)
                axes[1, 1].set_xlabel("Communication Rounds", fontsize=12)
                axes[1, 1].set_ylabel("Loss", fontsize=12)
            else:
                axes[1, 1].set_title("训练损失", fontproperties=self.font, fontsize=14)
                axes[1, 1].set_xlabel("通信轮数", fontproperties=self.font, fontsize=12)
                axes[1, 1].set_ylabel("损失", fontproperties=self.font, fontsize=12)
                
            axes[1, 1].grid(True, linestyle='--', alpha=0.7)
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图形
        if save:
            # 创建保存目录
            if save_path.startswith(self.base_path):
                # 确保路径不包含重复的base_path
                save_dir = os.path.normpath(save_path)
            else:
                save_dir = os.path.normpath(os.path.join(self.base_path, dataset, save_path))
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 构建保存文件名，并标准化路径
            save_file = os.path.join(save_dir, f"{dataset}_metrics.png")
            save_file = os.path.normpath(save_file)
            
            # 保存图表
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"已保存图表到: {save_file}")
        
        # 显示图形
        if show:
            plt.show()
        else:
            plt.close(fig)

# 主函数示例
if __name__ == "__main__":
    # 创建可视化器实例
    visualizer = ResultsVisualizer()
    
    # 打印最终指标表格
    visualizer.print_metrics_table()
    
    # 绘制不同数据集的准确率对比
    visualizer.plot_accuracy_comparison(save=True)
    
    # 绘制不同数据集的损失对比
    visualizer.plot_loss_comparison(save=True)
    
    # 为每个数据集绘制详细指标
    visualizer.plot_all_datasets_detailed(save=True) 