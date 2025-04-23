import os
import sys
import traceback
import numpy as np

def check_path_exists(path):
    """检查路径是否存在"""
    exists = os.path.exists(path)
    print(f"检查路径 {path}: {'存在' if exists else '不存在'}")
    return exists

def check_file_readable(file_path):
    """检查文件是否可读"""
    if not check_path_exists(file_path):
        return False
    
    try:
        with open(file_path, 'rb') as f:
            data = f.read(10)  # 尝试读取前10个字节
        print(f"文件 {file_path} 可以读取")
        return True
    except Exception as e:
        print(f"文件 {file_path} 读取错误: {str(e)}")
        return False

def check_npz_file(file_path):
    """检查npz文件是否可以加载"""
    if not check_file_readable(file_path):
        return False
    
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"NPZ文件 {file_path} 加载成功，包含键: {list(data.keys())}")
        return True
    except Exception as e:
        print(f"NPZ文件 {file_path} 加载错误: {str(e)}")
        traceback.print_exc()
        return False

def check_dataset_data(dataset):
    """检查特定数据集的数据文件"""
    print(f"\n检查数据集 {dataset} 的文件:")
    
    # 检查数据集目录
    dataset_dir = os.path.join('results', dataset)
    if not check_path_exists(dataset_dir):
        print(f"创建数据集目录 {dataset_dir}")
        os.makedirs(dataset_dir, exist_ok=True)
    
    # 检查比较目录
    comparison_dir = os.path.join(dataset_dir, 'comparison')
    if not check_path_exists(comparison_dir):
        print(f"创建比较目录 {comparison_dir}")
        os.makedirs(comparison_dir, exist_ok=True)
    
    # 检查比较数据文件
    dp_comparison_file = os.path.join(comparison_dir, 'dp_comparison_data.npz')
    check_npz_file(dp_comparison_file)
    
    # 检查图片目录和文件
    image_files = [
        os.path.join(comparison_dir, 'accuracy_comparison.png'),
        os.path.join(comparison_dir, 'loss_comparison.png'),
        os.path.join(comparison_dir, 'privacy_accuracy_tradeoff.png')
    ]
    
    for image_file in image_files:
        check_file_readable(image_file)

def main():
    """主函数"""
    print("=== 开始调试web可视化服务器 ===")
    print("当前工作目录:", os.getcwd())
    
    print("Python版本:", sys.version)
    print("NumPy版本:", np.__version__)
    
    # 检查结果目录
    results_dir = 'results'
    if not check_path_exists(results_dir):
        print(f"创建结果目录 {results_dir}")
        os.makedirs(results_dir, exist_ok=True)
    
    # 检查各数据集
    datasets = ['mnist', 'fashionmnist', 'cifar10', 'quick_test']
    for dataset in datasets:
        check_dataset_data(dataset)
    
    # 检查web_visualizer.py代码
    web_viz_path = 'web_visualizer.py'
    if check_file_readable(web_viz_path):
        print("\n检查web_visualizer.py文件:")
        try:
            with open(web_viz_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 查找handle_static_files方法
            for i, line in enumerate(lines):
                if "def handle_static_files" in line:
                    print(f"找到handle_static_files方法在第{i+1}行")
                    # 打印方法内容
                    method_lines = []
                    j = i + 1
                    while j < len(lines) and (lines[j].startswith('        ') or not lines[j].strip()):
                        method_lines.append(lines[j])
                        j += 1
                    print("方法内容:", "".join(method_lines[:10]) + "..." if len(method_lines) > 10 else "".join(method_lines))
                    break
        except Exception as e:
            print(f"读取web_visualizer.py时出错: {str(e)}")
    
    print("\n=== 调试完成 ===")
    
    # 检查web_visualizer.py中的路径处理代码
    print("\n检查web_visualizer.py中处理静态文件的代码:")
    print("""
    在web_visualizer.py中，静态文件的处理路径是:
    1. self.path[8:]  # Remove '/static/' prefix
    2. file_path = os.path.join(self.results_dir, path)
    
    确保图片文件被正确放置在:
    results/{dataset}/comparison/*.png
    
    当访问URL: /static/{dataset}/comparison/*.png
    服务器会查找: results/{dataset}/comparison/*.png
    """)

if __name__ == "__main__":
    main()
    print("调试脚本执行完毕") 