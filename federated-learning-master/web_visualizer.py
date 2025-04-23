#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于在浏览器中展示联邦学习实验结果的Web可视化服务器
"""

import os
import sys
import argparse
import json
import http.server
import socketserver
import webbrowser
from urllib.parse import parse_qs
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from plot_results import ResultsVisualizer
import experiment_manager as xpm
import traceback
import urllib.parse
import io
import base64

# HTML模板 - 使用英文避免字符编码问题
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Federated Learning Experiment Results Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 5px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: #3498db;
            margin-top: 30px;
        }
        .nav {
            display: flex;
            background-color: #3498db;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .nav a {
            color: white;
            text-decoration: none;
            padding: 8px 15px;
            margin-right: 5px;
            border-radius: 3px;
        }
        .nav a:hover, .nav a.active {
            background-color: #2980b9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }
        .metrics {
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: inset 0 0 5px rgba(0,0,0,0.1);
        }
        footer {
            text-align: center;
            margin-top: 30px;
            padding: 10px;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .error {
            color: #e74c3c;
            padding: 15px;
            background-color: #fadbd8;
            border-radius: 5px;
            margin: 20px 0;
        }
        .dp-comparison-link {
            text-align: center;
            margin: 20px 0;
        }
        
        .button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #4285f4;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        
        .button:hover {
            background-color: #3367d6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Federated Learning Experiment Results Visualization</h1>
        
        <div class="nav">
            {dataset_links}
        </div>
        
        {content}
        
        <footer>
            <p>Federated Learning Experiment Results Visualization Tool © 2023</p>
        </footer>
    </div>
</body>
</html>
"""

class FederatedLearningHandler(http.server.SimpleHTTPRequestHandler):
    """处理HTTP请求的自定义处理器"""
    
    def __init__(self, *args, results_dir='results', **kwargs):
        self.results_dir = results_dir
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """处理GET请求"""
        try:
            if self.path == '/' or self.path == '/index.html':
                self.serve_main_page()
            elif self.path.startswith('/dataset/'):
                dataset = self.path.split('/')[2].split('?')[0]
                self.serve_dataset_page(dataset)
            # 处理图片和其他静态文件
            elif self.path.startswith('/plots/') or self.path.endswith('.png'):
                self.serve_file(self.path[1:])  # 去掉开头的斜杠
            elif self.path.startswith('/results/'):  # 添加对/results/路径的处理
                self.serve_file(self.path[1:])
            elif self.path.startswith('/dp_comparison'):
                self.handle_dp_comparison()
            elif self.path.startswith('/experiment_data'):
                self.handle_experiment_data()
            elif self.path.startswith('/static'):
                self.handle_static_files()
            elif self.path.startswith('/direct-image'):
                self.handle_direct_image()
            else:
                self.send_error(404, "File Not Found")
        except Exception as e:
            print(f"Error handling request: {str(e)}")
            self.send_error(500, "Internal Server Error")
    
    def serve_main_page(self):
        """提供主页面"""
        try:
            # 获取可用的数据集
            datasets = self._get_available_datasets()
            
            if not datasets:
                content = """
                <div class="error">
                    <h2>Error</h2>
                    <p>No experiment results found. Please run federated learning experiments first, then use this visualization tool.</p>
                </div>
                """
                dataset_links = '<a href="/" class="active">Home</a>'
            else:
                # 创建数据集导航链接
                dataset_links = '<a href="/" class="active">Home</a>'
                for ds in datasets:
                    dataset_links += f'<a href="/dataset/{ds}">{self._get_display_name(ds)}</a>'
                
                # 创建内容
                content = f"""
                <h2>Welcome to Federated Learning Experiment Results Visualization Tool</h2>
                <p>Currently there are {len(datasets)} datasets with experiment results available to view. Please select a dataset from the navigation bar above.</p>
                
                <h2>Available Datasets</h2>
                <table>
                    <tr>
                        <th>Dataset</th>
                        <th>Number of Experiments</th>
                        <th>Action</th>
                    </tr>
                """
                
                for ds in datasets:
                    exp_count = self._count_experiments(ds)
                    content += f"""
                    <tr>
                        <td>{self._get_display_name(ds)}</td>
                        <td>{exp_count}</td>
                        <td><a href="/dataset/{ds}">View Details</a></td>
                    </tr>
                    """
                
                content += "</table>"
            
            try:
                # 直接复制模板并替换占位符
                html = HTML_TEMPLATE
                html = html.replace("{dataset_links}", dataset_links)
                html = html.replace("{content}", content)
                
                # 发送响应
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            except Exception as e:
                print(f"Error in HTML template processing: {str(e)}")
                import traceback
                traceback.print_exc()
                self.send_error(500, "Template Processing Error")
            
        except Exception as e:
            print(f"Error serving main page: {str(e)}")
            import traceback
            traceback.print_exc()
            self.send_error(500, "Internal Server Error")
    
    def _get_display_name(self, dataset):
        """获取数据集的显示名称"""
        dataset_map = {
            "mnist": "MNIST",
            "fashionmnist": "Fashion-MNIST",
            "cifar10": "CIFAR-10",
            "quick_test": "Quick Test",
            "trash": "Trash"
        }
        return dataset_map.get(dataset, dataset)
    
    def serve_dataset_page(self, dataset):
        """提供数据集详情页面"""
        try:
            # 获取可用的数据集
            datasets = self._get_available_datasets()
            
            if dataset not in datasets:
                self.send_error(404, f"Dataset not found: {dataset}")
                return
            
            # 创建数据集导航链接
            dataset_links = '<a href="/">Home</a>'
            for ds in datasets:
                if ds == dataset:
                    dataset_links += f'<a href="/dataset/{ds}" class="active">{self._get_display_name(ds)}</a>'
                else:
                    dataset_links += f'<a href="/dataset/{ds}">{self._get_display_name(ds)}</a>'
            
            # 加载实验结果
            try:
                # 确保plots目录存在
                plots_dir = os.path.join(self.results_dir, 'plots', dataset)
                plots_dir = os.path.normpath(plots_dir)
                
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
                
                # 生成图表
                plot_file = os.path.join(plots_dir, f"{dataset}_metrics.png")
                plot_file = os.path.normpath(plot_file)
                
                if not os.path.exists(plot_file):
                    # 创建可视化器
                    visualizer = ResultsVisualizer(datasets=[dataset], base_path=self.results_dir)
                    
                    # 生成图表
                    visualizer.plot_dataset_metrics(
                        dataset, 
                        show=False, 
                        save=True,
                        save_path=plots_dir
                    )
                
                # 获取指标数据
                metrics = self._get_metrics(dataset)
                
                # 创建内容
                content = f"""
                <h2>{self._get_display_name(dataset)} Dataset Experiment Results</h2>
                
                <div class="metrics">
                    <h3>Training Metrics Summary</h3>
                    {'<p class="error">Note: Showing example data, real experiment results not found</p>' if metrics.get('is_example', False) else ''}
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Final Test Accuracy</td>
                            <td>{metrics.get('final_test_acc', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Min Test Loss</td>
                            <td>{metrics.get('min_test_loss', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Training Rounds</td>
                            <td>{metrics.get('total_rounds', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Number of Clients</td>
                            <td>{metrics.get('num_clients', 'N/A')}</td>
                        </tr>
                    </table>
                </div>
                
                <h3>Performance Metrics Charts</h3>
                
                <div class="plot-container">
                    <img src="/plots/{dataset}/{dataset}_metrics.png" alt="{self._get_display_name(dataset)} Training Metrics">
                </div>
                
                <div class="dp-comparison-link">
                    <a href="/dp_comparison?dataset={dataset}" class="button">查看差分隐私对比</a>
                </div>
                """
                
            except Exception as e:
                error_msg = str(e)
                # 确保错误消息是ASCII编码兼容的
                error_msg = ''.join(c if ord(c) < 128 else '?' for c in error_msg)
                content = f"""
                <div class="error">
                    <h2>Error loading {self._get_display_name(dataset)} dataset</h2>
                    <p>Error message: {error_msg}</p>
                </div>
                """
            
            # 替换HTML模板中的占位符
            html = HTML_TEMPLATE.replace("{dataset_links}", dataset_links).replace("{content}", content)
            
            # 发送响应
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            
        except Exception as e:
            error_msg = str(e)
            # 确保错误消息是ASCII编码兼容的
            error_msg = ''.join(c if ord(c) < 128 else '?' for c in error_msg)
            print(f"Error serving dataset page: {error_msg}")
            self.send_error(500, "Internal Server Error")
    
    def serve_file(self, path):
        """提供静态文件"""
        try:
            # 确保路径安全，防止目录遍历攻击
            if '..' in path:
                self.send_error(403, "Access Forbidden")
                return
                
            # 构建文件的完整路径
            # 检查是否是plots路径，如果是，需要特殊处理
            if path.startswith('plots/'):
                # 从plots/dataset/file.png格式提取信息
                parts = path.split('/')
                if len(parts) >= 3:
                    dataset = parts[1]
                    filename = parts[2]
                    # 构建正确的路径
                    file_path = os.path.join(self.results_dir, 'plots', dataset, filename)
                else:
                    file_path = os.path.join(self.results_dir, path)
            else:
                file_path = os.path.join(self.results_dir, path)
                
            file_path = os.path.normpath(file_path)
            
            # 检查文件是否存在
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                # 尝试查找替代路径
                alt_paths = [
                    os.path.join(self.results_dir, path),  # 标准路径
                    os.path.join(self.results_dir, 'plots', path.replace('plots/', '')),  # 替代路径1
                    os.path.join(self.results_dir, path.replace('plots/', ''))  # 替代路径2
                ]
                
                found = False
                for alt_path in alt_paths:
                    alt_path = os.path.normpath(alt_path)
                    if os.path.exists(alt_path) and os.path.isfile(alt_path):
                        file_path = alt_path
                        found = True
                        break
                
                if not found:
                    self.send_error(404, f"File Not Found: {path}")
                    return
            
            # 确定文件类型
            if file_path.endswith('.png'):
                content_type = 'image/png'
            elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                content_type = 'image/jpeg'
            else:
                content_type = 'application/octet-stream'
            
            # 发送文件
            with open(file_path, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.end_headers()
                self.wfile.write(f.read())
                
        except Exception as e:
            error_msg = str(e)
            # 确保错误消息是ASCII编码兼容的
            error_msg = ''.join(c if ord(c) < 128 else '?' for c in error_msg)
            print(f"Error serving file: {error_msg}")
            self.send_error(500, "Internal Server Error")
    
    def _get_available_datasets(self):
        """获取可用的数据集"""
        try:
            # 默认数据集列表，确保这些数据集显示在界面上
            default_datasets = ["mnist", "fashionmnist", "cifar10"]
            detected_datasets = []
            
            results_path = self.results_dir
            
            # 检查结果目录是否存在
            if os.path.exists(results_path) and os.path.isdir(results_path):
                # 遍历子目录
                for item in os.listdir(results_path):
                    item_path = os.path.join(results_path, item)
                    if os.path.isdir(item_path):
                        # 检查目录中是否有.npz文件
                        if any(f.endswith('.npz') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))):
                            detected_datasets.append(item)
            
            # 合并检测到的数据集和默认数据集
            all_datasets = list(set(detected_datasets + default_datasets))
            
            # 确保返回的数据集在results目录中存在子目录
            valid_datasets = []
            for ds in all_datasets:
                ds_path = os.path.join(results_path, ds)
                if not os.path.exists(ds_path):
                    os.makedirs(ds_path)
                valid_datasets.append(ds)
                
            return sorted(valid_datasets)
        except Exception as e:
            print(f"Error getting available datasets: {str(e)}")
            return []
    
    def _count_experiments(self, dataset):
        """统计数据集的实验数量"""
        try:
            dataset_path = os.path.join(self.results_dir, dataset)
            if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
                return len([f for f in os.listdir(dataset_path) if f.endswith('.npz') and os.path.isfile(os.path.join(dataset_path, f))])
            return 0
        except Exception as e:
            print(f"Error counting experiments: {str(e)}")
            return 0
    
    def _get_metrics(self, dataset):
        """获取数据集的指标数据"""
        try:
            # 加载实验结果
            dataset_path = os.path.join(self.results_dir, dataset)
            dataset_path = os.path.normpath(dataset_path)
            
            # 查找.npz文件
            npz_files = [f for f in os.listdir(dataset_path) if f.endswith('.npz') and os.path.isfile(os.path.join(dataset_path, f))]
            if not npz_files:
                # 如果没有找到实验结果文件，返回示例数据
                print(f"No experiment files found for {dataset}, using example data")
                return self._get_example_metrics()
            
            # 加载第一个实验结果
            try:
                print(f"加载 {dataset} 的实验结果...")
                
                # 直接加载npz文件
                file_path = os.path.join(dataset_path, npz_files[0])
                data = np.load(file_path, allow_pickle=True)
                
                # 提取指标
                metrics = {}
                
                # 检查数据结构
                if 'results' in data:
                    # 新结构：通过results字典访问数据
                    results = data['results'].item()
                    
                    if 'accuracy_test' in results:
                        if isinstance(results['accuracy_test'], np.ndarray) and len(results['accuracy_test']) > 0:
                            metrics['final_test_acc'] = f"{results['accuracy_test'][-1]:.4f}"
                    
                    if 'loss_test' in results:
                        if isinstance(results['loss_test'], np.ndarray) and len(results['loss_test']) > 0:
                            metrics['min_test_loss'] = f"{min(results['loss_test']):.4f}"
                    
                    if 'communication_round' in results:
                        metrics['total_rounds'] = len(results['communication_round'])
                
                    # 客户端数量
                    client_keys = [k for k in results.keys() if k.startswith('client') and k.endswith('_loss')]
                    metrics['num_clients'] = len(client_keys)
                
                else:
                    # 旧结构：尝试直接访问键
                    if 'accuracy_test' in data:
                        if isinstance(data['accuracy_test'], np.ndarray) and len(data['accuracy_test']) > 0:
                            metrics['final_test_acc'] = f"{data['accuracy_test'][-1]:.4f}"
                    
                    if 'loss_test' in data:
                        if isinstance(data['loss_test'], np.ndarray) and len(data['loss_test']) > 0:
                            metrics['min_test_loss'] = f"{min(data['loss_test']):.4f}"
                    
                    if 'communication_round' in data:
                        metrics['total_rounds'] = len(data['communication_round'])
                    
                    # 客户端数量
                    client_keys = [k for k in data.keys() if k.startswith('client') and k.endswith('_loss')]
                    metrics['num_clients'] = len(client_keys)
                
                # 从超参数中获取客户端数量
                if 'hyperparameters' in data:
                    hyperparams = data['hyperparameters'].item()
                    if 'n_clients' in hyperparams:
                        metrics['num_clients'] = hyperparams['n_clients']
                
                # 确保有至少一个指标，否则使用示例数据
                if not metrics or 'final_test_acc' not in metrics:
                    print(f"没有找到有效的指标数据，使用示例数据")
                    return self._get_example_metrics()
                
                return metrics
                
            except Exception as e:
                print(f"加载实验数据时出错: {str(e)}")
                traceback.print_exc()
                return self._get_example_metrics()
                
        except Exception as e:
            print(f"获取指标数据时出错: {str(e)}")
            return self._get_example_metrics()
        
    def _get_example_metrics(self):
        """返回示例指标数据，在无法读取真实数据时使用"""
        return {
            'final_test_acc': '0.9234',
            'min_test_loss': '0.2156',
            'total_rounds': '100',
            'num_clients': '50',
            'is_example': True  # 标记为示例数据
        }

    def handle_dp_comparison(self):
        """Handle differential privacy comparison page"""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        dataset = query.get('dataset', ['mnist'])[0]
        
        # Check if comparison data exists
        comparison_dir = os.path.join(self.results_dir, dataset, 'comparison')
        data_file = os.path.join(comparison_dir, 'dp_comparison_data.npz')
        
        if not os.path.exists(data_file):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_dp_comparison_error(dataset).encode())
            return
        
        # Load comparison data
        try:
            data = np.load(data_file, allow_pickle=True)
            baseline = data['baseline'].item()
            dp_experiments = data['dp_experiments']
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            self.wfile.write(self.get_dp_comparison_page(dataset, baseline, dp_experiments).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>".encode())
    
    def handle_experiment_data(self):
        """Handle experiment data request"""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        dataset = query.get('dataset', ['mnist'])[0]
        filename = query.get('file', [''])[0]
        
        if not dataset or not filename:
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Missing dataset or filename parameter')
            return
        
        # Load experiment data
        file_path = os.path.join(self.results_dir, dataset, filename)
        if not os.path.exists(file_path):
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f'File not found: {file_path}'.encode())
            return
        
        try:
            data = np.load(file_path, allow_pickle=True)
            results = data['results'].item() if 'results' in data else {}
            hyperparameters = data['hyperparameters'].item() if 'hyperparameters' in data else {}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'results': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()},
                'hyperparameters': hyperparameters
            }
            
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f'Error loading data: {str(e)}'.encode())
    
    def handle_static_files(self):
        """Handle static file requests"""
        path = self.path[8:]  # Remove '/static/' prefix
        file_path = os.path.join(self.results_dir, path)
        
        print(f"Serving static file: {path}")
        print(f"Full path: {file_path}")
        
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            # 尝试其他可能的路径
            alternate_paths = [
                os.path.join(self.results_dir, path),
                os.path.join(os.path.dirname(__file__), 'static', path),
                os.path.join(os.path.dirname(__file__), path)
            ]
            
            for alt_path in alternate_paths:
                print(f"Trying alternate path: {alt_path}")
                if os.path.exists(alt_path) and os.path.isfile(alt_path):
                    file_path = alt_path
                    print(f"Found file at: {file_path}")
                    break
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(f'File not found: {path}'.encode())
                return
        
        # Determine content type based on file extension
        content_type = 'text/plain'
        if file_path.endswith('.png'):
            content_type = 'image/png'
        elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
            content_type = 'image/jpeg'
        elif file_path.endswith('.html'):
            content_type = 'text/html'
        elif file_path.endswith('.css'):
            content_type = 'text/css'
        elif file_path.endswith('.js'):
            content_type = 'application/javascript'
        
        # Serve the file
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.send_header('Content-length', len(file_data))
                self.send_header('Cache-Control', 'max-age=3600')  # Cache for an hour
                self.end_headers()
                self.wfile.write(file_data)
                print(f"Successfully served file: {file_path} ({len(file_data)} bytes)")
        except Exception as e:
            print(f"Error serving file {file_path}: {str(e)}")
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f'Error reading file: {str(e)}'.encode())
    
    def get_dp_comparison_error(self, dataset):
        """Generate error page for missing differential privacy comparison data"""
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Differential Privacy Comparison - {dataset}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                .error {{ color: #e74c3c; }}
                a {{ color: #3498db; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>Differential Privacy Comparison - {dataset.upper()}</h1>
            
            <div class="error">
                <h2>Data Not Found</h2>
                <p>No differential privacy comparison data found for {dataset}.</p>
                <p>To generate comparison data, run:</p>
                <pre>python compare_dp.py</pre>
            </div>
            
            <p><a href="/">Back to Home</a></p>
        </body>
        </html>
        '''
    
    def get_dp_comparison_page(self, dataset, baseline, dp_experiments):
        """Generate differential privacy comparison page"""
        # Generate accuracy table
        accuracy_table = ""
        baseline_acc = baseline['accuracy_test'][-1]
        
        # Add baseline row
        accuracy_table += f'''
        <tr>
            <td>No Differential Privacy</td>
            <td>∞</td>
            <td>{baseline_acc:.4f}</td>
            <td>0.00%</td>
        </tr>
        '''
        
        # Add rows for each DP experiment
        for exp in dp_experiments:
            eps = exp['epsilon']
            acc = exp['accuracy_test'][-1]
            acc_loss = ((baseline_acc - acc) / baseline_acc) * 100
            
            accuracy_table += f'''
            <tr>
                <td>DP (ε={eps:.2f})</td>
                <td>{eps:.2f}</td>
                <td>{acc:.4f}</td>
                <td>{acc_loss:.2f}%</td>
            </tr>
            '''
        
        # 直接引用图片文件
        privacy_accuracy_path = f"/results/{dataset}/comparison/privacy_accuracy_tradeoff.png"
        accuracy_comparison_path = f"/results/{dataset}/comparison/accuracy_comparison.png"
        loss_comparison_path = f"/results/{dataset}/comparison/loss_comparison.png"
        
        # Generate full HTML
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Differential Privacy Comparison - {dataset}</title>
            <style>
                body {{ font-family: Arial, 'Microsoft YaHei', 'SimHei', 'SimSun', sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                .section {{ margin-bottom: 40px; }}
                .plot-container {{ margin: 20px 0; text-align: center; }}
                .plot {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                a {{ color: #3498db; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>Differential Privacy Comparison - {dataset.upper()}</h1>
            
            <div class="section">
                <h2>Privacy-Accuracy Trade-off</h2>
                <p>
                    This analysis shows how differential privacy affects model accuracy. 
                    Higher privacy (lower ε) typically results in lower accuracy due to added noise.
                </p>
                
                <div class="plot-container">
                    <img class="plot" src="{privacy_accuracy_path}" alt="Privacy-Accuracy Tradeoff">
                </div>
            </div>
            
            <div class="section">
                <h2>Accuracy Comparison</h2>
                <div class="plot-container">
                    <img class="plot" src="{accuracy_comparison_path}" alt="Accuracy Comparison">
                </div>
            </div>
            
            <div class="section">
                <h2>Loss Comparison</h2>
                <div class="plot-container">
                    <img class="plot" src="{loss_comparison_path}" alt="Loss Comparison">
                </div>
            </div>
            
            <div class="section">
                <h2>Privacy Settings Summary</h2>
                <table>
                    <tr>
                        <th>Configuration</th>
                        <th>Privacy Budget (ε)</th>
                        <th>Final Accuracy</th>
                        <th>Accuracy Loss</th>
                    </tr>
                    {accuracy_table}
                </table>
            </div>
            
            <p><a href="/">Back to Home</a></p>
        </body>
        </html>
        '''

    def handle_direct_image(self):
        """Handle direct image requests"""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        img_path = query.get('path', [''])[0]
        
        if not img_path:
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Missing path parameter')
            return
        
        # Sanitize image path for security
        img_path = img_path.replace('..', '').strip('/')
        full_path = os.path.join(self.results_dir, img_path)
        
        print(f"Direct image request: {img_path}")
        print(f"Full path: {full_path}")
        
        # 尝试查找图片的多个可能路径
        possible_paths = [
            # 原始路径
            full_path,
            # 尝试绝对路径
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', img_path),
            # 尝试相对于当前工作目录的路径
            os.path.join('results', img_path),
            # 尝试相对于results目录的路径
            os.path.join(self.results_dir, img_path.split('/')[-1]),
            # 直接使用文件名（针对落在comparison目录下的图片）
            os.path.join(self.results_dir, img_path.split('/')[0], 'comparison', img_path.split('/')[-1])
        ]
        
        found_path = None
        for path in possible_paths:
            print(f"Trying path: {path}")
            if os.path.exists(path) and os.path.isfile(path):
                found_path = path
                print(f"Found image at: {found_path}")
                break
        
        if not found_path:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f'Image not found: {img_path}, tried paths: {", ".join(possible_paths)}'.encode())
            return
        
        # Determine content type
        _, ext = os.path.splitext(found_path)
        content_type = 'image/png' if ext.lower() == '.png' else 'image/jpeg' if ext.lower() in ['.jpg', '.jpeg'] else 'application/octet-stream'
        
        # Serve the image
        try:
            with open(found_path, 'rb') as f:
                img_data = f.read()
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.send_header('Content-length', len(img_data))
                self.send_header('Cache-Control', 'max-age=3600')  # Cache for an hour
                self.end_headers()
                self.wfile.write(img_data)
                print(f"Successfully served image: {found_path} ({len(img_data)} bytes)")
        except Exception as e:
            print(f"Error serving image {found_path}: {str(e)}")
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f'Error reading image: {str(e)}'.encode())

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Federated Learning Experiment Results Web Visualization Tool')
    
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Root directory for result files (default: results)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Web server port (default: 8000)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not automatically open the browser')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查结果目录是否存在
    if not os.path.exists(args.results_dir):
        print(f"Warning: Results directory '{args.results_dir}' does not exist.")
        print("Creating empty directory...")
        os.makedirs(args.results_dir)
    
    # 确保plots目录存在
    plots_dir = os.path.join(args.results_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 创建自定义的HTTP请求处理器
    handler = lambda *handler_args, **handler_kwargs: FederatedLearningHandler(
        *handler_args, results_dir=args.results_dir, **handler_kwargs
    )
    
    # 创建HTTP服务器
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        server_url = f"http://localhost:{args.port}"
        print(f"Web server started: {server_url}")
        
        # 自动打开浏览器
        if not args.no_browser:
            webbrowser.open(server_url)
        
        # 启动服务器
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main() 