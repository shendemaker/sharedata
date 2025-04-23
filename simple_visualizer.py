#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于在浏览器中展示联邦学习实验结果的简化Web可视化服务器
"""

import os
import sys
import argparse
import http.server
import socketserver
import webbrowser
import traceback
import glob
import numpy as np
import shutil  # 用于复制文件
from urllib.parse import parse_qs, urlparse

# 简化版HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Federated Learning Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .nav {
            display: flex;
            background-color: #3498db;
            padding: 10px;
            margin-bottom: 20px;
        }
        .nav a {
            color: white;
            text-decoration: none;
            padding: 8px 15px;
            margin-right: 5px;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Federated Learning Visualization</h1>
        <div class="nav">
            DATASET_LINKS_PLACEHOLDER
        </div>
        CONTENT_PLACEHOLDER
    </div>
</body>
</html>
"""

class SimpleVisualizerHandler(http.server.SimpleHTTPRequestHandler):
    results_dir = 'results'  # 默认结果目录
    
    def do_GET(self):
        try:
            parsed_url = urlparse(self.path)
            
            print(f"处理请求: {self.path}")
            
            if self.path == '/' or self.path == '/index.html':
                self.serve_main_page()
            elif parsed_url.path.startswith('/dataset'):
                query = parse_qs(parsed_url.query)
                dataset = query.get('dataset', ['mnist'])[0]
                self.serve_dataset_page(dataset)
            elif parsed_url.path.startswith('/dp_comparison'):
                query = parse_qs(parsed_url.query)
                dataset = query.get('dataset', ['mnist'])[0]
                self.serve_dp_comparison_page(dataset)
            elif parsed_url.path.startswith('/direct-image'):
                query = parse_qs(parsed_url.query)
                img_path = query.get('path', [''])[0]
                self.serve_direct_image(img_path)
            elif self.path.startswith('/static/'):
                self.serve_static_file()
            else:
                print(f"未知路径: {self.path}")
                # 对于未知路径，重定向到主页
                self.send_response(302)
                self.send_header('Location', '/')
                self.end_headers()
        except Exception as e:
            print(f"Error handling request: {str(e)}")
            traceback.print_exc()
            # 发送简单的错误页面，避免空响应
            self.send_response(500)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>服务器错误</title>
                <meta charset="UTF-8">
            </head>
            <body>
                <h1>500 - 服务器错误</h1>
                <p>处理请求时发生错误: {str(e)}</p>
                <p><a href="/">返回主页</a></p>
            </body>
            </html>
            """
            self.wfile.write(error_html.encode('utf-8'))
    
    def serve_main_page(self):
        try:
            # 获取可用数据集
            datasets = self._get_available_datasets()
            
            if not datasets:
                # 如果没有数据集，创建一些示例数据集目录
                example_datasets = ['mnist', 'fashionmnist', 'cifar10']
                for ds in example_datasets:
                    ds_path = os.path.join(self.results_dir, ds)
                    if not os.path.exists(ds_path):
                        os.makedirs(ds_path)
                datasets = example_datasets
            
            # 创建导航链接
            nav_links = '<a href="/">主页</a>'
            for ds in datasets:
                nav_links += f'<a href="/dataset?dataset={ds}">{ds.upper()}</a>'
            
            # 创建内容
            content = """
            <h2>欢迎使用联邦学习可视化工具</h2>
            <p>请从上方导航栏选择数据集查看详情</p>
            
            <h3>可用数据集:</h3>
            <ul>
            """
            
            for ds in datasets:
                content += f"<li><a href='/dataset?dataset={ds}'>{ds.upper()}</a></li>"
            
            content += "</ul>"
            
            # 替换模板中的占位符
            html = HTML_TEMPLATE
            html = html.replace("DATASET_LINKS_PLACEHOLDER", nav_links)
            html = html.replace("CONTENT_PLACEHOLDER", content)
            
            # 发送响应
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            
        except Exception as e:
            print(f"Error serving main page: {str(e)}")
            traceback.print_exc()
            self.send_error(500, "Internal Server Error")
    
    def serve_dataset_page(self, dataset):
        try:
            # 获取可用数据集
            datasets = self._get_available_datasets()
            
            # 创建导航链接
            nav_links = '<a href="/">主页</a>'
            for ds in datasets:
                nav_links += f'<a href="/dataset?dataset={ds}">{ds.upper()}</a>'
            
            # 创建内容
            content = f"""
            <h2>{dataset.upper()} 数据集结果</h2>
            
            <div>
                <h3>差分隐私对比</h3>
                <p><a href="/dp_comparison?dataset={dataset}">查看差分隐私对比</a></p>
            </div>
            
            <div>
                <h3>实验结果</h3>
                <p>实验数据加载中...</p>
            </div>
            """
            
            # 替换模板中的占位符
            html = HTML_TEMPLATE
            html = html.replace("DATASET_LINKS_PLACEHOLDER", nav_links)
            html = html.replace("CONTENT_PLACEHOLDER", content)
            
            # 发送响应
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            
        except Exception as e:
            print(f"Error serving dataset page: {str(e)}")
            traceback.print_exc()
            self.send_error(500, "Internal Server Error")
    
    def serve_dp_comparison_page(self, dataset):
        try:
            # 获取可用数据集
            datasets = self._get_available_datasets()
            
            # 创建导航链接
            nav_links = '<a href="/">主页</a>'
            for ds in datasets:
                nav_links += f'<a href="/dataset?dataset={ds}">{ds.upper()}</a>'
            
            # 检查数据是否存在
            comparison_dir = os.path.join(self.results_dir, dataset, 'comparison')
            data_file = os.path.join(comparison_dir, 'dp_comparison_data.npz')
            
            print(f"检查差分隐私对比数据: {data_file}")
            
            if not os.path.exists(data_file):
                # 数据不存在时显示错误页面
                print(f"数据文件不存在: {data_file}")
                content = f"""
                <h2>差分隐私对比 - {dataset.upper()}</h2>
                
                <div style="color: red; padding: 15px; background-color: #fadbd8; border-radius: 5px; margin: 20px 0;">
                    <h3>数据未找到</h3>
                    <p>未找到{dataset}的差分隐私对比数据。</p>
                    <p>请运行以下命令生成对比数据：</p>
                    <pre>python compare_dp.py</pre>
                    <p>或者：</p>
                    <pre>python fix_dp_compare.py</pre>
                </div>
                """
            else:
                # 数据存在时显示对比页面
                print(f"数据文件存在: {data_file}")
                
                # 检查图片文件
                image_files = [
                    "privacy_accuracy_tradeoff.png",
                    "accuracy_comparison.png",
                    "loss_comparison.png"
                ]
                
                for img_file in image_files:
                    full_path = os.path.join(comparison_dir, img_file)
                    if os.path.exists(full_path):
                        print(f"图片文件存在: {full_path}")
                    else:
                        print(f"图片文件不存在: {full_path}")
                
                # 使用直接图片路径
                content = f"""
                <h2>差分隐私对比 - {dataset.upper()}</h2>
                
                <div style="margin-bottom: 40px;">
                    <h3>隐私-准确率权衡</h3>
                    <p>
                        该分析展示了差分隐私如何影响模型准确率。
                        更高的隐私保护（较低的ε值）通常会因为添加的噪声而导致较低的准确率。
                    </p>
                    
                    <div style="margin: 20px 0; text-align: center;">
                        <img style="max-width: 100%; border: 1px solid #ddd;" 
                             src="/direct-image?path={dataset}/comparison/privacy_accuracy_tradeoff.png" 
                             alt="隐私-准确率权衡">
                    </div>
                </div>
                
                <div style="margin-bottom: 40px;">
                    <h3>准确率对比</h3>
                    <div style="margin: 20px 0; text-align: center;">
                        <img style="max-width: 100%; border: 1px solid #ddd;" 
                             src="/direct-image?path={dataset}/comparison/accuracy_comparison.png" 
                             alt="准确率对比">
                    </div>
                </div>
                
                <div style="margin-bottom: 40px;">
                    <h3>损失对比</h3>
                    <div style="margin: 20px 0; text-align: center;">
                        <img style="max-width: 100%; border: 1px solid #ddd;" 
                             src="/direct-image?path={dataset}/comparison/loss_comparison.png" 
                             alt="损失对比">
                    </div>
                </div>
                """
            
            # 替换模板中的占位符
            html = HTML_TEMPLATE
            html = html.replace("DATASET_LINKS_PLACEHOLDER", nav_links)
            html = html.replace("CONTENT_PLACEHOLDER", content)
            
            # 发送响应
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            
        except Exception as e:
            print(f"Error serving DP comparison page: {str(e)}")
            traceback.print_exc()
            self.send_error(500, "Internal Server Error")
    
    def serve_direct_image(self, img_path):
        """直接提供图片文件"""
        try:
            # 构建文件的完整路径
            full_path = os.path.join(self.results_dir, img_path)
            
            print(f"请求图片: {img_path}")
            print(f"完整路径: {full_path}")
            
            # 检查文件是否存在
            if not os.path.exists(full_path):
                print(f"图片不存在: {full_path}")
                self.send_error(404, f"图片未找到: {img_path}")
                return
            
            # 确定内容类型
            content_type = 'image/png' if full_path.endswith('.png') else 'image/jpeg'
            
            # 发送图片
            print(f"发送图片: {full_path}")
            with open(full_path, 'rb') as f:
                img_data = f.read()
                print(f"图片大小: {len(img_data)} 字节")
                
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.send_header('Content-length', len(img_data))
                self.end_headers()
                self.wfile.write(img_data)
                print(f"图片发送成功")
                
        except Exception as e:
            print(f"发送图片出错: {str(e)}")
            traceback.print_exc()
            self.send_error(500, "Internal Server Error")
    
    def serve_static_file(self):
        try:
            # 解析文件路径
            file_path = self.path[8:]  # 移除'/static/'前缀
            full_path = os.path.join(self.results_dir, file_path)
            
            print(f"请求静态文件: {file_path}")
            print(f"完整路径: {full_path}")
            
            # 检查文件是否存在
            if not os.path.exists(full_path):
                print(f"文件不存在: {full_path}")
                # 尝试检查备用位置
                alt_paths = [
                    os.path.join(self.results_dir, 'plots', file_path),
                    os.path.join(os.path.dirname(__file__), 'static', file_path)
                ]
                
                for alt_path in alt_paths:
                    print(f"尝试备用路径: {alt_path}")
                    if os.path.exists(alt_path):
                        full_path = alt_path
                        print(f"找到文件: {full_path}")
                        break
                else:
                    self.send_error(404, f"文件未找到: {file_path}")
                    return
            
            # 确定内容类型
            content_type = 'text/plain'
            if full_path.endswith('.png'):
                content_type = 'image/png'
            elif full_path.endswith('.jpg') or full_path.endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif full_path.endswith('.html'):
                content_type = 'text/html'
            elif full_path.endswith('.css'):
                content_type = 'text/css'
            elif full_path.endswith('.js'):
                content_type = 'application/javascript'
            
            # 发送文件
            print(f"发送文件: {full_path}, 类型: {content_type}")
            with open(full_path, 'rb') as f:
                file_data = f.read()
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.send_header('Content-length', len(file_data))
                self.end_headers()
                self.wfile.write(file_data)
                print(f"文件发送成功, 大小: {len(file_data)} 字节")
                
        except Exception as e:
            print(f"Error serving static file: {str(e)}")
            traceback.print_exc()
            self.send_error(500, "Internal Server Error")
    
    def _get_available_datasets(self):
        """获取可用的数据集"""
        datasets = []
        for dir_path in glob.glob(os.path.join(self.results_dir, '*')):
            if os.path.isdir(dir_path):
                datasets.append(os.path.basename(dir_path))
        return datasets

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Simple Federated Learning Visualization')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Root directory for result files (default: results)')
    parser.add_argument('--port', type=int, default=8001,
                        help='Web server port (default: 8001)')
    
    args = parser.parse_args()
    
    print(f"使用结果目录: {args.results_dir}")
    print(f"使用端口: {args.port}")
    
    # 检查结果目录是否存在
    if not os.path.exists(args.results_dir):
        print(f"警告: 结果目录 '{args.results_dir}' 不存在，将创建空目录.")
        os.makedirs(args.results_dir)
    
    # 设置结果目录为类变量
    SimpleVisualizerHandler.results_dir = args.results_dir
    
    # 启动服务器
    with socketserver.TCPServer(("", args.port), SimpleVisualizerHandler) as httpd:
        print(f"服务器已启动: http://localhost:{args.port}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")

if __name__ == "__main__":
    main() 