#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Web Visualization Server for Federated Learning Experiment Results
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
import shutil  # For file copying
from urllib.parse import parse_qs, urlparse

# Simplified HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Federated Learning Visualization</title>
    <style>
        body {
            font-family: Arial, 'Microsoft YaHei', 'SimHei', 'SimSun', 'Heiti SC', 'Noto Sans CJK SC', sans-serif;
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
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-container img {
            max-width: 100%;
            border: 1px solid #ddd;
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
    results_dir = 'results'  # Default results directory
    
    def do_GET(self):
        try:
            parsed_url = urlparse(self.path)
            
            print(f"Processing request: {self.path}")
            
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
                print(f"Unknown path: {self.path}")
                # For unknown paths, redirect to home page
                self.send_response(302)
                self.send_header('Location', '/')
                self.end_headers()
        except Exception as e:
            print(f"Error handling request: {str(e)}")
            traceback.print_exc()
            # Send a simple error page to avoid empty response
            self.send_response(500)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Server Error</title>
                <meta charset="UTF-8">
            </head>
            <body>
                <h1>500 - Server Error</h1>
                <p>Error processing request: {str(e)}</p>
                <p><a href="/">Return to Home</a></p>
            </body>
            </html>
            """
            self.wfile.write(error_html.encode('utf-8'))
    
    def serve_main_page(self):
        try:
            # Get available datasets
            datasets = self._get_available_datasets()
            
            if not datasets:
                # If no datasets, create some example dataset directories
                example_datasets = ['mnist', 'fashionmnist', 'cifar10']
                for ds in example_datasets:
                    ds_path = os.path.join(self.results_dir, ds)
                    if not os.path.exists(ds_path):
                        os.makedirs(ds_path)
                datasets = example_datasets
            
            # Create navigation links
            nav_links = '<a href="/">Home</a>'
            for ds in datasets:
                nav_links += f'<a href="/dataset?dataset={ds}">{ds.upper()}</a>'
            
            # Create content
            content = """
            <h2>Welcome to Federated Learning Visualization Tool</h2>
            <p>Please select a dataset from the navigation bar above for details</p>
            
            <h3>Available Datasets:</h3>
            <ul>
            """
            
            for ds in datasets:
                content += f"<li><a href='/dataset?dataset={ds}'>{ds.upper()}</a></li>"
            
            content += "</ul>"
            
            # Replace placeholders in the template
            html = HTML_TEMPLATE
            html = html.replace("DATASET_LINKS_PLACEHOLDER", nav_links)
            html = html.replace("CONTENT_PLACEHOLDER", content)
            
            # Send response
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
            # Get available datasets
            datasets = self._get_available_datasets()
            
            # Create navigation links
            nav_links = '<a href="/">Home</a>'
            for ds in datasets:
                nav_links += f'<a href="/dataset?dataset={ds}">{ds.upper()}</a>'
            
            # Create content
            content = f"""
            <h2>{dataset.upper()} Dataset Results</h2>
            
            <div>
                <h3>Differential Privacy Comparison</h3>
                <p><a href="/dp_comparison?dataset={dataset}">View Differential Privacy Comparison</a></p>
            </div>
            
            <div>
                <h3>Experiment Results</h3>
                <p>Loading experiment data...</p>
            </div>
            """
            
            # Replace placeholders in the template
            html = HTML_TEMPLATE
            html = html.replace("DATASET_LINKS_PLACEHOLDER", nav_links)
            html = html.replace("CONTENT_PLACEHOLDER", content)
            
            # Send response
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
            # Get available datasets
            datasets = self._get_available_datasets()
            
            # Create navigation links
            nav_links = '<a href="/">Home</a>'
            for ds in datasets:
                nav_links += f'<a href="/dataset?dataset={ds}">{ds.upper()}</a>'
            
            # Check if data exists
            comparison_dir = os.path.join(self.results_dir, dataset, 'comparison')
            data_file = os.path.join(comparison_dir, 'dp_comparison_data.npz')
            
            print(f"Checking differential privacy comparison data: {data_file}")
            
            if not os.path.exists(data_file):
                # If data doesn't exist, show error page
                print(f"Data file does not exist: {data_file}")
                content = f"""
                <h2>Differential Privacy Comparison - {dataset.upper()}</h2>
                
                <div style="color: red; padding: 15px; background-color: #fadbd8; border-radius: 5px; margin: 20px 0;">
                    <h3>Data Not Found</h3>
                    <p>Differential privacy comparison data for {dataset} not found.</p>
                    <p>Please run the following command to generate comparison data:</p>
                    <pre>python compare_dp.py</pre>
                    <p>Or:</p>
                    <pre>python fix_dp_compare.py</pre>
                </div>
                """
            else:
                # If data exists, show comparison page
                print(f"Data file exists: {data_file}")
                
                # Check image files
                image_files = [
                    "privacy_accuracy_tradeoff.png",
                    "accuracy_comparison.png",
                    "loss_comparison.png"
                ]
                
                for img_file in image_files:
                    full_path = os.path.join(comparison_dir, img_file)
                    if os.path.exists(full_path):
                        print(f"Image file exists: {full_path}")
                    else:
                        print(f"Image file does not exist: {full_path}")
                
                # Use direct image paths
                content = f"""
                <h2>Differential Privacy Comparison - {dataset.upper()}</h2>
                
                <div style="margin-bottom: 40px;">
                    <h3>Privacy-Accuracy Trade-off</h3>
                    <p>
                        This analysis shows how differential privacy affects model accuracy.
                        Higher privacy protection (lower ε values) typically results in lower accuracy due to added noise.
                    </p>
                    
                    <div style="margin: 20px 0; text-align: center;">
                        <img style="max-width: 100%; border: 1px solid #ddd;" 
                             src="/direct-image?path={dataset}/comparison/privacy_accuracy_tradeoff.png" 
                             alt="Privacy-Accuracy Trade-off">
                    </div>
                </div>
                
                <div style="margin-bottom: 40px;">
                    <h3>Accuracy Comparison</h3>
                    <div style="margin: 20px 0; text-align: center;">
                        <img style="max-width: 100%; border: 1px solid #ddd;" 
                             src="/direct-image?path={dataset}/comparison/accuracy_comparison.png" 
                             alt="Accuracy Comparison">
                    </div>
                </div>
                
                <div style="margin-bottom: 40px;">
                    <h3>Loss Comparison</h3>
                    <div style="margin: 20px 0; text-align: center;">
                        <img style="max-width: 100%; border: 1px solid #ddd;" 
                             src="/direct-image?path={dataset}/comparison/loss_comparison.png" 
                             alt="Loss Comparison">
                    </div>
                </div>
                """
            
            # Replace placeholders in the template
            html = HTML_TEMPLATE
            html = html.replace("DATASET_LINKS_PLACEHOLDER", nav_links)
            html = html.replace("CONTENT_PLACEHOLDER", content)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            
        except Exception as e:
            print(f"Error serving DP comparison page: {str(e)}")
            traceback.print_exc()
            self.send_error(500, "Internal Server Error")
    
    def serve_direct_image(self, img_path):
        """Serve an image file directly from the specified path"""
        try:
            # Sanitize the image path for security
            img_path = img_path.replace('..', '').strip('/')
            img_path = os.path.join(self.results_dir, img_path)
            
            if not os.path.isfile(img_path):
                print(f"Image not found: {img_path}")
                self.send_error(404, "Image not found")
                return
            
            # Determine content type based on file extension
            _, ext = os.path.splitext(img_path)
            content_type = 'image/png' if ext.lower() == '.png' else 'image/jpeg' if ext.lower() in ['.jpg', '.jpeg'] else 'application/octet-stream'
            
            # Send response headers
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Cache-Control', 'max-age=3600')  # Cache for an hour to improve performance
            self.end_headers()
            
            # Send image data
            with open(img_path, 'rb') as f:
                self.wfile.write(f.read())
            
            print(f"Served image: {img_path}")
            
        except Exception as e:
            print(f"Error serving image {img_path}: {str(e)}")
            traceback.print_exc()
            self.send_error(500, f"Error serving image: {str(e)}")
    
    def serve_static_file(self):
        try:
            # Parse file path
            file_path = self.path[8:]  # Remove '/static/' prefix
            full_path = os.path.join(self.results_dir, file_path)
            
            print(f"Static file request: {file_path}")
            print(f"Full path: {full_path}")
            
            # Check if file exists
            if not os.path.exists(full_path):
                print(f"File does not exist: {full_path}")
                # Try checking alternate locations
                alt_paths = [
                    os.path.join(self.results_dir, 'plots', file_path),
                    os.path.join(os.path.dirname(__file__), 'static', file_path)
                ]
                
                for alt_path in alt_paths:
                    print(f"Trying alternate path: {alt_path}")
                    if os.path.exists(alt_path):
                        full_path = alt_path
                        print(f"File found: {full_path}")
                        break
                else:
                    self.send_error(404, f"File not found: {file_path}")
                    return
            
            # Determine content type
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
            
            # Send file
            print(f"Sending file: {full_path}, type: {content_type}")
            with open(full_path, 'rb') as f:
                file_data = f.read()
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.send_header('Content-length', len(file_data))
                self.end_headers()
                self.wfile.write(file_data)
                print(f"File sent successfully, size: {len(file_data)} bytes")
                
        except Exception as e:
            print(f"Error serving static file: {str(e)}")
            traceback.print_exc()
            self.send_error(500, "Internal Server Error")
    
    def _get_available_datasets(self):
        """Get available datasets"""
        datasets = []
        for dir_path in glob.glob(os.path.join(self.results_dir, '*')):
            if os.path.isdir(dir_path):
                datasets.append(os.path.basename(dir_path))
        return datasets

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple Federated Learning Visualization')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Root directory for result files (default: results)')
    parser.add_argument('--port', type=int, default=8001,
                        help='Web server port (default: 8001)')
    
    args = parser.parse_args()
    
    print(f"Using results directory: {args.results_dir}")
    print(f"Using port: {args.port}")
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Warning: Results directory '{args.results_dir}' does not exist, creating empty directory.")
        os.makedirs(args.results_dir)
    
    # Set results directory as class variable
    SimpleVisualizerHandler.results_dir = args.results_dir
    
    # Start server
    with socketserver.TCPServer(("", args.port), SimpleVisualizerHandler) as httpd:
        print(f"Server started: http://localhost:{args.port}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

if __name__ == "__main__":
    main() 