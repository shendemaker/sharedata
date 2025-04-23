@echo off
REM 切换到脚本所在目录
cd /d %~dp0

REM 运行Web可视化服务器
python web_visualizer.py

REM 如果需要指定不同的结果目录和端口，可以使用下面的命令：
REM python web_visualizer.py --results_dir results --port 8080

REM 按Ctrl+C可以退出服务器 