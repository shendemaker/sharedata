@echo off
REM 切换到脚本所在目录
cd /d %~dp0

REM 运行可视化脚本
python visualize_results.py

REM 暂停，让用户查看结果
echo.
echo 按任意键退出...
pause > nul 