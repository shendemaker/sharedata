# 联邦学习多数据集训练与可视化

本扩展为原始联邦学习模拟器添加了多数据集支持和可视化功能。现在，您可以轻松地在MNIST、Fashion MNIST和CIFAR10数据集上运行实验，并通过直观的图形界面比较它们的性能。

## 新增功能

1. **多数据集支持**：
   - 现在支持同时或单独对MNIST、Fashion MNIST和CIFAR10数据集进行训练
   - 每个数据集的实验结果保存在独立的目录中，便于比较和分析

2. **结果可视化**：
   - 创建了全面的可视化工具，用于分析和比较实验结果
   - 支持绘制准确率、损失、学习率变化等多种指标
   - 可以比较不同数据集之间的性能差异

3. **用户友好的运行脚本**：
   - 提供了简单的命令行界面，用于控制训练和可视化过程
   - 支持灵活选择要处理的数据集和可视化选项

## 使用方法

### 运行新的多数据集实验

使用`run_and_visualize.py`脚本可以轻松运行实验：

```bash
# 在所有数据集上运行实验并可视化结果
python run_and_visualize.py

# 仅运行训练过程，不进行可视化
python run_and_visualize.py --train

# 仅可视化现有的实验结果，不进行训练
python run_and_visualize.py --visualize

# 在选定的数据集上运行实验
python run_and_visualize.py --datasets mnist fashionmnist

# 保存可视化结果到指定目录
python run_and_visualize.py --visualize --save_plots --plot_path my_plots
```

### 可视化功能

`plot_results.py`提供了丰富的可视化功能：

```python
# 使用可视化器查看结果
from plot_results import ResultsVisualizer

# 创建可视化器实例
visualizer = ResultsVisualizer()

# 打印最终性能指标表格
visualizer.print_metrics_table()

# 绘制不同数据集的准确率对比
visualizer.plot_accuracy_comparison()

# 绘制不同数据集的损失对比
visualizer.plot_loss_comparison()

# 为特定数据集绘制详细指标
visualizer.plot_dataset_metrics('mnist')

# 为所有数据集绘制详细指标
visualizer.plot_all_datasets_detailed()
```

## 文件结构

新增的文件：

- `plot_results.py`：实现了结果可视化功能的类库
- `run_and_visualize.py`：用户友好的运行脚本
- `VISUALIZATION_README.md`：本文档，介绍新增功能的使用方法

修改的文件：

- `federated_learning.json`：更新为支持多数据集配置
- `data_utils.py`：优化了数据加载函数，兼容新版PyTorch

## 数据集说明

- **MNIST**：手写数字数据集，含有0-9的数字图像
- **Fashion MNIST**：服装物品数据集，设计为MNIST的直接替代品，但更具挑战性
- **CIFAR10**：常见物体的彩色图像数据集，包含10个类别

## 注意事项

1. 首次运行时，系统会自动下载所需的数据集
2. 确保您的环境变量`TRAINING_DATA`已正确设置
3. 默认配置适用于大多数情况，但您可以根据需要调整`federated_learning.json`中的参数
4. 生成的图表可以保存为高分辨率PNG文件，适合用于报告和演示 