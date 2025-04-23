# 联邦学习项目快速启动指南

此指南帮助您快速启动和测试联邦学习项目，通过减少数据量和计算资源需求，使项目可以在短时间内完成训练和测试。

## 为什么需要快速启动？

原始项目配置使用2000个客户端和大量训练迭代，这在测试阶段会消耗大量时间和计算资源。快速启动模式通过以下方式优化：

- 大幅减少客户端数量（默认50个）
- 减少训练迭代次数（默认100次）
- 只使用单个数据集进行训练
- 简化日志记录频率

这些优化使您能够在几分钟内验证项目的基本功能，而不必等待完整训练过程完成。

## 使用方法

### 基本用法

最简单的启动方式是直接运行快速启动脚本：

```bash
python quick_start.py
```

这将使用默认配置（MNIST数据集、50个客户端、100次迭代）进行快速训练。

### 自定义参数

您可以通过命令行参数自定义快速启动配置：

```bash
# 使用Fashion MNIST数据集，30个客户端，50次迭代
python quick_start.py --dataset fashionmnist --clients 30 --iterations 50

# 训练完成后立即可视化结果
python quick_start.py --visualize

# 使用CIFAR10数据集，增加客户端数量
python quick_start.py --dataset cifar10 --clients 100
```

### 参数说明

- `--dataset`：选择要使用的数据集（mnist、fashionmnist或cifar10）
- `--clients`：设置客户端数量
- `--iterations`：设置训练迭代次数
- `--visualize`：训练完成后是否显示可视化结果

## 注意事项

1. 快速模式主要用于功能验证和测试，不应用于最终性能评估
2. 虽然客户端数量减少，但模型的基本训练逻辑保持不变
3. 如果需要进行完整实验，请使用原始配置文件和 `python federated_learning.py` 命令
4. 快速模式的实验结果保存在 `results/quick_test/` 目录下

## 从快速测试到完整实验

完成快速测试后，如果您希望运行完整实验，可以：

```bash
# 使用原始配置运行完整实验
python federated_learning.py

# 使用可视化工具运行和可视化完整实验
python run_and_visualize.py
```

快速启动是迭代开发和测试的理想方式，一旦您确认基本功能正常，可以逐步增加数据量进行更全面的性能评估。 