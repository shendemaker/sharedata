

## 1. 项目简介

### 1.1 背景介绍

联邦学习是一种分布式机器学习方法，它允许多个客户端(如移动设备、组织等)在不共享原始数据的情况下协作训练机器学习模型。这种方法解决了传统集中式学习中的数据隐私、安全和监管问题。

本项目实现了一个非交互式联邦学习系统，支持多种数据集(MNIST、Fashion-MNIST和CIFAR-10)和模型架构，可以模拟不同数据分布场景下的联邦学习过程。

### 1.2 功能特点

- **多数据集支持**: MNIST、Fashion-MNIST和CIFAR-10
- **非IID数据分布**: 支持客户端数据非独立同分布的场景
- **可配置的客户端数量**: 可灵活设置参与训练的客户端数量
- **可视化工具**: 提供训练过程和结果的可视化界面
- **快速启动功能**: 提供简化配置的快速启动选项

## 2. 系统架构

### 2.1 整体架构

系统由以下主要组件构成:

1. **数据加载器**: 负责加载和分发数据集到各个客户端
2. **客户端模拟器**: 模拟多个客户端的本地训练行为
3. **服务器聚合器**: 汇总客户端更新，更新全局模型
4. **配置管理器**: 管理系统的各项参数设置
5. **可视化服务器**: 提供训练过程和结果的图形化展示

### 2.2 工作流程

1. 系统根据配置文件初始化参数
2. 数据加载器将数据分配给模拟的客户端
3. 在每个通信轮次:
   - 服务器选择一部分客户端参与训练
   - 选中的客户端使用本地数据进行模型训练
   - 客户端将模型更新发送给服务器
   - 服务器聚合更新并更新全局模型
4. 训练结束后，系统记录结果并可通过可视化工具查看

## 3. 数据集介绍

### 3.1 MNIST

手写数字识别数据集，包含0-9十个类别的手写数字图像:
- 60,000张训练图像
- 10,000张测试图像
- 图像尺寸: 28×28 像素
- 灰度图像

### 3.2 Fashion-MNIST

服装图像分类数据集，结构与MNIST相同:
- 60,000张训练图像
- 10,000张测试图像
- 图像尺寸: 28×28 像素
- 10个类别: T恤、裤子、套头衫等
- 灰度图像

### 3.3 CIFAR-10

彩色图像分类数据集:
- 50,000张训练图像
- 10,000张测试图像
- 图像尺寸: 32×32×3 像素
- 10个类别: 飞机、汽车、鸟类等
- 彩色图像(RGB)

## 4. 安装与配置

### 4.1 系统要求

- Python 3.6+
- PyTorch 1.8+
- NumPy
- Matplotlib
- TensorBoard (可选，用于高级可视化)

### 4.2 安装步骤

```bash
# 克隆项目仓库
git clone https://github.com/username/nonIID-non-interactive-federated-learning.git

# 进入项目目录
cd nonIID-non-interactive-federated-learning

# 安装依赖
pip install -r requirements.txt
```

### 4.3 配置文件说明

系统使用JSON格式的配置文件进行设置。主要配置文件为`federated_learning.json`，快速启动配置文件为`federated_learning_quick.json`。

配置字段说明:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| dataset | 数据集名称(mnist/fashionmnist/cifar10) | mnist |
| n_clients | 客户端数量 | 100 |
| batch_size | 批量大小 | 10 |
| rounds | 通信轮数 | 200 |
| participation_rate | 客户端参与率 | 0.1 |
| lr | 学习率 | 0.01 |
| classes_per_client | 每个客户端拥有的类别数 | 2 |
| balancedness | 数据平衡度(0-1，1表示完全平衡) | 1.0 |

## 5. 快速入门

### 5.1 使用快速启动脚本

```bash
# 使用快速启动配置运行系统
python quick_start.py
```

快速启动使用简化配置(`federated_learning_quick.json`)，主要包括:
- 数据集限制为MNIST
- 客户端数量减少到50
- 通信轮数减少到100

### 5.2 完整系统启动

```bash
# 使用完整配置运行系统
python main.py
```

此命令使用`federated_learning.json`中的完整配置运行系统，包括多数据集支持和更复杂的参数设置。

## 6. 数据可视化

### 6.1 启动可视化服务器

```bash
# 启动web可视化服务器
python web_visualizer.py
```

默认情况下，服务器在本地8000端口启动。访问`http://localhost:8000`查看可视化界面。

### 6.2 可视化功能说明

可视化界面提供以下功能:

1. **实验结果概览**: 显示不同数据集的实验结果摘要
2. **训练过程图表**: 
   - 测试准确率曲线
   - 测试损失曲线
   - 训练准确率曲线
   - 训练损失曲线
3. **比较视图**:
   - 训练与测试精度对比
   - 训练与测试损失对比
4. **客户端分析**: 查看各个客户端的训练状况

### 6.3 修复可视化问题

如果遇到图表不显示问题，可以运行修复脚本:

```bash
python fix_plot.py
```

此脚本会为所有数据集生成示例图表，解决因字体或其他显示问题导致的图表不可见问题。

## 7. 高级功能

### 7.1 数据分布配置

通过修改配置文件中的`classes_per_client`和`balancedness`参数，可以模拟不同的非IID数据分布场景:

```json
{
  "classes_per_client": 2,  // 每个客户端拥有2个类别
  "balancedness": 0.8       // 80%的数据平衡度
}
```

### 7.2 自定义模型

系统支持自定义模型架构，可以通过修改模型定义文件来使用不同的神经网络结构:

```python
# 示例: 自定义CNN模型
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # 定义模型结构
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 更多层...
    
    def forward(self, x):
        # 前向传播逻辑
        x = self.conv1(x)
        # 更多操作...
        return x
```

### 7.3 实验数据管理

系统会自动保存实验数据到`results`目录，包括:

- 训练和测试精度/损失
- 模型参数
- 配置信息

保存的数据可用于后续分析或可视化。

## 8. 常见问题解答

### 8.1 中文字体显示问题

**问题**: 在可视化图表中中文字符显示为方框或乱码。

**解决方案**: 系统现已更新为使用英文标签和指示，解决了字体显示问题。如果仍有问题，请运行`fix_plot.py`脚本修复。

### 8.2 内存不足错误

**问题**: 使用大量客户端时出现内存不足错误。

**解决方案**: 减少客户端数量或使用较小的批量大小，也可以使用`federated_learning_quick.json`进行快速测试。

### 8.3 训练速度过慢

**问题**: 在大型数据集(如CIFAR-10)上训练速度很慢。

**解决方案**: 
- 减少通信轮数
- 降低客户端数量
- 考虑使用GPU加速(如果可用)
- 使用快速配置启动系统

## 9. 开发者指南

### 9.1 目录结构

```
federated-learning-master/
├── data/                   # 数据集存储目录
├── models/                 # 模型定义
├── results/                # 实验结果
│   ├── mnist/              # MNIST实验结果
│   ├── fashionmnist/       # Fashion-MNIST实验结果
│   └── cifar10/            # CIFAR-10实验结果
├── federated_learning.json # 主配置文件
├── federated_learning_quick.json # 快速启动配置
├── main.py                 # 主程序入口
├── quick_start.py          # 快速启动脚本
├── web_visualizer.py       # 可视化服务器
└── fix_plot.py             # 图表修复工具
```

### 9.2 扩展指南

#### 添加新数据集

1. 在`data_loader.py`中添加新数据集加载函数
2. 在配置文件中添加新数据集选项
3. 为新数据集创建合适的模型结构

#### 实现新的聚合算法

1. 在聚合器模块中添加新算法实现
2. 在配置中添加新算法选项
3. 更新服务器代码使用新算法

## 10. 附录

### 10.1 参考文献

1. McMahan, H. B., Moore, E., Ramage, D., & Hampson, S. (2016). Communication-efficient learning of deep networks from decentralized data. arXiv preprint arXiv:1602.05629.
2. Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). Federated learning: Challenges, methods, and future directions. IEEE Signal Processing Magazine, 37(3), 50-60.

### 10.2 术语表

- **联邦学习(Federated Learning)**: 一种分布式机器学习方法，使多个客户端可以在不分享原始数据的情况下协作训练模型。
- **非IID数据(Non-IID Data)**: 非独立同分布数据，指客户端间数据分布不一致的情况。
- **通信轮(Communication Round)**: 联邦学习中客户端与服务器之间进行一次完整模型更新的周期。
- **聚合(Aggregation)**: 服务器将多个客户端模型更新合并为全局模型的过程。
- **客户端参与率(Participation Rate)**: 每轮训练中参与的客户端比例。

### 10.3 配置实例

完整配置文件示例:

```json
{
  "dataset": "mnist",
  "n_clients": 100,
  "batch_size": 10,
  "test_batch_size": 100,
  "rounds": 200,
  "lr": 0.01,
  "log_interval": 10,
  "classes_per_client": 2,
  "balancedness": 1.0,
  "participation_rate": 0.1
}
```

快速启动配置文件示例:

```json
{
  "dataset": "mnist",
  "n_clients": 50,
  "batch_size": 10,
  "test_batch_size": 100,
  "rounds": 100,
  "lr": 0.01,
  "log_interval": 10,
  "classes_per_client": 2,
  "balancedness": 1.0,
  "participation_rate": 0.1
}
```
