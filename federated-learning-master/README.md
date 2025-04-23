# Federated Learning Simulator 联邦学习模拟器

Simulate Federated Learning with compressed communication on a large number of Clients.
利用压缩通信在大量客户端上模拟联邦学习。

Recreate experiments described in [*Sattler, F., Wiedemann, S., Müller, K. R., & Samek, W. (2019). Robust and Communication-Efficient Federated Learning from Non-IID Data. arXiv preprint arXiv:1903.02891.*](https://arxiv.org/abs/1903.02891)
中描述的再现实验


## Usage  用法
First, set environment variable 'TRAINING_DATA' to point to the directory where you want your training data to be stored. MNIST, FASHION-MNIST and CIFAR10 will download automatically. 
首先，设置环境变量'TRAINING_DATA'来指向您想要存储培训数据的目录。MNIST、FASHION-MNIST和CIFAR10将自动下载。
`python federated_learning.py`

will run the Federated Learning experiment specified in  
将运行指定的联邦学习实验
`federated_learning.json`.

You can specify:
您可以指定：

### Task任务
- `"dataset"` : Choose from `["mnist", "cifar10", "kws", "fashionmnist"]`
- `"net"` : Choose from `["logistic", "lstm", "cnn", "vgg11", "vgg11s"]`

### Federated Learning Environment 联邦学习环境

- `"n_clients"` : Number of Clients
- `"classes_per_client"` : Number of different Classes every Client holds in it's local data
- `"participation_rate"` : Fraction of Clients which participate in every Communication Round
- `"batch_size"` : Batch-size used by the Clients
- `"balancedness"` : Default 1.0, if <1.0 data will be more concentrated on some clients  ????? non-iid
- `"iterations"` : Total number of training iterations
- `"momentum"` : Momentum used during training on the clients

### Compression Method ????? 压缩方法

- `"compression"` : Choose from `[["none", {}], ["fedavg", {"n" : ?}], ["signsgd", {"lr" : ?}], ["stc_updown", [{"p_up" : ?, "p_down" : ?}]], ["stc_up", {"p_up" : ?}], ["dgc_updown", [{"p_up" : ?, "p_down" : ?}]], ["dgc_up", {"p_up" : ?}] ]`

### Logging  日志记录
- `"log_frequency"` : Number of communication rounds after which results are logged and saved to disk
- `"log_path"` : e.g. "results/experiment1/"

Run multiple experiments by listing different configurations.

## Options 选项
- `--schedule` : specify which batch of experiments to run, defaults to "main"
指定运行哪一批实验，默认为"main"
## Citation 
[Paper](https://arxiv.org/abs/1903.02891)

Sattler, F., Wiedemann, S., Müller, K. R., & Samek, W. (2019). Robust and Communication-Efficient Federated Learning from Non-IID Data. arXiv preprint arXiv:1903.02891.

## Web可视化界面

我们新增了一个Web可视化功能，可以在浏览器中查看实验结果。使用方法如下：

### Windows用户

直接双击`start_web_visualizer.bat`文件即可启动Web服务器并自动打开浏览器。

### 命令行用户

```
cd federated-learning-master
python web_visualizer.py
```

### 可选参数

Web服务器支持以下参数：

- `--results_dir`: 指定结果文件的根目录（默认：results）
- `--port`: 指定Web服务器端口（默认：8000）
- `--no-browser`: 启动服务器时不自动打开浏览器

例如：
```
python web_visualizer.py --results_dir custom_results --port 8080 --no-browser
```

### 功能特点

- 自动检测并显示所有可用的数据集
- 展示每个数据集的训练指标和性能图表
- 自适应布局，支持在各种设备上查看
- 无需额外安装依赖，使用Python内置的HTTP服务器
