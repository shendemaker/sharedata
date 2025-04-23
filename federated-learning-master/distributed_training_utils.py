import torch, torchvision
import numpy as np
import itertools as it
import re
from math import sqrt
import random

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time

import compression_utils as comp
# Import differential privacy module
try:
    import differential_privacy as dp
except ImportError:
    # Create a path relative to the current file
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import differential_privacy as dp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def copy(target, source):
  for name in target:
    target[name].data = source[name].data.clone()

def add(target, source):
  for name in target:
    target[name].data += source[name].data.clone()

def scale(target, scaling):
  for name in target:
    target[name].data = scaling*target[name].data.clone()

def subtract(target, source):
  for name in target:
    target[name].data -= source[name].data.clone()

def subtract_(target, minuend, subtrahend):
  for name in target:
    target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()

def average(target, sources):
  for name in target:
    target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()

def weighted_average(target, sources, weights):
  for name in target:
    summ = torch.sum(weights)
    n = len(sources)
    modify = [weight/summ*n for weight in weights]
    target[name].data = torch.mean(torch.stack([m*source[name].data for source, m in zip(sources, modify)]), dim=0).clone()

def majority_vote(target, sources, lr):
  for name in target:
    threshs = torch.stack([torch.max(source[name].data) for source in sources])
    mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
    target[name].data = (lr*mask).clone()

def compress(target, source, compress_fun):
  '''
  compress_fun : a function f : tensor (shape) -> tensor (shape)一个函数f:张量(shape) ->张量(shape)
  '''
  for name in target:
    target[name].data = compress_fun(source[name].data.clone())


      
class DistributedTrainingDevice(object):
  '''
  A distributed training device (Client or Server)    分布式培训设备(客户端或服务器)
  data : a pytorch dataset consisting datapoints (x,y)   pytorch数据集包含数据点(x,y)
  model : a pytorch neural net f mapping x -> f(x)=y_   pytorch神经网络f映射x - >f (x)=y_
  hyperparameters : a python dict containing all hyperparameters包含所有超参数的python字典
  '''
  def __init__(self, dataloader, model, hyperparameters, experiment):
    self.hp = hyperparameters
    self.xp = experiment
    self.loader = dataloader
    self.model = model
    self.loss_fn = nn.CrossEntropyLoss()


class Client(DistributedTrainingDevice):

  def __init__(self, dataloader, model, hyperparameters, experiment, id_num=0, dp_mechanism=None):
    super().__init__(dataloader, model, hyperparameters, experiment)

    self.id = id_num
    self.dp_mechanism = dp_mechanism

    # Parameters参数
    self.W = {name : value for name, value in self.model.named_parameters()}
    self.W_old = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.dW = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.dW_compressed = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.A = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

    self.n_params = sum([T.numel() for T in self.W.values()])
    self.bits_sent = []

    # Optimizer (specified in self.hp, initialized using the suitable parameters from self.hp)优化器
    optimizer_object = getattr(optim, self.hp['optimizer'])
    optimizer_parameters = {k : v for k, v in self.hp.items() if k in optimizer_object.__init__.__code__.co_varnames}

    self.optimizer = optimizer_object(self.model.parameters(), **optimizer_parameters)

    # Learning Rate Schedule学习率时间表
    self.scheduler = getattr(optim.lr_scheduler, self.hp['lr_decay'][0])(self.optimizer, **self.hp['lr_decay'][1])

    # State状态
    self.epoch = 0
    self.train_loss = 0.0
    
    # Create DP mechanism if enabled in hyperparameters
    if self.dp_mechanism is None and self.hp.get('differential_privacy', {}).get('enabled', False):
        dp_config = self.hp.get('differential_privacy', {})
        self.dp_mechanism = dp.GaussianNoiseDP(
            clip_bound=dp_config.get('clip_bound', 1.0),
            noise_multiplier=dp_config.get('noise_multiplier', 1.0),
            delta=dp_config.get('delta', 1e-5)
        )


  def synchronize_with_server(self, server):
    # W_client = W_server
    copy(target=self.W, source=server.W)


  def train_cnn(self, iterations):

    running_loss = 0.0
    for i in range(iterations):
      
      try: # Load new batch of data 加载新的一批数据
        x, y = next(self.epoch_loader)
      except: # Next epoch
        self.epoch_loader = iter(self.loader)
        self.epoch += 1

        # Adapt lr according to schedule 根据进度调整学习率
        if isinstance(self.scheduler, optim.lr_scheduler.LambdaLR):
          self.scheduler.step()
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and 'loss_test' in self.xp.results:
          self.scheduler.step(self.xp.results['loss_test'][-1])
        
        x, y = next(self.epoch_loader)

      x, y = x.to(device), y.to(device)
        
      # zero the parameter gradients 将梯度参数归零
      self.optimizer.zero_grad()

      # forward + backward + optimize
      y_ = self.model(x)

      loss = self.loss_fn(y_, y)
      loss.backward()
      
      # Apply differential privacy if enabled
      if self.dp_mechanism is not None:
          for param in self.model.parameters():
              if param.requires_grad and param.grad is not None:
                  param.grad = self.dp_mechanism.apply_dp(param.grad)
      
      self.optimizer.step()
      
      running_loss += loss.item()

    return running_loss / iterations


  def compute_weight_update(self, iterations=1):

    # Training mode 训练模式
    self.model.train()

    # W_old = W
    #同步参数
    copy(target=self.W_old, source=self.W)
    
    # W = SGD(W, D)
    #计算损失函数/不该是用SGD吗？权重
    self.train_loss = self.train_cnn(iterations)

    # dW = W - W_old
    subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)


  # 压缩重量更新
  def compress_weight_update_up(self, compression=None, accumulate=False, count_bits=False):

    if accumulate and compression[0] != "none":
      # compression with error accumulation     
      add(target=self.A, source=self.dW)
      compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
      subtract(target=self.A, source=self.dW_compressed)

    else: 
      # compression without error accumulation
      compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

    if count_bits:
      # Compute the update size
      self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]



class Server(DistributedTrainingDevice):

  def __init__(self, dataloader, model, hyperparameters, experiment, stats):
    super().__init__(dataloader, model, hyperparameters, experiment)

    # Parameters参数
    self.W = {name : value for name, value in self.model.named_parameters()}
    self.dW_compressed = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.dW = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

    self.A = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

    self.n_params = sum([T.numel() for T in self.W.values()])
    self.bits_sent = []

    self.client_sizes = torch.Tensor(stats["split"]).to(device)

    # Store privacy metrics if DP is enabled
    self.privacy_metrics = None
    if hyperparameters.get('differential_privacy', {}).get('enabled', False):
        dp_config = hyperparameters.get('differential_privacy', {})
        mechanism = dp.GaussianNoiseDP(
            clip_bound=dp_config.get('clip_bound', 1.0),
            noise_multiplier=dp_config.get('noise_multiplier', 1.0),
            delta=dp_config.get('delta', 1e-5)
        )
        self.privacy_metrics = {
            'epsilon': mechanism.get_privacy_spent(hyperparameters.get('communication_rounds', 200)),
            'delta': dp_config.get('delta', 1e-5),
            'noise_multiplier': dp_config.get('noise_multiplier', 1.0)
        }


  def aggregate_weight_updates(self, clients, aggregation="mean"):

    # dW = aggregate(dW_i, i=1,..,n)
    if aggregation == "mean":
      average(target=self.dW, sources=[client.dW_compressed for client in clients])
    #加权平均值
    elif aggregation == "weighted_mean":
      weighted_average(target=self.dW, sources=[client.dW_compressed for client in clients], 
        weights=torch.stack([self.client_sizes[client.id] for client in clients]))
    
    elif aggregation == "majority":
      majority_vote(target=self.dW, sources=[client.dW_compressed for client in clients], lr=self.hp["lr"])


  def compress_weight_update_down(self, compression=None, accumulate=False, count_bits=False):
    if accumulate and compression[0] != "none":
      # compression with error accumulation   
      add(target=self.A, source=self.dW)
      compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
      subtract(target=self.A, source=self.dW_compressed)

    else: 
      # compression without error accumulation无错误累计的压缩
      compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

    add(target=self.W, source=self.dW_compressed)

    if count_bits:
      # Compute the update size 计算更新大小
      self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]

 
  def evaluate(self, loader=None, max_samples=50000, verbose=True):
    """Evaluates local model stored in self.W on local dataset or other 'loader if specified for a maximum of 
    'max_samples and returns a dict containing all evaluation metrics
    评估存储在self中的本地模型。W在本地数据集或其他'loader如果指定的最大max_samples '，并返回一个包含所有评估指标的字典"""
    self.model.eval()

    eval_loss, correct, samples, iters = 0.0, 0, 0, 0
    if not loader:
      loader = self.loader
    with torch.no_grad():
      for i, (x,y) in enumerate(loader):

        x, y = x.to(device), y.to(device)
        y_ = self.model(x)
        _, predicted = torch.max(y_.data, 1)
        eval_loss += self.loss_fn(y_, y).item()
        correct += (predicted == y).sum().item()
        samples += y_.shape[0]
        iters += 1

        if samples >= max_samples:
          break
      if verbose:
        print("Evaluated on {} samples ({} batches)".format(samples, iters))
  
      results_dict = {'loss' : eval_loss/iters, 'accuracy' : correct/samples}

    return results_dict