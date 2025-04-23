import os
import collections
import logging
import glob
import re

import torch, torchvision
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset

import itertools as it
import copy


#-------------------------------------------------------------------------------------------------------
# DATASETS 数据集
#-------------------------------------------------------------------------------------------------------
os.environ['TRAINING_DATA'] = r"D:\dateset"
DATA_PATH = os.path.join(os.environ['TRAINING_DATA'], 'PyTorch')


def get_kws():
  '''Return MNIST train/test data and labels as numpy arrays  以numpy数组的形式返回MNIST序列/测试数据和标签'''
  data = np.load(os.path.join(DATA_PATH, "Speech_Commands/data_noaug.npz"))
  
  x_train, y_train = data["x_train"], data["y_train"].astype("int").flatten()
  x_test, y_test = data["x_test"], data["y_test"].astype("int").flatten()

  x_train, x_test = (x_train-0.5412386)/0.2746128, (x_test-0.5412386)/0.2746128
  
  return x_train, y_train, x_test, y_test


def get_mnist():
  '''Return MNIST train/test data and labels as numpy arrays  以numpy数组的形式返回MNIST序列/测试数据和标签'''
  data_train = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=True, download=True) 
  data_test = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=False, download=True) 
  
  # 兼容新版本PyTorch
  try:
    x_train, y_train = data_train.train_data.numpy().reshape(-1,1,28,28)/255, np.array(data_train.train_labels)
    x_test, y_test = data_test.test_data.numpy().reshape(-1,1,28,28)/255, np.array(data_test.test_labels)
  except AttributeError:
    x_train, y_train = data_train.data.numpy().reshape(-1,1,28,28)/255, np.array(data_train.targets)
    x_test, y_test = data_test.data.numpy().reshape(-1,1,28,28)/255, np.array(data_test.targets)
  
  return x_train, y_train, x_test, y_test


def get_fashionmnist():
  '''Return MNIST train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_PATH, "FashionMNIST"), train=True, download=True) 
  data_test = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_PATH, "FashionMNIST"), train=False, download=True) 
  
  # 兼容新版本PyTorch
  try:
    x_train, y_train = data_train.train_data.numpy().reshape(-1,1,28,28)/255, np.array(data_train.train_labels)
    x_test, y_test = data_test.test_data.numpy().reshape(-1,1,28,28)/255, np.array(data_test.test_labels)
  except AttributeError:
    x_train, y_train = data_train.data.numpy().reshape(-1,1,28,28)/255, np.array(data_train.targets)
    x_test, y_test = data_test.data.numpy().reshape(-1,1,28,28)/255, np.array(data_test.targets)

  return x_train, y_train, x_test, y_test


def get_cifar10():
  '''Return CIFAR10 train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=True, download=True) 
  data_test = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=False, download=True) 
  
  # 兼容新版本PyTorch
  try:
    x_train, y_train = data_train.train_data.transpose((0,3,1,2)), np.array(data_train.train_labels)
    x_test, y_test = data_test.test_data.transpose((0,3,1,2)), np.array(data_test.test_labels)
  except AttributeError:
    x_train, y_train = np.transpose(data_train.data, (0,3,1,2)), np.array(data_train.targets)
    x_test, y_test = np.transpose(data_test.data, (0,3,1,2)), np.array(data_test.targets)
  
  return x_train, y_train, x_test, y_test


def print_image_data_stats(data_train, labels_train, data_test, labels_test):
  print("训练数据标签数据的大小，特征范围，标签范围")
  print("\nData: ")
  print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
      np.min(labels_train), np.max(labels_train)))
  # 获取non-iid数据是要排序获取数据  获取iid数据是打乱数据集获取数据
  print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
      np.min(labels_test), np.max(labels_test)))


#-------------------------------------------------------------------------------------------------------
# SPLIT DATA AMONG CLIENTS 划分数据
#-------------------------------------------------------------------------------------------------------
def split_image_data(data, labels, n_clients=10, classes_per_client=10, shuffle=True, verbose=True, balancedness=None):
  '''
  Splits (data, labels) evenly among 'n_clients s.t. every client holds 'classes_per_client  每个参与者平均分割
  different labels
  data : [n_data x shape]
  labels : [n_data (x 1)] from 0 to n_labels
  '''
  # constants
  n_data = data.shape[0]
  print("n_data=60000",n_data)
  n_labels = np.max(labels) + 1
  print("n_labels=10", n_labels)
  print("n_clients=2000",n_clients)
  if balancedness >= 1.0:
    # data_per_client = 60000?
    data_per_client = [n_data // n_clients]*n_clients
    print("data_per_client 每个人有30个样本 ", data_per_client )
    data_per_client_per_class = [data_per_client[0] // classes_per_client]*n_clients
    print("data_per_client_per_class 30个样本中，共有10个类别，每个类别有3个",data_per_client_per_class)
    print(len(data_per_client_per_class)) #2000
  else:
    fracs = balancedness**np.linspace(0,n_clients-1, n_clients)
    fracs /= np.sum(fracs)
    fracs = 0.1/n_clients + (1-0.1)*fracs
    data_per_client = [np.floor(frac*n_data).astype('int') for frac in fracs]

    data_per_client = data_per_client[::-1]

    data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]

  if sum(data_per_client) > n_data:
    print("Impossible Split")
    exit()
  
  # sort for labels 对标签排序 就是想获取
  data_idcs = [[] for i in range(n_labels)]
  for j, label in enumerate(labels):
    data_idcs[label] += [j]
  if shuffle:
    for idcs in data_idcs:
      np.random.shuffle(idcs)
    
  # split data among clients 在客户端之间分割数据
  clients_split = []
  c = 0
  print("n_clients",n_clients)
  for i in range(n_clients):
    client_idcs = []
    budget = data_per_client[i] # budget=30
    c = np.random.randint(n_labels)
    while budget > 0:
      take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
      
      client_idcs += data_idcs[c][:take]
      data_idcs[c] = data_idcs[c][take:]
      
      budget -= take
      c = (c + 1) % n_labels
      
    clients_split += [(data[client_idcs], labels[client_idcs])]
  # 主要就是这句话是什么意思了
  # print("clients_split第一个客户端的特征",clients_split[0][0])
  print("clients_split第一个客户端的标签形状", clients_split[0][0].shape)
  print("clients_split第一个客户端的标签", clients_split[0][1])
  # 使用动态索引，打印最后一个客户端的信息，避免索引超出范围
  last_client_idx = n_clients - 1
  print(f"clients_split第{last_client_idx}个客户端的标签", clients_split[last_client_idx][1])
  print(f"clients_split第{last_client_idx}个客户端的标签形状", clients_split[last_client_idx][1].shape)

 # 打印每个客户端，30个样本中每个类别的数量
 #  print("np.arange(n_labels)",np.arange(n_labels))
 #  print("-1,-1是什么",np.arange(n_labels).reshape(-1,1))
  def print_split(clients_split):
    print("数据划分")
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()

  if verbose:
    print_split(clients_split)
        
  return clients_split


#-------------------------------------------------------------------------------------------------------
# IMAGE DATASET CLASS 图像数据集类
#-------------------------------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
  '''
  A custom Dataset class for images
  inputs : numpy array [n_data x shape]
  labels : numpy array [n_data (x 1)]
  '''
  def __init__(self, inputs, labels, transforms=None):
      assert inputs.shape[0] == labels.shape[0]
      self.inputs = torch.Tensor(inputs)
      self.labels = torch.Tensor(labels).long()
      self.transforms = transforms 

  def __getitem__(self, index):
      img, label = self.inputs[index], self.labels[index]

      if self.transforms is not None:
        img = self.transforms(img)

      return (img, label)

  def __len__(self):
      return self.inputs.shape[0]
          
# 处理图片
def get_default_data_transforms(name, train=True, verbose=True):
  transforms_train = {
  'mnist' : transforms.Compose([
    transforms.ToPILImage(),
    # 将图片压缩为32×32
    transforms.Resize((32, 32)),
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    # 将数据归一化到0.06-0.197
    transforms.Normalize((0.06078,),(0.1957,))
    ]),
  'fashionmnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]),
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
  'kws' : None
  }
  transforms_eval = {
  'mnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.06078,),(0.1957,))
    ]),
  'fashionmnist' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]),
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#
  'kws' : None
  }

  if verbose:
    print("\nData preprocessing: ")
    for transformation in transforms_train[name].transforms:
      print(' -', transformation)
    print()

  return (transforms_train[name], transforms_eval[name])


def get_data_loaders(hp, verbose=True):
  
  x_train, y_train, x_test, y_test = globals()['get_'+hp['dataset']]()
  # 60000 28 28 1
  # # print("训练数据集",x_train)
  # print("训练数据集的形状",x_train.shape)
  # print("训练标签数据集",y_train)
  # print("训练标签数据集的形状", y_train.shape)
  if verbose:
    # 输出图片的数据
    print_image_data_stats(x_train, y_train, x_test, y_test)

  transforms_train, transforms_eval = get_default_data_transforms(hp['dataset'], verbose=False)

  # 开始划分数据，就是每个人多少条数据 有多少个特征值看不出来  划分数据是这里
  split = split_image_data(x_train, y_train, n_clients=hp['n_clients'], 
          classes_per_client=hp['classes_per_client'], balancedness=hp['balancedness'], verbose=verbose)

  # 每个客户端 是 30个样本 ，每个样本是28×28的像素
  # print("一个客户端的特征：" ,split[0][0])
  print("一个客户端的特征的形状：", (split[0][0]).shape)
  # 因为是30个样本，所以每个客户端是30个标签值
  print("一个客户端的标签：", split[0][1])
  # print("一个客户端的标签的形状：", split[0][1].shape)

  # 划分好数据 2000个客户端可以进行下载
  client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), 
                                                                batch_size=hp['batch_size'], shuffle=True) for x, y in split]
  #客户端去下载数据
  # print("client_loaders是什么",client_loaders)

  # 客户端下载训练数据
  train_loader = torch.utils.data.DataLoader(CustomImageDataset(x_train, y_train, transforms_eval), batch_size=100, shuffle=False)
  # 客户端下载测试数据
  test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

  stats = {"split" : [x.shape[0] for x, y in split]}
  # 每个人有30个样本
  print("stats",stats)
  print("stats的长度", len(stats))

  return client_loaders, train_loader, test_loader, stats

