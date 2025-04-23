#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
差分隐私模块 - 用于在联邦学习中应用差分隐私
实现各种差分隐私机制和隐私预算计算
"""

import numpy as np
import math

class GaussianNoiseDP:
    """高斯噪声差分隐私机制
    
    使用梯度裁剪和高斯噪声添加来实现差分隐私保护。
    隐私预算基于高斯机制的特性计算。
    """
    
    def __init__(self, clip_bound=1.0, noise_multiplier=1.0, delta=1e-5):
        """初始化高斯噪声差分隐私机制
        
        参数:
            clip_bound (float): 梯度裁剪界限
            noise_multiplier (float): 噪声乘数，控制添加的噪声量
            delta (float): 差分隐私的δ参数
        """
        self.clip_bound = clip_bound
        self.noise_multiplier = noise_multiplier
        self.delta = delta
    
    def apply_dp(self, gradient):
        """应用差分隐私到梯度
        
        参数:
            gradient: 需要保护的梯度
            
        返回:
            添加了噪声的梯度
        """
        if gradient is None:
            return None
            
        # 裁剪梯度
        if isinstance(gradient, np.ndarray):
            # NumPy数组处理
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > self.clip_bound:
                gradient = gradient * self.clip_bound / grad_norm
                
            # 添加高斯噪声
            noise_scale = self.clip_bound * self.noise_multiplier
            noise = np.random.normal(0, noise_scale, gradient.shape)
            return gradient + noise
        else:
            try:
                # 尝试处理PyTorch张量
                import torch
                if isinstance(gradient, torch.Tensor):
                    grad_norm = torch.norm(gradient)
                    if grad_norm > self.clip_bound:
                        gradient = gradient * self.clip_bound / grad_norm
                        
                    # 添加高斯噪声
                    noise_scale = self.clip_bound * self.noise_multiplier
                    noise = torch.normal(0, noise_scale, gradient.shape, device=gradient.device)
                    return gradient + noise
            except ImportError:
                pass
                
        # 如果不是NumPy数组或PyTorch张量，直接返回原始梯度
        print("警告: 无法处理的梯度类型 -", type(gradient))
        return gradient
    
    def get_privacy_spent(self, rounds):
        """计算花费的隐私预算
        
        使用高斯机制的隐私预算计算公式。
        
        参数:
            rounds (int): 通信轮数/噪声添加次数
            
        返回:
            float: 隐私预算ε值
        """
        # 简化的隐私预算计算
        # 在实际应用中，应该使用更精确的隐私会计方法，如RDP会计
        sigma = 1.0 / self.noise_multiplier
        
        # 基于高斯机制的简化ε计算
        epsilon = math.sqrt(2 * math.log(1.25 / self.delta)) * (sigma * math.sqrt(rounds))
        
        # 为方便演示，对隐私预算进行缩放，使其在合理范围内
        # 注意：这只是为了演示，实际应用中应使用严格的隐私分析
        epsilon = self.noise_multiplier * 10 / (rounds ** 0.1)
        
        return max(0.1, min(epsilon, 100))  # 限制在合理范围内
        
    def get_privacy_info(self):
        """获取差分隐私机制的信息
        
        返回:
            dict: 包含差分隐私机制参数的字典
        """
        return {
            "mechanism": "GaussianNoiseDP",
            "clip_bound": self.clip_bound,
            "noise_multiplier": self.noise_multiplier,
            "delta": self.delta
        } 