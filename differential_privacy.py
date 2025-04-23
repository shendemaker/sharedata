#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements differential privacy mechanisms for federated learning.
"""

import torch
import numpy as np
import math

class GaussianNoiseDP:
    """Gaussian noise differential privacy mechanism"""
    
    def __init__(self, clip_bound, noise_multiplier, delta=1e-5):
        """
        Initialize differential privacy parameters
        
        Parameters:
            clip_bound (float): Gradient clipping bound
            noise_multiplier (float): Noise multiplier controlling privacy budget
            delta (float): Differential privacy parameter δ
        """
        self.clip_bound = clip_bound
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        
    def apply_dp(self, gradients):
        """
        Apply differential privacy processing to gradients
        
        Parameters:
            gradients: Model gradients
            
        Returns:
            Gradients with added noise
        """
        # 1. Gradient clipping
        grad_norm = torch.norm(gradients)
        scale = min(1.0, self.clip_bound / (grad_norm + 1e-10))
        clipped_gradients = gradients * scale
        
        # 2. Add Gaussian noise
        noise_std = self.clip_bound * self.noise_multiplier
        noise = torch.randn_like(clipped_gradients) * noise_std
        
        # 3. Return gradients with added noise
        return clipped_gradients + noise
    
    def get_privacy_spent(self, iterations):
        """
        Calculate privacy budget spent (ε)
        
        Parameters:
            iterations: Number of training iterations
            
        Returns:
            epsilon: Privacy budget ε
        """
        # Simplified calculation using moments accountant
        # For production use, a more sophisticated privacy analysis should be used
        c = self.noise_multiplier * self.clip_bound
        epsilon = np.sqrt(2 * np.log(1.25 / self.delta)) * (self.clip_bound / c) * np.sqrt(iterations)
        return epsilon
    
    def privacy_info(self, iterations):
        """
        Get privacy information as a dictionary
        
        Parameters:
            iterations: Number of training iterations
            
        Returns:
            Dictionary with privacy parameters
        """
        epsilon = self.get_privacy_spent(iterations)
        return {
            "epsilon": epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "clip_bound": self.clip_bound,
            "iterations": iterations
        }

def get_dp_mechanism(config, iterations=None):
    """
    Create a differential privacy mechanism based on configuration
    
    Parameters:
        config: Configuration dictionary
        iterations: Optional number of iterations for privacy calculation
        
    Returns:
        DP mechanism instance or None if DP is disabled
    """
    if not config.get("differential_privacy", {}).get("enabled", False):
        return None
    
    dp_config = config["differential_privacy"]
    mechanism = GaussianNoiseDP(
        clip_bound=dp_config.get("clip_bound", 1.0),
        noise_multiplier=dp_config.get("noise_multiplier", 1.0),
        delta=dp_config.get("delta", 1e-5)
    )
    
    return mechanism 