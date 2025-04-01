import torch
import torch.nn as nn
import numpy as np


class MaskLayer:
    """管理网络层的剪枝掩码"""
    def __init__(self, weight, threshold=0.001):
        self.threshold = threshold
        self.mask = torch.ones_like(weight.data)  # 初始全为1的掩码
        self.pruned_weights = 0
        self.total_weights = weight.numel()
        self.weight_tensor = weight  # 保存对权重张量的引用
    
    def prune(self):
        """剪枝操作：将小于阈值的权重对应的掩码设为0"""
        new_mask = torch.where(torch.abs(self.weight_tensor.data) < self.threshold, 
                              torch.zeros_like(self.mask), 
                              self.mask)
        self.pruned_weights += torch.sum((self.mask - new_mask) > 0).item()
        self.mask = new_mask
        # 应用掩码
        self.weight_tensor.data *= self.mask
    
    def splice(self):
        """恢复操作：如果权重变得大于阈值，恢复对应的掩码"""
        weight_abs = torch.abs(self.weight_tensor.data)
        # 找出那些当前被掩码为0但权重变大的参数
        potential_splices = (self.mask == 0) & (weight_abs > self.threshold)
        self.mask[potential_splices] = 1
        self.pruned_weights -= torch.sum(potential_splices).item()
    
    def get_prune_ratio(self):
        """获取剪枝率"""
        return self.pruned_weights / self.total_weights


class DNSPruner:
    """Dynamic Network Surgery剪枝器"""
    def __init__(self, model, thresholds=None, excluded_layers=None):
        self.model = model
        self.masks = {}
        self.thresholds = thresholds or {}
        self.excluded_layers = excluded_layers or []
        self._init_masks()
    
    def _init_masks(self):
        """为网络中的参数初始化掩码"""
        for name, module in self.model.named_modules():
            if name in self.excluded_layers:
                continue
                
            # 只为卷积层和全连接层创建掩码
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                threshold = self.thresholds.get(name, 0.001)  # 默认阈值
                self.masks[name] = MaskLayer(module.weight, threshold)
    
    def prune(self):
        """对所有层执行剪枝操作"""
        for mask in self.masks.values():
            mask.prune()
    
    def splice(self):
        """对所有层执行恢复操作"""
        for mask in self.masks.values():
            mask.splice()
    
    def apply_masks(self):
        """应用所有掩码到模型权重"""
        for name, mask in self.masks.items():
            for module_name, module in self.model.named_modules():
                if module_name == name:
                    module.weight.data *= mask.mask
    
    def get_prune_statistics(self):
        """获取剪枝统计信息"""
        stats = {}
        total_params = 0
        total_pruned = 0
        
        for name, mask in self.masks.items():
            prune_ratio = mask.get_prune_ratio()
            stats[name] = prune_ratio
            total_params += mask.total_weights
            total_pruned += mask.pruned_weights
        
        overall_ratio = total_pruned / total_params if total_params > 0 else 0
        stats['overall'] = overall_ratio
        return stats


class PruningHook:
    """在训练循环中应用DNS剪枝的钩子"""
    def __init__(self, pruner, prune_frequency=100, splice_frequency=300):
        self.pruner = pruner
        self.prune_frequency = prune_frequency
        self.splice_frequency = splice_frequency
        self.global_step = 0
    
    def after_step(self):
        """在优化器步骤后调用"""
        # 应用掩码
        self.pruner.apply_masks()
        
        # 执行剪枝和恢复操作
        if self.global_step % self.prune_frequency == 0:
            self.pruner.prune()
        
        if self.global_step % self.splice_frequency == 0:
            self.pruner.splice()
        
        self.global_step += 1
