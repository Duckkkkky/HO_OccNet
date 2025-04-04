#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :transformer.py
#@Date        :2022/09/29 15:06:45
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VideoTransformer(nn.Module):
    def __init__(self, image_size, num_frames, dim=256, depth=4, heads=4, in_channels=256, dim_head=64, dropout=0., scale_dim=4, use_temporary_embedding=False, patch_size=1, sep_output=True):
        super().__init__()
        self.sep_output = sep_output
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.image_size = image_size
        if use_temporary_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches, dim))
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        x += self.pos_embedding[:, :t, :n]
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b t) n d -> (b n) t d', t=t)
        x = self.temporal_transformer(x)
        x = rearrange(x, '(b n) t d -> b t d n', n=n)
        x = rearrange(x, 'b t d (h w) -> b t d h w', h=self.image_size // self.patch_size, w=self.image_size // self.patch_size)

        if self.sep_output:
            feat_list = []
            for i in range(t):
                feat_list.append(x[:, i, :, :, :])
            return feat_list
        else:
            return x

class ILAVideoTransformer(nn.Module):
    def __init__(self, image_size, num_frames, dim=256, depth=4, heads=4, in_channels=256, 
                 dim_head=64, dropout=0., scale_dim=4, use_temporary_embedding=False, 
                 patch_size=1, sep_output=True):
        super().__init__()
        self.sep_output = sep_output
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.image_size = image_size
        self.dim = dim
        
        # 计算分块数量和维度
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        
        # 特征提取器
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        #位置编码
        if use_temporary_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches, dim))
        
        # cls_token用于全局时序建模
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
        
        # 空间和时间transformer
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        # self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        
        # ILA相关组件
        self.interactive_block = nn.Sequential(
            nn.Conv2d(dim * 2, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.prediction_block = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            SqueezeAndRerange(-2, -1, T=num_frames),
            nn.Conv1d(128, 64, 1, groups=2),
            nn.ReLU(),
            nn.Conv1d(64, 4, 1, groups=2),
            nn.Tanh()
        )
        
        self.conv = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.fc = nn.Linear(128, dim)
        
    def forward(self, x):
        b, t, c, h, w = x.shape
        # 编码特征
        x = self.to_patch_embedding(x)  # b t n d
        b, t, n, d = x.shape
        
        # 添加位置编码
        x += self.pos_embedding[:, :t, :n]
        
        # ILA处理：处理相邻帧间交互
        if t > 1:
            # 前后帧特征
            pre_features = x[:, :-1]  # b (t-1) n d
            cur_features = x[:, 1:]   # b (t-1) n d
            
            # 将前后帧特征组织为交互特征
            pre_projed = rearrange(pre_features, "b t n d -> b d t n", b=b, t=t-1, n=n)
            cur_projed = rearrange(cur_features, "b t n d -> b d t n", b=b, t=t-1, n=n)
            
            # 重整特征用于卷积处理
            pre_projed = rearrange(pre_projed, "b d t (h w) -> b d t h w", h=h//self.patch_size, w=w//self.patch_size)
            cur_projed = rearrange(cur_projed, "b d t (h w) -> b d t h w", h=h//self.patch_size, w=w//self.patch_size)
            
            # 特征融合
            pairs = torch.cat([pre_projed, cur_projed], dim=1)
            pairs = rearrange(pairs, "b d t h w -> (b t) d h w")
            
            # 交互特征计算
            interactive_feature = self.interactive_block(pairs)
            
            # 尝试使用prediction_block产生位置信息
            position_offsets = self.prediction_block(interactive_feature)
            
            
            # 分离交互特征
            pre_interactive = interactive_feature[:, :128]
            cur_interactive = interactive_feature[:, 128:]
            
            # 重塑为时间序列
            pre_interactive = rearrange(pre_interactive, "(b t) d h w -> b d t h w", b=b, t=t-1)
            cur_interactive = rearrange(cur_interactive, "(b t) d h w -> b d t h w", b=b, t=t-1)
            
            # 通过位置偏移量调整交互特征
            # 这里简单地将偏移量作为加权因子，影响很小但确保梯度流动
            position_weight = position_offsets.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
            position_weight = torch.sigmoid(position_weight) * 0.1 + 0.95  # 使范围在0.95-1.05之间
            
            # 对交互特征使用位置权重
            pre_interactive = pre_interactive * position_weight.unsqueeze(-1).unsqueeze(-1)
            cur_interactive = cur_interactive * position_weight.unsqueeze(-1).unsqueeze(-1)
            
            # 时序融合
            first_frame = pre_interactive[:, :, :1]
            middle_frames = (pre_interactive[:, :, 1:] + cur_interactive[:, :, :-1]) / 2 if t > 2 else torch.tensor([]).to(pre_interactive.device)
            last_frame = cur_interactive[:, :, -1:]
            
            # 组合所有帧
            if t > 2:
                all_frames = torch.cat([first_frame, middle_frames, last_frame], dim=2)
            else:
                all_frames = torch.cat([first_frame, last_frame], dim=2)
            
            # 3D卷积增强时序关系
            enhanced_features = self.conv(all_frames)
            enhanced_features = rearrange(enhanced_features, "b d t h w -> b t (h w) d")
            
            # 将特征维度从128映射回原始维度256
            aligned_features = self.fc(enhanced_features)
            
            # 在这里，将对齐后的特征与原始特征结合
            x_aligned = rearrange(aligned_features, "b t n d -> (b t) n d")
        else:
            x_aligned = rearrange(x, "b t n d -> (b t) n d")
            
        # 添加cls_token
        cls_tokens = repeat(self.cls_token, '1 1 1 d -> b t 1 d', b=b, t=t)
        cls_tokens = rearrange(cls_tokens, "b t 1 d -> (b t) 1 d")
        x_with_cls = torch.cat((cls_tokens, x_aligned), dim=1)
        
        # 只进行空间transformer处理
        x_final = self.spatial_transformer(x_with_cls)
        
        # 提取cls_token和空间特征
        cls_spatial = x_final[:, 0:1, :]
        x_spatial = x_final[:, 1:, :]
        
        # 转换为最终输出格式，不包含cls_token
        x = rearrange(x_spatial, '(b t) n d -> b t d n', b=b, t=t)
        x = rearrange(x, 'b t d (h w) -> b t d h w', h=self.image_size // self.patch_size, w=self.image_size // self.patch_size)
        
        if self.sep_output:
            feat_list = []
            for i in range(t):
                feat_list.append(x[:, i, :, :, :])
            return feat_list
        else:
            return x

class FSATransformer(nn.Module):
    """Factorized Self-Attention Transformer Encoder"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                 ]))

    def forward(self, x):
        b, t, n, _ = x.shape
        x = rearrange(x, 'b t n d -> (b t) n d')

        for sp_attn, temp_attn, ff in self.layers:
            sp_attn_x = sp_attn(x) + x  # Spatial attention

            # Reshape tensors for temporal attention
            sp_attn_x = rearrange(sp_attn_x, '(b t) n d -> (b n) t d', n=n, t=t)
            temp_attn_x = temp_attn(sp_attn_x) + sp_attn_x  # Temporal attention
            x = ff(temp_attn_x) + temp_attn_x  # MLP

            # Again reshape tensor for spatial attention
            x = rearrange(x, '(b n) t d -> (b t) n d', n=n, t=t)

        x = rearrange(x, '(b t) n d -> b t d n', t=t, n=n)

        return x


class FactorizedVideoTransformer(nn.Module):
    def __init__(self, image_size, num_frames, dim=256, depth=4, heads=4, in_channels=256, dim_head=64, dropout=0., scale_dim=4, use_temporary_embedding=False, patch_size=1, sep_output=True):
        super().__init__()
        self.sep_output = sep_output
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.image_size = image_size
        if use_temporary_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches, dim))
        self.transformer = FSATransformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        x += self.pos_embedding[:, :t, :n]
        x = self.transformer(x)
        x = rearrange(x, 'b t d (h w) -> b t d h w', h=self.image_size // self.patch_size, w=self.image_size // self.patch_size)

        if self.sep_output:
            feat_list = []
            for i in range(t):
                feat_list.append(x[:, i, :, :, :])
            return feat_list
        else:
            return x