"""
病理特征聚合器
用于将变长的病理特征 (N, 1024) 聚合为固定长度的特征向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal, Optional


class PathologyAggregator(nn.Module):
    """
    病理特征聚合器
    
    输入: 病理特征列表，每个元素形状为 (N_i, 1024)
    输出: 聚合后的特征，形状为 (batch_size, 512)
    
    支持两种聚合模式:
    1. 平均池化 + MLP
    2. 基于注意力的聚合 (类似ABMIL)
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 512,
        aggregation_mode: Literal["mean", "attention"] = "mean",
        dropout: float = 0.1,
        num_attention_heads: int = 4
    ):
        """
        初始化聚合器
        
        Args:
            input_dim: 输入特征维度 (默认1024)
            hidden_dim: 隐藏层维度 (默认512)
            output_dim: 输出特征维度 (默认512)
            aggregation_mode: 聚合模式，"mean" 或 "attention"
            dropout: Dropout概率
            num_attention_heads: 注意力头数 (仅用于attention模式)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aggregation_mode = aggregation_mode
        self.dropout = dropout
        
        if aggregation_mode == "mean":
            # 平均池化 + MLP
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
        elif aggregation_mode == "attention":
            # 基于注意力的聚合 (类似ABMIL)
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)  # 注意力分数
            )
            
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # 多头注意力版本 (可选)
            self.multihead_attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.use_multihead = num_attention_heads > 1
            
        else:
            raise ValueError(f"不支持的聚合模式: {aggregation_mode}。请使用 'mean' 或 'attention'")
    
    def forward_mean(self, path_features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        平均池化聚合
        
        Args:
            path_features_list: 病理特征列表，每个元素形状 (N_i, 1024)
            
        Returns:
            聚合后的特征，形状 (batch_size, 512)
        """
        batch_size = len(path_features_list)
        aggregated_features = []
        
        for i in range(batch_size):
            # 对每个样本的patch进行平均池化
            features = path_features_list[i]  # (N_i, 1024)
            mean_features = torch.mean(features, dim=0)  # (1024,)
            
            # 通过MLP
            output = self.mlp(mean_features)  # (512,)
            aggregated_features.append(output)
        
        # 堆叠为批次
        return torch.stack(aggregated_features, dim=0)  # (batch_size, 512)
    
    def forward_attention(self, path_features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        注意力聚合
        
        Args:
            path_features_list: 病理特征列表，每个元素形状 (N_i, 1024)
            
        Returns:
            聚合后的特征，形状 (batch_size, 512)
        """
        batch_size = len(path_features_list)
        aggregated_features = []
        
        for i in range(batch_size):
            features = path_features_list[i]  # (N_i, 1024)
            N = features.shape[0]
            
            if self.use_multihead:
                # 使用多头注意力
                # 添加批次维度: (N_i, 1024) -> (1, N_i, 1024)
                features_batch = features.unsqueeze(0)
                
                # 多头注意力
                attn_output, _ = self.multihead_attention(
                    features_batch, features_batch, features_batch
                )  # (1, N_i, 1024)
                
                # 计算注意力权重
                attn_scores = self.attention(attn_output.squeeze(0))  # (N_i, 1)
            else:
                # 计算注意力分数
                attn_scores = self.attention(features)  # (N_i, 1)
            
            # 应用softmax得到注意力权重
            attn_weights = F.softmax(attn_scores, dim=0)  # (N_i, 1)
            
            # 加权求和
            weighted_features = torch.sum(features * attn_weights, dim=0)  # (1024,)
            
            # 通过MLP
            output = self.mlp(weighted_features)  # (512,)
            aggregated_features.append(output)
        
        # 堆叠为批次
        return torch.stack(aggregated_features, dim=0)  # (batch_size, 512)
    
    def forward(self, path_features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            path_features_list: 病理特征列表，每个元素形状 (N_i, 1024)
            
        Returns:
            聚合后的特征，形状 (batch_size, 512)
        """
        if self.aggregation_mode == "mean":
            return self.forward_mean(path_features_list)
        elif self.aggregation_mode == "attention":
            return self.forward_attention(path_features_list)
        else:
            raise ValueError(f"不支持的聚合模式: {self.aggregation_mode}")
    
    def get_attention_weights(self, path_features: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重 (仅适用于attention模式)
        
        Args:
            path_features: 单个样本的病理特征，形状 (N, 1024)
            
        Returns:
            注意力权重，形状 (N, 1)
        """
        if self.aggregation_mode != "attention":
            raise ValueError("注意力权重仅适用于attention模式")
        
        if self.use_multihead:
            features_batch = path_features.unsqueeze(0)  # (1, N, 1024)
            attn_output, _ = self.multihead_attention(
                features_batch, features_batch, features_batch
            )
            attn_scores = self.attention(attn_output.squeeze(0))  # (N, 1)
        else:
            attn_scores = self.attention(path_features)  # (N, 1)
        
        attn_weights = F.softmax(attn_scores, dim=0)
        return attn_weights


def test_aggregator():
    """测试病理特征聚合器"""
    import numpy as np
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模拟数据
    batch_size = 4
    path_features_list = []
    
    for i in range(batch_size):
        N = np.random.randint(10, 100)  # 随机patch数
        features = torch.randn(N, 1024)
        path_features_list.append(features)
    
    print(f"创建了 {batch_size} 个样本")
    print(f"每个样本的patch数: {[f.shape[0] for f in path_features_list]}")
    
    # 测试平均池化聚合器
    print("\n测试平均池化聚合器...")
    mean_aggregator = PathologyAggregator(aggregation_mode="mean")
    mean_output = mean_aggregator(path_features_list)
    print(f"输出形状: {mean_output.shape}")
    print(f"期望形状: ({batch_size}, 512)")
    
    # 测试注意力聚合器
    print("\n测试注意力聚合器...")
    attention_aggregator = PathologyAggregator(aggregation_mode="attention")
    attention_output = attention_aggregator(path_features_list)
    print(f"输出形状: {attention_output.shape}")
    print(f"期望形状: ({batch_size}, 512)")
    
    # 测试多头注意力聚合器
    print("\n测试多头注意力聚合器...")
    multihead_aggregator = PathologyAggregator(
        aggregation_mode="attention",
        num_attention_heads=4
    )
    multihead_output = multihead_aggregator(path_features_list)
    print(f"输出形状: {multihead_output.shape}")
    print(f"期望形状: ({batch_size}, 512)")
    
    # 测试注意力权重
    print("\n测试注意力权重...")
    sample_features = path_features_list[0]
    attn_weights = attention_aggregator.get_attention_weights(sample_features)
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"权重和: {attn_weights.sum().item():.4f} (应接近1.0)")
    
    # 验证输出维度
    assert mean_output.shape == (batch_size, 512), f"平均池化输出形状错误: {mean_output.shape}"
    assert attention_output.shape == (batch_size, 512), f"注意力输出形状错误: {attention_output.shape}"
    assert multihead_output.shape == (batch_size, 512), f"多头注意力输出形状错误: {multihead_output.shape}"
    
    print("\n所有测试通过!")


if __name__ == "__main__":
    test_aggregator()