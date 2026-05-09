"""
组学特征编码器
用于编码组学特征 (17472个基因 × 3个通道)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal


class OmicsEncoder(nn.Module):
    """
    组学特征编码器
    
    输入: 组学特征张量，形状 (batch_size, G, 3)，其中 G=17472
    输出: 编码后的特征，形状 (batch_size, 512)
    
    支持两种编码架构:
    1. 1D CNN: 使用卷积层沿基因方向编码
    2. Transformer: 使用Transformer编码器
    """
    
    def __init__(
        self,
        input_genes: int = 17472,
        input_channels: int = 3,
        hidden_dim: int = 512,
        output_dim: int = 512,
        encoder_type: Literal["cnn", "transformer"] = "cnn",
        num_layers: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.1,
        num_heads: int = 8,
        intermediate_dim: int = 2048
    ):
        """
        初始化组学编码器
        
        Args:
            input_genes: 输入基因数量 (默认17472)
            input_channels: 输入通道数 (默认3: RNA-seq, CNV, Methylation)
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            encoder_type: 编码器类型，"cnn" 或 "transformer"
            num_layers: 编码器层数
            kernel_size: 卷积核大小 (仅用于CNN)
            dropout: Dropout概率
            num_heads: 注意力头数 (仅用于Transformer)
            intermediate_dim: Transformer中间层维度
        """
        super().__init__()
        
        self.input_genes = input_genes
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        
        if encoder_type == "cnn":
            # 1D CNN编码器
            self.conv_layers = nn.ModuleList()
            
            # 第一层: 将通道维度转换为隐藏维度
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=hidden_dim // 4,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            
            # 中间层
            for i in range(1, num_layers - 1):
                in_ch = hidden_dim // 4 if i == 1 else hidden_dim // 2
                out_ch = hidden_dim // 2
                self.conv_layers.append(
                    nn.Conv1d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )
                )
            
            # 最后一层
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=hidden_dim // 2,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            
            # 批归一化和激活函数
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim // 4),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim)
            ])
            
            # 全局平均池化
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            
            # 输出投影层
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
            
        elif encoder_type == "transformer":
            # Transformer编码器
            # 首先将输入投影到隐藏维度
            self.input_proj = nn.Linear(input_channels, hidden_dim)
            
            # 位置编码
            self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
            
            # Transformer编码器层
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=intermediate_dim,
                dropout=dropout,
                activation='relu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
            
            # 全局平均池化 (沿基因维度)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            
            # 输出投影层
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
            
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}。请使用 'cnn' 或 'transformer'")
    
    def forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        CNN编码器前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, G, 3)
            
        Returns:
            编码后的特征，形状 (batch_size, 512)
        """
        batch_size = x.shape[0]
        
        # 调整维度: (batch, G, channels) -> (batch, channels, G)
        x = x.transpose(1, 2)  # (batch, 3, 17472)
        
        # 通过卷积层
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            
            # 应用批归一化和激活函数
            if i < len(self.bn_layers):
                x = self.bn_layers[i](x)
            
            x = F.relu(x)
            
            # 除了最后一层，添加dropout
            if i < len(self.conv_layers) - 1:
                x = F.dropout(x, p=0.1, training=self.training)
        
        # 全局平均池化
        x = self.global_pool(x)  # (batch, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch, hidden_dim)
        
        # 输出投影
        x = self.output_proj(x)  # (batch, output_dim)
        
        return x
    
    def forward_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer编码器前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, G, 3)
            
        Returns:
            编码后的特征，形状 (batch_size, 512)
        """
        batch_size = x.shape[0]
        
        # 输入投影: (batch, G, 3) -> (batch, G, hidden_dim)
        x = self.input_proj(x)  # (batch, 17472, hidden_dim)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch, 17472, hidden_dim)
        
        # 全局平均池化 (沿基因维度)
        x = x.transpose(1, 2)  # (batch, hidden_dim, 17472)
        x = self.global_pool(x)  # (batch, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch, hidden_dim)
        
        # 输出投影
        x = self.output_proj(x)  # (batch, output_dim)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, G, 3)
            
        Returns:
            编码后的特征，形状 (batch_size, 512)
        """
        if self.encoder_type == "cnn":
            return self.forward_cnn(x)
        elif self.encoder_type == "transformer":
            return self.forward_transformer(x)
        else:
            raise ValueError(f"不支持的编码器类型: {self.encoder_type}")


class PositionalEncoding(nn.Module):
    """
    位置编码模块 (用于Transformer)
    参考: "Attention Is All You Need"
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区 (不参与梯度更新)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            
        Returns:
            添加位置编码后的张量
        """
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def test_omics_encoder():
    """测试组学特征编码器"""
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模拟数据
    batch_size = 4
    G = 17472  # 基因数量
    C = 3      # 通道数量
    
    # 创建输入张量
    x = torch.randn(batch_size, G, C)
    print(f"输入形状: {x.shape}")
    print(f"期望: (batch_size={batch_size}, genes={G}, channels={C})")
    
    # 测试CNN编码器
    print("\n测试CNN编码器...")
    cnn_encoder = OmicsEncoder(encoder_type="cnn")
    cnn_output = cnn_encoder(x)
    print(f"CNN输出形状: {cnn_output.shape}")
    print(f"期望形状: ({batch_size}, 512)")
    
    # 测试不同配置的CNN编码器
    print("\n测试不同配置的CNN编码器...")
    cnn_encoder2 = OmicsEncoder(
        encoder_type="cnn",
        num_layers=4,
        kernel_size=5,
        dropout=0.2
    )
    cnn_output2 = cnn_encoder2(x)
    print(f"CNN2输出形状: {cnn_output2.shape}")
    
    # 测试Transformer编码器
    print("\n测试Transformer编码器...")
    transformer_encoder = OmicsEncoder(encoder_type="transformer")
    transformer_output = transformer_encoder(x)
    print(f"Transformer输出形状: {transformer_output.shape}")
    print(f"期望形状: ({batch_size}, 512)")
    
    # 测试不同配置的Transformer编码器
    print("\n测试不同配置的Transformer编码器...")
    transformer_encoder2 = OmicsEncoder(
        encoder_type="transformer",
        num_layers=4,
        num_heads=8,
        dropout=0.2
    )
    transformer_output2 = transformer_encoder2(x)
    print(f"Transformer2输出形状: {transformer_output2.shape}")
    
    # 验证输出维度
    assert cnn_output.shape == (batch_size, 512), f"CNN输出形状错误: {cnn_output.shape}"
    assert transformer_output.shape == (batch_size, 512), f"Transformer输出形状错误: {transformer_output.shape}"
    
    # 测试位置编码
    print("\n测试位置编码...")
    pos_enc = PositionalEncoding(d_model=512)
    test_input = torch.randn(2, 100, 512)
    output = pos_enc(test_input)
    print(f"位置编码输入形状: {test_input.shape}")
    print(f"位置编码输出形状: {output.shape}")
    
    print("\n所有测试通过!")


if __name__ == "__main__":
    test_omics_encoder()