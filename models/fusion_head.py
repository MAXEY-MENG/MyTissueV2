"""
融合与预测头
用于融合病理特征和组学特征，并进行多任务预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class FusionHead(nn.Module):
    """
    融合与预测头
    
    输入:
    - 病理特征: (batch_size, 512)
    - 组学特征: (batch_size, 512)
    
    输出:
    - 风险评分: (batch_size, 1) 用于Cox回归
    - 风险等级: (batch_size, 3) 用于分类 (低/中/高)
    """
    
    def __init__(
        self,
        input_dim: int = 512,  # 每个模态的特征维度
        hidden_dim: int = 256,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        num_classes: int = 3  # 风险等级数量
    ):
        """
        初始化融合头
        
        Args:
            input_dim: 输入特征维度 (每个模态)
            hidden_dim: 隐藏层维度
            fusion_dim: 融合特征维度
            dropout: Dropout概率
            num_classes: 风险等级数量
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
        # 融合层: 拼接两个模态的特征
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 回归头: 用于Cox风险评分
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            # 注意: 最后一层不使用激活函数，因为风险评分可以是任意实数
        )
        
        # 分类头: 用于风险等级分类
        self.classification_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        path_features: torch.Tensor,
        omics_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            path_features: 病理特征，形状 (batch_size, 512)
            omics_features: 组学特征，形状 (batch_size, 512)
            
        Returns:
            risk_score: 风险评分，形状 (batch_size, 1)
            risk_class: 风险等级logits，形状 (batch_size, num_classes)
        """
        # 拼接特征
        fused = torch.cat([path_features, omics_features], dim=1)  # (batch, 1024)
        
        # 融合层
        fused = self.fusion_layer(fused)  # (batch, fusion_dim)
        
        # 回归头: 风险评分
        risk_score = self.regression_head(fused)  # (batch, 1)
        
        # 分类头: 风险等级
        risk_class = self.classification_head(fused)  # (batch, num_classes)
        
        return risk_score, risk_class
    
    def predict_risk_level(self, risk_class_logits: torch.Tensor) -> torch.Tensor:
        """
        预测风险等级
        
        Args:
            risk_class_logits: 分类头输出的logits，形状 (batch, num_classes)
            
        Returns:
            风险等级预测 (0:低, 1:中, 2:高)，形状 (batch,)
        """
        return torch.argmax(risk_class_logits, dim=1)


class CoxLoss(nn.Module):
    """
    Cox比例风险模型损失函数 (负部分似然)
    
    参考: https://github.com/havakv/pycox
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        初始化Cox损失
        
        Args:
            reduction: 损失缩减方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        risk_score: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Cox损失
        
        Args:
            risk_score: 风险评分，形状 (batch_size, 1)
            time: 生存时间，形状 (batch_size,)
            event: 事件指示器 (1=死亡, 0=删失)，形状 (batch_size,)
            
        Returns:
            Cox损失值
        """
        # 确保输入形状正确: (batch_size, 1) -> (batch_size,)
        risk_score = risk_score.view(-1)  # 使用view而不是squeeze，避免batch_size=1时变成0维
        
        # 裁剪风险评分，防止exp溢出
        risk_score = torch.clamp(risk_score, min=-20, max=20)
        
        # 按时间降序排序
        sort_idx = torch.argsort(time, descending=True)
        risk_score = risk_score[sort_idx]
        time = time[sort_idx]
        event = event[sort_idx]
        
        # 计算风险集合
        batch_size = risk_score.shape[0]
        num_events = 0
        
        # 使用向量化方法计算Cox损失，避免Python循环中的梯度问题
        # 对于每个事件样本i，计算 log(sum(exp(risk_set))) - risk_score[i]
        # 其中 risk_set = risk_score[i:]
        
        # 找到所有事件样本的索引
        event_indices = torch.where(event == 1)[0]
        num_events = event_indices.shape[0]
        
        if num_events == 0:
            # 如果没有事件，返回一个带有梯度的零张量
            return torch.tensor(0.0, device=risk_score.device, requires_grad=True)
        
        # 向量化计算
        losses = []
        for i in event_indices:
            risk_set = risk_score[i:]
            log_sum_exp = torch.logsumexp(risk_set, dim=0)
            losses.append(log_sum_exp - risk_score[i])
        
        loss = torch.stack(losses).mean()
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * num_events
        else:  # 'none'
            return torch.full((batch_size,), loss / batch_size, device=risk_score.device)


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    结合Cox回归损失和分类损失
    """
    
    def __init__(
        self,
        cox_weight: float = 1.0,
        ce_weight: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        初始化多任务损失
        
        Args:
            cox_weight: Cox损失权重
            ce_weight: 交叉熵损失权重
            reduction: 损失缩减方式
        """
        super().__init__()
        self.cox_weight = cox_weight
        self.ce_weight = ce_weight
        self.cox_loss = CoxLoss(reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
        self,
        risk_score: torch.Tensor,
        risk_class: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
        risk_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            risk_score: 风险评分，形状 (batch_size, 1)
            risk_class: 风险等级logits，形状 (batch_size, num_classes)
            time: 生存时间，形状 (batch_size,)
            event: 事件指示器，形状 (batch_size,)
            risk_labels: 风险等级标签 (可选)，形状 (batch_size,)
            
        Returns:
            total_loss: 总损失
            cox_loss: Cox损失
            ce_loss: 交叉熵损失 (如果提供了标签)
        """
        # Cox损失
        cox_loss = self.cox_loss(risk_score, time, event)
        
        # 交叉熵损失 (如果提供了标签)
        if risk_labels is not None:
            ce_loss = self.ce_loss(risk_class, risk_labels)
            total_loss = self.cox_weight * cox_loss + self.ce_weight * ce_loss
            return total_loss, cox_loss, ce_loss
        else:
            # 如果没有标签，只使用Cox损失
            total_loss = self.cox_weight * cox_loss
            return total_loss, cox_loss, torch.tensor(0.0, device=risk_score.device)


def test_fusion_head():
    """测试融合与预测头"""
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模拟数据
    batch_size = 8
    input_dim = 512
    
    # 创建输入特征
    path_features = torch.randn(batch_size, input_dim)
    omics_features = torch.randn(batch_size, input_dim)
    
    print(f"病理特征形状: {path_features.shape}")
    print(f"组学特征形状: {omics_features.shape}")
    
    # 测试融合头
    print("\n测试融合头...")
    fusion_head = FusionHead()
    risk_score, risk_class = fusion_head(path_features, omics_features)
    
    print(f"风险评分形状: {risk_score.shape}")
    print(f"风险等级logits形状: {risk_class.shape}")
    
    # 测试风险等级预测
    risk_levels = fusion_head.predict_risk_level(risk_class)
    print(f"风险等级预测形状: {risk_levels.shape}")
    print(f"预测的风险等级: {risk_levels}")
    
    # 测试损失函数
    print("\n测试损失函数...")
    
    # 创建模拟生存数据
    time = torch.rand(batch_size) * 100  # 生存时间 0-100
    event = torch.randint(0, 2, (batch_size,))  # 事件指示器
    risk_labels = torch.randint(0, 3, (batch_size,))  # 风险等级标签
    
    print(f"生存时间形状: {time.shape}")
    print(f"事件指示器形状: {event.shape}")
    print(f"风险等级标签形状: {risk_labels.shape}")
    
    # 测试Cox损失
    cox_loss_fn = CoxLoss()
    cox_loss = cox_loss_fn(risk_score, time, event)
    print(f"Cox损失: {cox_loss.item():.4f}")
    
    # 测试多任务损失
    multi_loss_fn = MultiTaskLoss(cox_weight=1.0, ce_weight=0.5)
    total_loss, cox_loss_val, ce_loss_val = multi_loss_fn(
        risk_score, risk_class, time, event, risk_labels
    )
    
    print(f"总损失: {total_loss.item():.4f}")
    print(f"Cox损失: {cox_loss_val.item():.4f}")
    print(f"交叉熵损失: {ce_loss_val.item():.4f}")
    
    # 验证输出维度
    assert risk_score.shape == (batch_size, 1), f"风险评分形状错误: {risk_score.shape}"
    assert risk_class.shape == (batch_size, 3), f"风险等级形状错误: {risk_class.shape}"
    assert risk_levels.shape == (batch_size,), f"风险等级预测形状错误: {risk_levels.shape}"
    
    print("\n所有测试通过!")


if __name__ == "__main__":
    test_fusion_head()