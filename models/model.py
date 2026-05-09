"""
完整的多模态生存预测模型
整合病理特征聚合器、组学特征编码器和融合预测头
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any

from .aggregator import PathologyAggregator
from .omic_encoder import OmicsEncoder
from .fusion_head import FusionHead, MultiTaskLoss


class MultiModalSurvivalModel(nn.Module):
    """
    多模态生存预测模型
    
    完整流程:
    1. 病理特征聚合: (N, 1024) -> (512)
    2. 组学特征编码: (17472, 3) -> (512)
    3. 特征融合与预测: (512 + 512) -> 风险评分 + 风险等级
    """
    
    def __init__(
        self,
        # 病理聚合器配置
        path_aggregator_mode: str = "mean",
        path_hidden_dim: int = 512,
        path_dropout: float = 0.1,
        # 组学编码器配置
        omics_encoder_type: str = "cnn",
        omics_hidden_dim: int = 512,
        omics_num_layers: int = 3,
        omics_dropout: float = 0.1,
        # 融合头配置
        fusion_hidden_dim: int = 256,
        fusion_dim: int = 128,
        fusion_dropout: float = 0.1,
        num_classes: int = 3,
        # 设备
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化多模态生存预测模型
        
        Args:
            path_aggregator_mode: 病理聚合模式 ("mean" 或 "attention")
            path_hidden_dim: 病理聚合器隐藏维度
            path_dropout: 病理聚合器dropout概率
            omics_encoder_type: 组学编码器类型 ("cnn" 或 "transformer")
            omics_hidden_dim: 组学编码器隐藏维度
            omics_num_layers: 组学编码器层数
            omics_dropout: 组学编码器dropout概率
            fusion_hidden_dim: 融合头隐藏维度
            fusion_dim: 融合特征维度
            fusion_dropout: 融合头dropout概率
            num_classes: 风险等级数量
            device: 设备 ("cuda" 或 "cpu")
        """
        super().__init__()
        
        self.device = device
        
        # 病理特征聚合器
        self.path_aggregator = PathologyAggregator(
            input_dim=1024,
            hidden_dim=path_hidden_dim,
            output_dim=512,
            aggregation_mode=path_aggregator_mode,
            dropout=path_dropout
        )
        
        # 组学特征编码器
        self.omics_encoder = OmicsEncoder(
            input_genes=17472,
            input_channels=3,
            hidden_dim=omics_hidden_dim,
            output_dim=512,
            encoder_type=omics_encoder_type,
            num_layers=omics_num_layers,
            dropout=omics_dropout
        )
        
        # 融合与预测头
        self.fusion_head = FusionHead(
            input_dim=512,
            hidden_dim=fusion_hidden_dim,
            fusion_dim=fusion_dim,
            dropout=fusion_dropout,
            num_classes=num_classes
        )
        
        # 损失函数
        self.loss_fn = MultiTaskLoss(cox_weight=1.0, ce_weight=0.5)
        
        # 移动到设备
        self.to(self.device)
    
    def forward(
        self,
        path_features_list: List[torch.Tensor],
        omics_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            path_features_list: 病理特征列表，每个元素形状 (N_i, 1024)
            omics_features: 组学特征，形状 (batch_size, 17472, 3)
            
        Returns:
            risk_score: 风险评分，形状 (batch_size, 1)
            risk_class: 风险等级logits，形状 (batch_size, num_classes)
        """
        # 病理特征聚合（病理特征已在外部移到GPU）
        path_features = self.path_aggregator(path_features_list)  # (batch, 512)
        
        # 组学特征编码
        omics_encoded = self.omics_encoder(omics_features)  # (batch, 512)
        
        # 特征融合与预测
        risk_score, risk_class = self.fusion_head(path_features, omics_encoded)
        
        return risk_score, risk_class
    
    def compute_loss(
        self,
        risk_score: torch.Tensor,
        risk_class: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
        risk_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算损失
        
        Args:
            risk_score: 风险评分，形状 (batch_size, 1)
            risk_class: 风险等级logits，形状 (batch_size, num_classes)
            time: 生存时间，形状 (batch_size,)
            event: 事件指示器，形状 (batch_size,)
            risk_labels: 风险等级标签 (可选)，形状 (batch_size,)
            
        Returns:
            total_loss: 总损失
            cox_loss: Cox损失
            ce_loss: 交叉熵损失
        """
        # 确保数据在正确的设备上
        time = time.to(self.device)
        event = event.to(self.device)
        if risk_labels is not None:
            risk_labels = risk_labels.to(self.device)
        
        return self.loss_fn(risk_score, risk_class, time, event, risk_labels)
    
    def predict(
        self,
        path_features_list: List[torch.Tensor],
        omics_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        预测
        
        Args:
            path_features_list: 病理特征列表
            omics_features: 组学特征
            
        Returns:
            包含预测结果的字典:
            - risk_score: 风险评分
            - risk_class_logits: 风险等级logits
            - risk_level: 风险等级预测 (0:低, 1:中, 2:高)
        """
        self.eval()
        with torch.no_grad():
            risk_score, risk_class = self.forward(path_features_list, omics_features)
            risk_level = self.fusion_head.predict_risk_level(risk_class)
        
        return {
            'risk_score': risk_score,
            'risk_class_logits': risk_class,
            'risk_level': risk_level
        }
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'path_aggregator_mode': self.path_aggregator.aggregation_mode,
            'path_hidden_dim': self.path_aggregator.hidden_dim,
            'path_dropout': self.path_aggregator.dropout,
            'omics_encoder_type': self.omics_encoder.encoder_type,
            'omics_hidden_dim': self.omics_encoder.hidden_dim,
            'omics_num_layers': self.omics_encoder.num_layers,
            'omics_dropout': 0.1,  # 从编码器配置中获取
            'fusion_hidden_dim': self.fusion_head.hidden_dim,
            'fusion_dim': self.fusion_head.fusion_dim,
            'fusion_dropout': 0.1,  # 从融合头配置中获取
            'num_classes': self.fusion_head.num_classes,
            'device': self.device
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)
        print(f"模型已保存到: {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        # 如果提供了设备参数，覆盖保存的配置
        if device is not None:
            config['device'] = device
        
        # 创建模型实例
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config['device'])
        
        print(f"模型已从 {path} 加载")
        return model


def test_model():
    """测试完整模型"""
    import numpy as np
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模拟数据
    batch_size = 4
    
    # 病理特征列表 (变长)
    path_features_list = []
    for i in range(batch_size):
        N = np.random.randint(10, 100)  # 随机patch数
        features = torch.randn(N, 1024)
        path_features_list.append(features)
    
    # 组学特征
    omics_features = torch.randn(batch_size, 17472, 3)
    
    # 生存数据
    time = torch.rand(batch_size) * 100
    event = torch.randint(0, 2, (batch_size,))
    risk_labels = torch.randint(0, 3, (batch_size,))
    
    print(f"病理特征列表长度: {len(path_features_list)}")
    print(f"组学特征形状: {omics_features.shape}")
    print(f"生存时间形状: {time.shape}")
    print(f"事件指示器形状: {event.shape}")
    print(f"风险等级标签形状: {risk_labels.shape}")
    
    # 测试不同配置的模型
    print("\n测试平均池化 + CNN模型...")
    model1 = MultiModalSurvivalModel(
        path_aggregator_mode="mean",
        omics_encoder_type="cnn"
    )
    
    # 前向传播
    risk_score1, risk_class1 = model1(path_features_list, omics_features)
    print(f"风险评分形状: {risk_score1.shape}")
    print(f"风险等级logits形状: {risk_class1.shape}")
    
    # 计算损失
    total_loss1, cox_loss1, ce_loss1 = model1.compute_loss(
        risk_score1, risk_class1, time, event, risk_labels
    )
    print(f"总损失: {total_loss1.item():.4f}")
    print(f"Cox损失: {cox_loss1.item():.4f}")
    print(f"交叉熵损失: {ce_loss1.item():.4f}")
    
    # 测试预测
    predictions1 = model1.predict(path_features_list, omics_features)
    print(f"风险等级预测: {predictions1['risk_level']}")
    
    print("\n测试注意力 + Transformer模型...")
    model2 = MultiModalSurvivalModel(
        path_aggregator_mode="attention",
        omics_encoder_type="transformer"
    )
    
    risk_score2, risk_class2 = model2(path_features_list, omics_features)
    print(f"风险评分形状: {risk_score2.shape}")
    print(f"风险等级logits形状: {risk_class2.shape}")
    
    # 测试保存和加载
    print("\n测试模型保存和加载...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pth")
        model1.save(save_path)
        
        # 加载模型
        loaded_model = MultiModalSurvivalModel.load(save_path)
        
        # 验证加载的模型与原始模型输出一致
        loaded_risk_score, loaded_risk_class = loaded_model(path_features_list, omics_features)
        
        assert torch.allclose(risk_score1, loaded_risk_score, rtol=1e-4), "加载的模型输出不一致"
        assert torch.allclose(risk_class1, loaded_risk_class, rtol=1e-4), "加载的模型输出不一致"
        
        print("模型保存和加载测试通过!")
    
    # 验证输出维度
    assert risk_score1.shape == (batch_size, 1), f"风险评分形状错误: {risk_score1.shape}"
    assert risk_class1.shape == (batch_size, 3), f"风险等级形状错误: {risk_class1.shape}"
    assert risk_score2.shape == (batch_size, 1), f"风险评分形状错误: {risk_score2.shape}"
    assert risk_class2.shape == (batch_size, 3), f"风险等级形状错误: {risk_class2.shape}"
    
    print("\n所有测试通过!")


if __name__ == "__main__":
    test_model()