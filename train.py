"""
训练脚本
用于训练多模态生存预测模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data.dataset import MultiModalDataset
from data.split import load_split_files
from models.model import MultiModalSurvivalModel


class Trainer:
    """
    训练器类
    管理模型训练、验证和测试
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "./output"
    ):
        """
        初始化训练器
        
        Args:
            config: 训练配置字典
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置设备
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"使用设备: {self.device}")
        
        # 设置随机种子
        self.seed = config.get('seed', 42)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        
        # 初始化模型
        self.model = self._create_model()
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_cindex = 0.0
        self.best_epoch = 0
        self.train_history = []
        self.val_history = []
        
        # 创建日志文件
        self.log_file = os.path.join(output_dir, "training_log.csv")
        self._init_log_file()
        
        # 保存配置
        self._save_config()
    
    def _create_model(self) -> MultiModalSurvivalModel:
        """创建模型"""
        model_config = {
            'path_aggregator_mode': self.config.get('path_aggregator_mode', 'mean'),
            'path_hidden_dim': self.config.get('path_hidden_dim', 512),
            'path_dropout': self.config.get('path_dropout', 0.1),
            'omics_encoder_type': self.config.get('omics_encoder_type', 'cnn'),
            'omics_hidden_dim': self.config.get('omics_hidden_dim', 512),
            'omics_num_layers': self.config.get('omics_num_layers', 3),
            'omics_dropout': self.config.get('omics_dropout', 0.1),
            'fusion_hidden_dim': self.config.get('fusion_hidden_dim', 256),
            'fusion_dim': self.config.get('fusion_dim', 128),
            'fusion_dropout': self.config.get('fusion_dropout', 0.1),
            'num_classes': self.config.get('num_classes', 3),
            'device': str(self.device)
        }
        
        model = MultiModalSurvivalModel(**model_config)
        print(f"模型创建完成")
        print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        learning_rate = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")
        
        print(f"使用优化器: {optimizer_type}, 学习率: {learning_rate}")
        return optimizer
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 100)
        
        if scheduler_type.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        elif scheduler_type.lower() == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # 监控C-index，越大越好
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif scheduler_type.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = None
        
        if scheduler:
            print(f"使用学习率调度器: {scheduler_type}")
        
        return scheduler
    
    def _init_log_file(self):
        """初始化日志文件"""
        if not os.path.exists(self.log_file):
            columns = [
                'epoch', 'train_loss', 'train_cox_loss', 'train_ce_loss',
                'val_loss', 'val_cox_loss', 'val_ce_loss', 'val_cindex',
                'learning_rate', 'time'
            ]
            pd.DataFrame(columns=columns).to_csv(self.log_file, index=False)
    
    def _save_config(self):
        """保存配置到文件"""
        config_file = os.path.join(self.output_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"配置已保存到: {config_file}")
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        # 数据路径配置
        data_config = self.config['data']
        
        # 加载划分结果
        split_dir = data_config.get('split_dir', './processed/splits')
        train_patients, val_patients, test_patients = load_split_files(split_dir)
        
        print(f"训练集: {len(train_patients)} 患者")
        print(f"验证集: {len(val_patients)} 患者")
        print(f"测试集: {len(test_patients)} 患者")
        
        # 创建数据集
        train_dataset = MultiModalDataset(
            tensor_path=data_config['tensor_path'],
            patients_path=data_config['patients_path'],
            path_feature_dir=data_config['path_feature_dir'],
            survival_path=data_config['survival_path'],
            patient_ids=train_patients
        )
        
        val_dataset = MultiModalDataset(
            tensor_path=data_config['tensor_path'],
            patients_path=data_config['patients_path'],
            path_feature_dir=data_config['path_feature_dir'],
            survival_path=data_config['survival_path'],
            patient_ids=val_patients
        )
        
        test_dataset = MultiModalDataset(
            tensor_path=data_config['tensor_path'],
            patients_path=data_config['patients_path'],
            path_feature_dir=data_config['path_feature_dir'],
            survival_path=data_config['survival_path'],
            patient_ids=test_patients
        )
        
        # 创建数据加载器
        batch_size = self.config.get('batch_size', 8)
        num_workers = self.config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=MultiModalDataset.collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=MultiModalDataset.collate_fn,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=MultiModalDataset.collate_fn,
            pin_memory=True
        )
        
        print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_cox_loss = 0.0
        total_ce_loss = 0.0
        num_batches = 0
        
        # 梯度累积步数（实际batch_size = batch_size * accumulation_steps）
        accumulation_steps = self.config.get('gradient_accumulation_steps', 4)
        self.optimizer.zero_grad()
        
        for batch_idx, (path_list, omics_batch, time_batch, event_batch, _) in enumerate(train_loader):
            # 组学特征移到GPU
            omics_batch = omics_batch.to(self.device)
            time_batch = time_batch.to(self.device)
            event_batch = event_batch.to(self.device)
            
            # 病理特征保持在CPU，逐个处理以节省显存
            # 将每个病理特征移到GPU并立即聚合
            path_features_batch = []
            for pf in path_list:
                # 病理特征在CPU上，逐个移到GPU处理
                pf_gpu = pf.to(self.device)
                # 如果patch数太多，进行下采样
                if pf_gpu.shape[0] > 5000:
                    # 随机采样5000个patch
                    idx = torch.randperm(pf_gpu.shape[0], device=self.device)[:5000]
                    pf_gpu = pf_gpu[idx]
                path_features_batch.append(pf_gpu)
            
            # 前向传播
            risk_score, risk_class = self.model(path_features_batch, omics_batch)
            
            # 计算损失
            total_loss_batch, cox_loss_batch, ce_loss_batch = self.model.compute_loss(
                risk_score, risk_class, time_batch, event_batch
            )
            
            # 梯度累积：缩放损失
            total_loss_batch = total_loss_batch / accumulation_steps
            total_loss_batch.backward()
            
            # 梯度累积：每 accumulation_steps 步更新一次参数
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # 更新参数
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 累加损失（恢复缩放前的值）
            total_loss += total_loss_batch.item() * accumulation_steps
            total_cox_loss += cox_loss_batch.item()
            total_ce_loss += ce_loss_batch.item()
            num_batches += 1
            
            # 清理GPU缓存
            del path_features_batch, pf_gpu, risk_score, risk_class
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"  批次 {batch_idx + 1}/{len(train_loader)}, "
                      f"损失: {total_loss_batch.item() * accumulation_steps:.4f}")
        
        # 处理剩余的梯度（如果 accumulation_steps 不能整除 num_batches）
        if num_batches % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_cox_loss = total_cox_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        
        return {
            'loss': avg_loss,
            'cox_loss': avg_cox_loss,
            'ce_loss': avg_ce_loss
        }
    
    def _process_path_features(self, path_list):
        """处理病理特征：移到GPU并下采样"""
        path_features_batch = []
        for pf in path_list:
            pf_gpu = pf.to(self.device)
            if pf_gpu.shape[0] > 5000:
                idx = torch.randperm(pf_gpu.shape[0], device=self.device)[:5000]
                pf_gpu = pf_gpu[idx]
            path_features_batch.append(pf_gpu)
        return path_features_batch
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_cox_loss = 0.0
        total_ce_loss = 0.0
        num_batches = 0
        
        # 收集预测结果用于计算C-index
        all_risk_scores = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for path_list, omics_batch, time_batch, event_batch, _ in val_loader:
                # 移动到设备
                omics_batch = omics_batch.to(self.device)
                time_batch = time_batch.to(self.device)
                event_batch = event_batch.to(self.device)
                
                # 处理病理特征（下采样以节省显存）
                path_features_batch = self._process_path_features(path_list)
                
                # 前向传播
                risk_score, risk_class = self.model(path_features_batch, omics_batch)
                
                # 计算损失
                total_loss_batch, cox_loss_batch, ce_loss_batch = self.model.compute_loss(
                    risk_score, risk_class, time_batch, event_batch
                )
                
                # 累加损失
                total_loss += total_loss_batch.item()
                total_cox_loss += cox_loss_batch.item()
                total_ce_loss += ce_loss_batch.item()
                num_batches += 1
                
                # 收集预测结果
                all_risk_scores.extend(risk_score.cpu().numpy().flatten())
                all_times.extend(time_batch.cpu().numpy())
                all_events.extend(event_batch.cpu().numpy())
                
                # 清理GPU缓存
                del path_features_batch, risk_score, risk_class
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_cox_loss = total_cox_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        
        # 计算C-index
        cindex = self._calculate_c_index(
            np.array(all_times),
            np.array(all_events),
            np.array(all_risk_scores)
        )
        
        return {
            'loss': avg_loss,
            'cox_loss': avg_cox_loss,
            'ce_loss': avg_ce_loss,
            'cindex': cindex
        }
    
    def _calculate_c_index(self, time: np.ndarray, event: np.ndarray, risk_score: np.ndarray) -> float:
        """
        计算C-index (一致性指数)
        
        Args:
            time: 生存时间数组
            event: 事件指示器数组 (1=事件发生, 0=删失)
            risk_score: 风险评分数组
            
        Returns:
            C-index值 (0-1之间，越高越好)
        """
        # 简单实现C-index计算
        # 更复杂的实现可以使用lifelines库
        
        n = len(time)
        if n < 2:
            return 0.5
        
        concordant = 0
        permissible = 0
        
        for i in range(n):
            if event[i] == 1:  # 只考虑事件发生的样本
                for j in range(n):
                    if time[j] > time[i]:  # j的生存时间更长
                        permissible += 1
                        if risk_score[i] > risk_score[j]:  # i的风险评分更高
                            concordant += 1
                    elif time[j] == time[i] and event[j] == 0:  # 相同时间，j删失
                        permissible += 1
                        if risk_score[i] > risk_score[j]:
                            concordant += 1
        
        if permissible == 0:
            return 0.5
        
        return concordant / permissible
    
    def _save_log(self, epoch: int, train_metrics: Dict[str, float], 
                  val_metrics: Dict[str, float], lr: float, elapsed_time: float):
        """保存训练日志"""
        log_entry = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_cox_loss': train_metrics['cox_loss'],
            'train_ce_loss': train_metrics['ce_loss'],
            'val_loss': val_metrics['loss'],
            'val_cox_loss': val_metrics['cox_loss'],
            'val_ce_loss': val_metrics['ce_loss'],
            'val_cindex': val_metrics['cindex'],
            'learning_rate': lr,
            'time': elapsed_time
        }
        
        # 追加到CSV文件
        df = pd.DataFrame([log_entry])
        df.to_csv(self.log_file, mode='a', header=False, index=False)
    
    def _save_best_model(self):
        """保存最佳模型"""
        model_path = os.path.join(self.output_dir, "best_model.pth")
        self.model.save(model_path)
    
    def _load_best_model(self):
        """加载最佳模型"""
        model_path = os.path.join(self.output_dir, "best_model.pth")
        if os.path.exists(model_path):
            self.model = MultiModalSurvivalModel.load(model_path, device=str(self.device))
            print(f"已加载最佳模型 (epoch {self.best_epoch}, C-index: {self.best_val_cindex:.4f})")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """在测试集上评估模型"""
        self.model.eval()
        
        # 收集预测结果
        all_risk_scores = []
        all_risk_levels = []
        all_times = []
        all_events = []
        all_patient_ids = []
        
        with torch.no_grad():
            for path_list, omics_batch, time_batch, event_batch, patient_ids in test_loader:
                # 移动到设备
                omics_batch = omics_batch.to(self.device)
                
                # 处理病理特征（下采样以节省显存）
                path_features_batch = self._process_path_features(path_list)
                
                # 预测
                predictions = self.model.predict(path_features_batch, omics_batch)
                
                # 收集结果
                all_risk_scores.extend(predictions['risk_score'].cpu().numpy().flatten())
                all_risk_levels.extend(predictions['risk_level'].cpu().numpy())
                all_times.extend(time_batch.cpu().numpy())
                all_events.extend(event_batch.cpu().numpy())
                all_patient_ids.extend(patient_ids)
        
        # 计算C-index
        cindex = self._calculate_c_index(
            np.array(all_times),
            np.array(all_events),
            np.array(all_risk_scores)
        )
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'patient_id': all_patient_ids,
            'risk_score': all_risk_scores,
            'risk_level': all_risk_levels,
            'time': all_times,
            'event': all_events
        })
        
        results_path = os.path.join(self.output_dir, "test_predictions.csv")
        results_df.to_csv(results_path, index=False)
        
        print(f"测试集C-index: {cindex:.4f}")
        print(f"预测结果已保存到: {results_path}")
        
        return {
            'cindex': cindex,
            'predictions_path': results_path,
            'num_samples': len(all_patient_ids)
        }
    
    def _save_final_results(self, test_metrics: Dict[str, Any]):
        """保存最终结果"""
        # 创建结果摘要
        results_summary = {
            'best_epoch': self.best_epoch,
            'best_val_cindex': self.best_val_cindex,
            'test_cindex': test_metrics['cindex'],
            'num_test_samples': test_metrics['num_samples'],
            'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': self.config
        }
        
        # 保存结果摘要
        summary_path = os.path.join(self.output_dir, "results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"结果摘要已保存到: {summary_path}")
    
    def train(self, epochs: int = None, patience: int = None):
        """训练模型"""
        if epochs is None:
            epochs = self.config.get('epochs', 100)
        if patience is None:
            patience = self.config.get('patience', 20)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self._create_dataloaders()

        # ========== 数据调试：检查第一个训练批次 ==========
        print("\n" + "="*50)
        print("数据检查：第一个训练批次")
        try:
            # 获取一个batch
            first_batch = next(iter(train_loader))
            path_list, omics_batch, time_batch, event_batch, patient_ids = first_batch

            print(f"Batch大小: {len(path_list)}")
            print(f"omic特征形状: {omics_batch.shape}")
            print(f"特征是否有NaN: {torch.isnan(omics_batch).any()}")
            print(f"特征是否有Inf: {torch.isinf(omics_batch).any()}")
            print(f"特征值范围: [{omics_batch.min().item():.4f}, {omics_batch.max().item():.4f}]")

            print(f"时间: min={time_batch.min().item():.4f}, max={time_batch.max().item():.4f}")
            print(f"时间是否有NaN: {torch.isnan(time_batch).any()}")
            print(f"时间<=0的数量: {(time_batch <= 0).sum().item()}")

            print(f"事件唯一值: {torch.unique(event_batch)}")
            print(f"事件1的数量: {(event_batch == 1).sum().item()}, 事件0的数量: {(event_batch == 0).sum().item()}")

            # 检查病理特征
            path_shapes = [p.shape for p in path_list]
            print(f"病理特征形状: {path_shapes}")
            path_has_nan = any(torch.isnan(p).any() for p in path_list)
            path_has_inf = any(torch.isinf(p).any() for p in path_list)
            print(f"病理特征是否有NaN: {path_has_nan}, Inf: {path_has_inf}")

            # 小前传测试（关闭梯度，不影响训练）
            self.model.eval()
            with torch.no_grad():
                # 只需要omic部分和简化的病理特征（避免OOM）
                omics_batch_test = omics_batch.to(self.device)
                path_batch_test = self._process_path_features(path_list)
                risk_score, risk_class = self.model(path_batch_test, omics_batch_test)
                print(f"风险评分范围: [{risk_score.min().item():.4f}, {risk_score.max().item():.4f}]")
                print(f"风险评分是否有NaN: {torch.isnan(risk_score).any()}")
                print(f"风险类别分布: {torch.bincount(risk_class)}")
                # 清理
                del path_batch_test, risk_score, risk_class
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.model.train()
        except Exception as e:
            print(f"数据检查出错: {e}")
        print("="*50 + "\n")
        # ========== 数据检查结束 ==========

        print(f"\n开始训练，共 {epochs} 个epoch")
        print(f"早停耐心值: {patience}")
        
        # 早停计数器
        early_stop_counter = 0
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            
            # 训练
            print(f"\nEpoch {epoch}/{epochs}")
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['cindex'])
                else:
                    self.scheduler.step()
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # 保存日志
            self._save_log(epoch, train_metrics, val_metrics, current_lr, time.time() - start_time)
            
            # 打印结果
            print(f"  训练损失: {train_metrics['loss']:.4f} "
                  f"(Cox: {train_metrics['cox_loss']:.4f}, CE: {train_metrics['ce_loss']:.4f})")
            print(f"  验证损失: {val_metrics['loss']:.4f} "
                  f"(Cox: {val_metrics['cox_loss']:.4f}, CE: {val_metrics['ce_loss']:.4f})")
            print(f"  验证C-index: {val_metrics['cindex']:.4f}")
            print(f"  学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_metrics['cindex'] > self.best_val_cindex:
                self.best_val_cindex = val_metrics['cindex']
                self.best_epoch = epoch
                self._save_best_model()
                early_stop_counter = 0
                print(f"  ✓ 最佳模型已保存 (C-index: {val_metrics['cindex']:.4f})")
            else:
                early_stop_counter += 1
                print(f"  早停计数器: {early_stop_counter}/{patience}")
            
            # 早停检查
            if early_stop_counter >= patience:
                print(f"\n早停触发! 在epoch {epoch}停止训练")
                print(f"最佳验证C-index: {self.best_val_cindex:.4f} (epoch {self.best_epoch})")
                break
        
        # 训练完成
        print(f"\n训练完成!")
        print(f"最佳验证C-index: {self.best_val_cindex:.4f} (epoch {self.best_epoch})")
        
        # 加载最佳模型
        self._load_best_model()
        
        # 在测试集上评估
        print("\n在测试集上评估...")
        test_metrics = self.evaluate(test_loader)
        
        # 保存最终结果
        self._save_final_results(test_metrics)


def main():
    """主函数"""
    # 训练配置
    config = {
        # 数据配置
        'data': {
            'tensor_path': './processed/tensor.npy',
            'patients_path': './processed/patients_recommended.txt',
            'path_feature_dir': './path_features/',
            'survival_path': './data/survival.csv',
            'split_dir': './processed/splits'
        },
        
        # 模型配置
        'path_aggregator_mode': 'mean',  # 'mean' 或 'attention'
        'path_hidden_dim': 512,
        'path_dropout': 0.1,
        'omics_encoder_type': 'cnn',  # 'cnn' 或 'transformer'
        'omics_hidden_dim': 512,
        'omics_num_layers': 3,
        'omics_dropout': 0.1,
        'fusion_hidden_dim': 256,
        'fusion_dim': 128,
        'fusion_dropout': 0.1,
        'num_classes': 3,
        
        # 训练配置
        'batch_size': 2,
        'gradient_accumulation_steps': 4,  # 梯度累积步数，有效batch_size = 2 * 4 = 8
        'epochs': 50,
        'patience': 10,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'num_workers': 0,  # Windows下多进程加载可能有问题，设为0
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./output/training_{timestamp}"
    
    # 创建训练器
    trainer = Trainer(config, output_dir)
    
    # 开始训练
    trainer.train()
    
    print(f"\n训练完成! 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
        
