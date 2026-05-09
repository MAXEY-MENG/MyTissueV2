"""
评估与可视化脚本
用于加载训练好的模型并在测试集上进行评估，生成可视化结果
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========== 添加中文字体设置，解决中文显示方框问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DengXian', 'SimSun']  # Windows 中文字体
plt.rcParams['axes.unicode_minus'] = False          # 正常显示负号

# 导入自定义模块
from data.dataset import MultiModalDataset
from data.split import load_split_files
from models.model import MultiModalSurvivalModel
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


class Evaluator:
    """
    评估器类
    用于模型评估和可视化
    """

    def __init__(
        self,
        model_path: str,
        data_config: Dict[str, str],
        output_dir: str = "./output/evaluation"
    ):
        """
        初始化评估器

        Args:
            model_path: 模型文件路径
            data_config: 数据配置字典
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.data_config = data_config
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载模型
        self.model = self._load_model()

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建数据加载器
        self.test_loader = self._create_test_dataloader()

    def _load_model(self) -> MultiModalSurvivalModel:
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        model = MultiModalSurvivalModel.load(self.model_path)
        model.eval()
        print("模型加载完成")
        return model

    def _create_test_dataloader(self):
        """创建测试集数据加载器"""
        # 加载划分结果
        split_dir = self.data_config.get('split_dir', './processed/splits')
        _, _, test_patients = load_split_files(split_dir)

        print(f"测试集: {len(test_patients)} 患者")

        # 创建测试数据集
        test_dataset = MultiModalDataset(
            tensor_path=self.data_config['tensor_path'],
            patients_path=self.data_config['patients_path'],
            path_feature_dir=self.data_config['path_feature_dir'],
            survival_path=self.data_config['survival_path'],
            patient_ids=test_patients
        )

        # 创建数据加载器
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            collate_fn=MultiModalDataset.collate_fn,
            pin_memory=True
        )

        print(f"测试批次: {len(test_loader)}")
        return test_loader

    def evaluate(self) -> Dict[str, Any]:
        """执行完整评估"""
        print("\n开始评估...")

        # 收集预测结果
        all_predictions = self._collect_predictions()

        # 计算C-index
        cindex = self._calculate_c_index(
            all_predictions['times'],
            all_predictions['events'],
            all_predictions['risk_scores']
        )

        print(f"测试集C-index: {cindex:.4f}")

        # 计算分类指标
        if 'risk_labels' in all_predictions:
            classification_metrics = self._calculate_classification_metrics(
                all_predictions['risk_levels'],
                all_predictions['risk_labels']
            )
        else:
            classification_metrics = None

        # 保存预测结果
        predictions_df = pd.DataFrame({
            'patient_id': all_predictions['patient_ids'],
            'risk_score': all_predictions['risk_scores'],
            'risk_level': all_predictions['risk_levels'],
            'time': all_predictions['times'],
            'event': all_predictions['events']
        })

        predictions_path = os.path.join(self.output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"预测结果已保存到: {predictions_path}")

        # 生成可视化
        self._generate_visualizations(all_predictions)

        # 创建评估摘要
        evaluation_summary = {
            'cindex': float(cindex),
            'num_samples': len(all_predictions['patient_ids']),
            'predictions_path': predictions_path,
            'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if classification_metrics:
            evaluation_summary['classification_metrics'] = classification_metrics

        # 保存评估摘要
        summary_path = os.path.join(self.output_dir, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)

        print(f"评估摘要已保存到: {summary_path}")

        return evaluation_summary

    def _collect_predictions(self) -> Dict[str, List]:
        """收集预测结果（已修复设备不一致问题）"""
        self.model.eval()
        all_risk_scores, all_risk_levels = [], []
        all_times, all_events, all_patient_ids = [], [], []

        with torch.no_grad():
            for path_list, omics_batch, time_batch, event_batch, patient_ids in self.test_loader:
                # 移动组学数据到设备
                omics_batch = omics_batch.to(self.device)

                # 将病理特征列表中的每个张量移到设备（修复设备不一致）
                path_list = [p.to(self.device) if isinstance(p, torch.Tensor) else p
                             for p in path_list]

                # 预测
                predictions = self.model.predict(path_list, omics_batch)

                # 收集结果
                all_risk_scores.extend(predictions['risk_score'].cpu().numpy().flatten())
                all_risk_levels.extend(predictions['risk_level'].cpu().numpy())
                all_times.extend(time_batch.cpu().numpy())
                all_events.extend(event_batch.cpu().numpy())
                all_patient_ids.extend(patient_ids)

        return {
            'patient_ids': all_patient_ids,
            'risk_scores': np.array(all_risk_scores),
            'risk_levels': np.array(all_risk_levels),
            'times': np.array(all_times),
            'events': np.array(all_events)
        }

    def _calculate_c_index(self, time: np.ndarray, event: np.ndarray, risk_score: np.ndarray) -> float:
        """计算C-index"""
        n = len(time)
        if n < 2:
            return 0.5

        concordant = 0
        permissible = 0

        for i in range(n):
            if event[i] == 1:
                for j in range(n):
                    if time[j] > time[i]:
                        permissible += 1
                        if risk_score[i] > risk_score[j]:
                            concordant += 1
                    elif time[j] == time[i] and event[j] == 0:
                        permissible += 1
                        if risk_score[i] > risk_score[j]:
                            concordant += 1

        if permissible == 0:
            return 0.5

        return concordant / permissible

    def _calculate_classification_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        cm = confusion_matrix(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True)
        return {
            'confusion_matrix': cm.tolist(),
            'accuracy': float(report['accuracy']),
            'macro_avg': report['macro avg'],
            'weighted_avg': report['weighted avg']
        }

    def _generate_visualizations(self, predictions: Dict[str, np.ndarray]):
        """生成可视化图表"""
        print("\n生成可视化图表...")

        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        self._plot_kaplan_meier(predictions)
        self._plot_risk_score_distribution(predictions)
        self._plot_risk_level_distribution(predictions)
        self._plot_risk_vs_survival(predictions)

        print(f"可视化图表已保存到: {self.output_dir}")

    def _plot_kaplan_meier(self, predictions: Dict[str, np.ndarray]):
        times = predictions['times']
        events = predictions['events']
        risk_scores = predictions['risk_scores']

        median_risk = np.median(risk_scores)
        high_risk = risk_scores > median_risk
        low_risk = risk_scores <= median_risk

        kmf = KaplanMeierFitter()

        plt.figure(figsize=(10, 6))

        kmf.fit(
            times[high_risk],
            event_observed=events[high_risk],
            label=f'高风险组 (n={np.sum(high_risk)})'
        )
        kmf.plot_survival_function(ci_show=True)

        kmf.fit(
            times[low_risk],
            event_observed=events[low_risk],
            label=f'低风险组 (n={np.sum(low_risk)})'
        )
        kmf.plot_survival_function(ci_show=True)

        plt.title('Kaplan-Meier生存曲线 (按风险评分中位数分组)', fontsize=14, fontweight='bold')
        plt.xlabel('生存时间 (天)', fontsize=12)
        plt.ylabel('生存概率', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        result = logrank_test(
            times[high_risk], times[low_risk],
            event_observed_A=events[high_risk], event_observed_B=events[low_risk]
        )
        plt.text(0.02, 0.02, f'Log-rank检验 p值: {result.p_value:.4f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        km_path = os.path.join(self.output_dir, "kaplan_meier_curve.png")
        plt.savefig(km_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Kaplan-Meier曲线已保存: {km_path}")

    def _plot_risk_score_distribution(self, predictions: Dict[str, np.ndarray]):
        risk_scores = predictions['risk_scores']
        events = predictions['events']

        # 直接创建子图，不需要先创建空白 figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(risk_scores, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(x=np.median(risk_scores), color='red', linestyle='--',
                        label=f'中位数: {np.median(risk_scores):.2f}')
        axes[0].set_title('风险评分分布', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('风险评分', fontsize=11)
        axes[0].set_ylabel('频数', fontsize=11)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for event_val, color, label in [(1, 'coral', '事件发生'), (0, 'lightgreen', '删失')]:
            mask = events == event_val
            axes[1].hist(risk_scores[mask], bins=20, alpha=0.6, color=color,
                         label=f'{label} (n={np.sum(mask)})', edgecolor='black')

        axes[1].set_title('风险评分分布 (按事件状态)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('风险评分', fontsize=11)
        axes[1].set_ylabel('频数', fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        dist_path = os.path.join(self.output_dir, "risk_score_distribution.png")
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  风险评分分布图已保存: {dist_path}")

    def _plot_risk_level_distribution(self, predictions: Dict[str, np.ndarray]):
        risk_levels = predictions['risk_levels']

        plt.figure(figsize=(8, 6))

        unique_levels, counts = np.unique(risk_levels, return_counts=True)
        level_names = ['低风险', '中风险', '高风险']

        bars = plt.bar(level_names, counts, color=['lightgreen', 'gold', 'lightcoral'],
                       edgecolor='black', alpha=0.8)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{count}', ha='center', va='bottom', fontsize=11)

        plt.title('风险等级分布', fontsize=14, fontweight='bold')
        plt.xlabel('风险等级', fontsize=12)
        plt.ylabel('患者数量', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        level_path = os.path.join(self.output_dir, "risk_level_distribution.png")
        plt.savefig(level_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  风险等级分布图已保存: {level_path}")

    def _plot_risk_vs_survival(self, predictions: Dict[str, np.ndarray]):
        times = predictions['times']
        risk_scores = predictions['risk_scores']
        events = predictions['events']

        plt.figure(figsize=(10, 6))

        scatter = plt.scatter(times, risk_scores, c=events, cmap='coolwarm',
                              alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

        cbar = plt.colorbar(scatter)
        cbar.set_label('事件状态 (0=删失, 1=事件发生)', fontsize=11)
        cbar.set_ticks([0, 1])

        z = np.polyfit(times, risk_scores, 1)
        p = np.poly1d(z)
        plt.plot(times, p(times), "r--", alpha=0.8, linewidth=2,
                 label=f'趋势线: y = {z[0]:.3f}x + {z[1]:.3f}')

        plt.title('风险评分与生存时间的关系', fontsize=14, fontweight='bold')
        plt.xlabel('生存时间 (天)', fontsize=12)
        plt.ylabel('风险评分', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        correlation = np.corrcoef(times, risk_scores)[0, 1]
        plt.text(0.02, 0.98, f'相关系数: {correlation:.3f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 verticalalignment='top')

        plt.tight_layout()
        relation_path = os.path.join(self.output_dir, "risk_vs_survival.png")
        plt.savefig(relation_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  风险评分与生存时间关系图已保存: {relation_path}")


def main():
    """主函数"""
    data_config = {
        'tensor_path': './processed/tensor.npy',
        'patients_path': './processed/patients_recommended.txt',
        'path_feature_dir': './path_features/',
        'survival_path': './data/survival.csv',
        'split_dir': './processed/splits'
    }

    model_path = "./output/training_20260509_155958/best_model.pth"

    if not os.path.exists(model_path):
        output_dir = "./output"
        if os.path.exists(output_dir):
            training_dirs = [d for d in os.listdir(output_dir)
                             if d.startswith('training_') and os.path.isdir(os.path.join(output_dir, d))]
            if training_dirs:
                training_dirs.sort(reverse=True)
                latest_dir = training_dirs[0]
                model_path = os.path.join(output_dir, latest_dir, "best_model.pth")
                if os.path.exists(model_path):
                    print(f"使用最新训练目录中的模型: {model_path}")
                else:
                    print(f"错误: 在 {latest_dir} 中未找到模型文件")
                    print("请先运行 train.py 训练模型，或手动指定模型路径")
                    return
            else:
                print("错误: 未找到训练目录")
                print("请先运行 train.py 训练模型")
                return
        else:
            print("错误: 输出目录不存在")
            print("请先运行 train.py 训练模型")
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./output/evaluation_{timestamp}"

    evaluator = Evaluator(model_path, data_config, output_dir)
    results = evaluator.evaluate()

    print(f"\n评估完成! 结果保存在: {output_dir}")
    print(f"测试集C-index: {results['cindex']:.4f}")
    print(f"评估样本数: {results['num_samples']}")


if __name__ == "__main__":
    main()