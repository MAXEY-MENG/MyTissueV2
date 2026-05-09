# 多模态生存预测模型

基于PyTorch的多模态深度学习模型，用于整合病理特征和组学数据进行生存预测。

## 项目结构

```
D:/MyTissue/v2/
├── data/                    # 数据相关模块
│   ├── dataset.py          # 多模态数据集类
│   └── split.py            # 数据划分模块
├── models/                  # 模型定义
│   ├── aggregator.py       # 病理特征聚合器
│   ├── omic_encoder.py     # 组学特征编码器
│   ├── fusion_head.py      # 融合与预测头
│   └── model.py            # 完整模型
├── train.py                # 训练脚本
├── evaluate.py             # 评估与可视化脚本
├── requirements.txt        # 依赖包列表
└── README.md               # 项目说明
```

## 数据要求

### 1. 组学数据
- `tensor.npy`: 形状为 `(17472, 184, 3)` 的三维张量
  - 第一维: 17472个基因
  - 第二维: 184个患者
  - 第三维: 3个通道 (RNA-seq TPM, CNV, Methylation)
- `genes.txt`: 17472个基因名称列表
- `patients.txt`: 184个患者ID列表

### 2. 病理特征
- 目录: `./path_features/`
- 文件格式: `{patient_id}.pt`
- 每个文件包含形状为 `(N, 1024)` 的张量，其中N为patch数量

### 3. 生存数据
- 文件: `./data/survival.csv`
- 格式: CSV文件，包含以下列:
  - `patient_id`: 患者ID
  - `time`: 生存时间
  - `event`: 事件指示器 (1=死亡, 0=删失)

## 安装依赖

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖包
pip install -r requirements.txt

# 如果需要GPU支持，安装对应版本的PyTorch
# 参考: https://pytorch.org/get-started/locally/
```

## 快速开始

### 1. 数据准备
确保数据文件位于正确路径:
```
./processed/tensor.npy
./processed/patients.txt
./path_features/{patient_id}.pt
./data/survival.csv
```

### 2. 数据划分
```python
# 创建数据划分
from data.split import create_splits_from_patient_file

create_splits_from_patient_file(
    patients_file='./processed/patients.txt',
    output_dir='./processed/splits',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
```

### 3. 训练模型
```bash
# 使用默认配置训练
python train.py

# 训练完成后，结果将保存在:
# ./output/training_YYYYMMDD_HHMMSS/
```

### 4. 评估模型
```bash
# 评估最新训练的模型
python evaluate.py

# 评估指定模型
python evaluate.py --model_path ./output/training_20260509_155958/best_model.pth
```

## 模型架构

### 1. 病理特征聚合器 (`PathologyAggregator`)
- 输入: 变长的病理特征列表 `(N_i, 1024)`
- 输出: 固定长度的特征向量 `(batch_size, 512)`
- 支持两种聚合模式:
  - **平均池化**: 简单高效
  - **注意力聚合**: 类似ABMIL，学习不同patch的重要性权重

### 2. 组学特征编码器 (`OmicsEncoder`)
- 输入: 组学特征张量 `(batch_size, 17472, 3)`
- 输出: 编码后的特征 `(batch_size, 512)`
- 支持两种编码架构:
  - **1D CNN**: 沿基因方向使用卷积层
  - **Transformer**: 使用Transformer编码器

### 3. 融合与预测头 (`FusionHead`)
- 输入: 病理特征 `(512)` + 组学特征 `(512)`
- 输出:
  - 风险评分: `(batch_size, 1)` 用于Cox回归
  - 风险等级: `(batch_size, 3)` 用于分类 (低/中/高)
- 多任务损失: Cox损失 + 交叉熵损失

## 配置选项

### 训练配置 (`train.py` 中的config字典)
```python
config = {
    # 数据配置
    'data': {
        'tensor_path': './processed/tensor.npy',
        'patients_path': './processed/patients.txt',
        'path_feature_dir': './path_features/',
        'survival_path': './data/survival.csv',
        'split_dir': './processed/splits'
    },
    
    # 模型配置
    'path_aggregator_mode': 'mean',  # 'mean' 或 'attention'
    'omics_encoder_type': 'cnn',     # 'cnn' 或 'transformer'
    
    # 训练配置
    'batch_size': 8,
    'epochs': 100,
    'patience': 20,
    'learning_rate': 1e-3,
    'optimizer': 'adamw',
    'scheduler': 'cosine'
}
```

## 评估指标

### 1. C-index (一致性指数)
- 衡量模型预测风险评分与真实生存时间的一致性
- 值范围: 0-1，越高越好
- 随机预测: 0.5

### 2. Kaplan-Meier生存曲线
- 按预测风险评分中位数分组
- 比较高风险组和低风险组的生存差异
- 使用log-rank检验计算显著性

### 3. 分类指标 (风险等级)
- 准确率、精确率、召回率、F1分数
- 混淆矩阵

## 输出文件

### 训练输出
```
output/training_YYYYMMDD_HHMMSS/
├── best_model.pth          # 最佳模型权重
├── config.json             # 训练配置
├── training_log.csv        # 训练日志
├── test_predictions.csv    # 测试集预测结果
└── results_summary.json    # 结果摘要
```

### 评估输出
```
output/evaluation_YYYYMMDD_HHMMSS/
├── predictions.csv         # 预测结果
├── evaluation_summary.json # 评估摘要
├── kaplan_meier_curve.png  # Kaplan-Meier曲线
├── risk_score_distribution.png  # 风险评分分布
├── risk_level_distribution.png  # 风险等级分布
└── risk_vs_survival.png    # 风险与生存时间关系
```

## 扩展与定制

### 1. 添加新的数据模态
1. 在 `MultiModalDataset` 中添加新模态的数据加载
2. 创建对应的特征编码器
3. 修改 `FusionHead` 以融合新特征

### 2. 修改模型架构
- 在 `models/` 目录中修改相应模块
- 更新 `MultiModalSurvivalModel` 中的组件连接

### 3. 自定义损失函数
- 在 `models/fusion_head.py` 中添加新的损失函数
- 修改 `MultiTaskLoss` 类以包含新损失

### 4. 超参数调优
- 修改 `train.py` 中的config字典
- 使用网格搜索或贝叶斯优化

## 故障排除

### 常见问题

1. **内存不足**
   - 减小 `batch_size`
   - 使用梯度累积
   - 启用混合精度训练

2. **训练不收敛**
   - 检查学习率是否合适
   - 尝试不同的优化器
   - 检查数据预处理是否正确

3. **C-index为0.5**
   - 检查生存数据格式
   - 验证风险评分的计算
   - 检查模型是否过拟合

4. **缺少依赖包**
   ```bash
   pip install -r requirements.txt
   ```

### 调试建议

1. 运行测试函数验证各模块:
   ```bash
   python -m data.dataset
   python -m models.aggregator
   python -m models.model
   ```

2. 使用小批量数据测试:
   ```python
   # 在train.py中临时减小数据集
   train_patients = train_patients[:10]
   ```

## 引用

如果本项目对您的研究有帮助，请引用:

```bibtex
@software{multimodal_survival_prediction,
  title = {多模态生存预测模型},
  author = {MAXEY-MENG},
  year = {2026},
  url = {https://github.com/v2/multimodal-survival}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过以下方式联系:
- 邮箱: mengxiangyang0225@gmail.com
- GitHub Issues: [项目地址](https://github.com/yourusername/multimodal-survival/issues)