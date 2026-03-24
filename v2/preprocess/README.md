# TCGA-ESCA 多组学数据预处理脚本

## 概述

本脚本用于将TCGA-ESCA项目的RNA-seq、CNV和甲基化三种组学数据转换为三维张量 (G, S, 3)，其中：
- G: 共同基因数
- S: 患者数

## 功能特性

1. **数据预处理**：
   - RNA-seq: 过滤统计行，处理基因ID版本号，提取TPM值
   - CNV: 处理基因ID版本号，映射基因ID到基因名
   - 甲基化: 探针到基因映射，基因水平beta值平均

2. **数据整合**：
   - 自动识别三种组学都有的患者
   - 计算共同基因集（三种组学交集）
   - 构建三维张量 (G, S, 3)

3. **输出文件**：
   - `tensor.npy`: 三维numpy数组
   - `genes.txt`: 基因名称列表
   - `patients.txt`: 患者ID列表
   - `summary.txt`: 处理摘要
   - `gene_mapping.txt`: 基因ID到基因名映射（可选）

## 文件结构要求

### 输入文件结构
```
data/
├── rna/                    # RNA-seq文件目录
│   ├── TCGA-ES-0001_rna.tsv
│   ├── TCGA-ES-0002_rna.tsv
│   └── ...
├── cnv/                    # CNV文件目录
│   ├── TCGA-ES-0001_cnv.tsv
│   ├── TCGA-ES-0002_cnv.tsv
│   └── ...
├── meth/                   # 甲基化文件目录
│   ├── TCGA-ES-0001_meth.txt
│   ├── TCGA-ES-0002_meth.txt
│   └── ...
├── illumina_epic_manifest.csv  # 探针注释文件
└── gene_mapping.csv        # 基因ID映射文件（可选）
```

### 文件格式说明

1. **RNA-seq文件** (TSV格式):
   - 必需列: `gene_id`, `gene_name`, `tpm_unstranded`
   - 示例:
     ```
     gene_id	gene_name	tpm_unstranded
     ENSG00000000003.15	TSPAN6	10.5
     ENSG00000000005.6	TNMD	20.3
     ```

2. **CNV文件** (TSV格式):
   - 必需列: `gene_id`, `copy_number`
   - 示例:
     ```
     gene_id	copy_number
     ENSG00000000003.15	2.0
     ENSG00000000005.6	1.8
     ```

3. **甲基化文件** (TXT格式，制表符分隔):
   - 两列: 探针ID, beta值
   - 示例:
     ```
     cg00000029	0.1
     cg00000108	0.2
     ```

4. **探针注释文件** (CSV格式):
   - 必需列: `probe_id`, `gene_symbol`
   - 示例:
     ```
     probe_id,gene_symbol
     cg00000029,TSPAN6
     cg00000108,TNMD
     ```

5. **基因映射文件** (CSV格式，可选):
   - 必需列: `gene_id`, `gene_name`
   - 示例:
     ```
     gene_id,gene_name
     ENSG00000000003,TSPAN6
     ENSG00000000005,TNMD
     ```

## 安装要求

```bash
pip install numpy pandas
```

## 使用方法

### 基本用法

```bash
python preprocess_tcga_esca_improved.py \
  --rna_dir ./data/rna \
  --cnv_dir ./data/cnv \
  --meth_dir ./data/meth \
  --probe_annotation ./data/illumina_epic_manifest.csv \
  --output_dir ./output
```

### 使用基因映射文件

```bash
python preprocess_tcga_esca_improved.py \
  --rna_dir ./data/rna \
  --cnv_dir ./data/cnv \
  --meth_dir ./data/meth \
  --probe_annotation ./data/illumina_epic_manifest.csv \
  --gene_mapping ./data/gene_mapping.csv \
  --output_dir ./output
```

### 设置日志级别

```bash
python preprocess_tcga_esca_improved.py \
  --rna_dir ./data/rna \
  --cnv_dir ./data/cnv \
  --meth_dir ./data/meth \
  --probe_annotation ./data/illumina_epic_manifest.csv \
  --output_dir ./output \
  --log_level DEBUG
```

## 输出说明

处理完成后，输出目录将包含以下文件：

```
output/
├── tensor.npy          # 三维张量 (G, S, 3)
├── genes.txt           # 基因名称列表 (G个)
├── patients.txt        # 患者ID列表 (S个)
├── gene_mapping.txt    # 基因ID到基因名映射
└── summary.txt         # 处理摘要
```

### 张量结构

三维张量 `tensor.npy` 的形状为 `(G, S, 3)`，其中：
- 维度0: 基因 (按genes.txt顺序)
- 维度1: 患者 (按patients.txt顺序)
- 维度2: 组学数据通道
  - 通道0: RNA-seq TPM值
  - 通道1: CNV copy number
  - 通道2: 甲基化beta值

## 错误处理

脚本包含完善的错误处理机制：
1. 文件不存在检查
2. 必要列缺失检查
3. 数据格式验证
4. 空数据检查
5. 映射失败处理

## 测试

运行测试脚本验证功能：

```bash
python test_preprocess.py
```

## 注意事项

1. **基因名一致性**: 确保RNA-seq的gene_name、甲基化的gene_symbol和CNV映射后的基因名一致
2. **患者ID提取**: 脚本从文件名中提取患者ID，支持TCGA格式 (TCGA-XX-XXXX)
3. **内存使用**: 处理大量数据时注意内存使用，建议分批处理
4. **文件编码**: 确保所有文件使用UTF-8编码

## 故障排除

### 常见问题

1. **"没有找到同时具有三种组学数据的患者"**
   - 检查患者ID提取是否正确
   - 确保三种数据目录包含相同的患者

2. **"没有找到共同基因"**
   - 检查基因映射是否正确
   - 验证基因名在不同组学中是否一致

3. **文件读取错误**
   - 检查文件格式和分隔符
   - 验证文件编码

### 调试模式

使用 `--log_level DEBUG` 参数获取详细日志信息：

```bash
python preprocess_tcga_esca_improved.py ... --log_level DEBUG
```

## 版本说明

### v1.0 (基础版)
- 基本数据预处理功能
- 三维张量构建
- 基础错误处理

### v2.0 (改进版)
- 基因ID到基因名映射
- 更完善的错误处理
- 详细统计信息
- 测试脚本

## 许可证

本项目使用MIT许可证。

## 作者

MAXEY-MENG

## 更新日志

- 2026-03-23: 初始版本发布
- 2026-03-23: 改进版发布，增加基因映射功能