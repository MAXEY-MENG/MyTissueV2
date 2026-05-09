#!/bin/bash

# 多模态生存预测模型运行脚本
# 适用于Linux/Mac系统

set -e  # 遇到错误时退出

echo "========================================"
echo "多模态生存预测模型"
echo "========================================"

# 检查Python版本
echo "检查Python版本..."
python --version

## 检查依赖
#echo "检查依赖包..."
#if ! command -v pip &> /dev/null; then
#    echo "错误: pip未安装"
#    exit 1
#fi
#
## 安装依赖
#echo "安装依赖包..."
#pip install -r requirements.txt
#
## 创建必要的目录
#echo "创建目录结构..."
#mkdir -p processed/splits
#mkdir -p output
#mkdir -p logs

# 检查数据文件
echo "检查数据文件..."
if [ ! -f "./processed/tensor.npy" ]; then
    echo "警告: ./processed/tensor.npy 不存在"
    echo "请确保组学数据已预处理"
fi

if [ ! -f "./processed/patients.txt" ]; then
    echo "警告: ./processed/patients.txt 不存在"
fi

if [ ! -f "./data/survival.csv" ]; then
    echo "警告: ./data/survival.csv 不存在"
    echo "请准备生存数据文件"
fi

# 数据划分
echo "执行数据划分..."
python -c "
from data.split import create_splits_from_patient_file
try:
    create_splits_from_patient_file(
        patients_file='./processed/patients.txt',
        output_dir='./processed/splits',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    print('数据划分完成')
except Exception as e:
    print(f'数据划分失败: {e}')
"

# 训练模型
echo "开始训练模型..."
echo "========================================"
python train.py

# 评估模型
echo "评估模型..."
echo "========================================"
python evaluate.py

echo "========================================"
echo "完成!"
echo "结果保存在 output/ 目录中"
echo "========================================"