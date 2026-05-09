@echo off
chcp 65001 >nul
echo ========================================
echo 多模态生存预测模型
echo ========================================

REM 检查Python版本
echo 检查Python版本...
python --version
if errorlevel 1 (
    echo 错误: Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 检查pip
echo 检查依赖包...
python -m pip --version
if errorlevel 1 (
    echo 错误: pip未安装
    pause
    exit /b 1
)

REM 安装依赖
echo 安装依赖包...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo 警告: 依赖安装失败，尝试继续运行...
)

REM 创建必要的目录
echo 创建目录结构...
if not exist "processed\splits" mkdir "processed\splits"
if not exist "output" mkdir "output"
if not exist "logs" mkdir "logs"

REM 检查数据文件
echo 检查数据文件...
if not exist "processed\tensor.npy" (
    echo 警告: processed\tensor.npy 不存在
    echo 请确保组学数据已预处理
)

if not exist "processed\patients.txt" (
    echo 警告: processed\patients.txt 不存在
)

if not exist "data\survival.csv" (
    echo 警告: data\survival.csv 不存在
    echo 请准备生存数据文件
)

REM 数据划分
echo 执行数据划分...
python -c "from data.split import create_splits_from_patient_file
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
    print(f'数据划分失败: {e}')"

REM 训练模型
echo 开始训练模型...
echo ========================================
python train.py
if errorlevel 1 (
    echo 训练失败
    pause
    exit /b 1
)

REM 评估模型
echo 评估模型...
echo ========================================
python evaluate.py
if errorlevel 1 (
    echo 评估失败
    pause
    exit /b 1
)

echo ========================================
echo 完成!
echo 结果保存在 output\ 目录中
echo ========================================
pause