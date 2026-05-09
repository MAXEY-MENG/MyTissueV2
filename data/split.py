"""
数据划分模块
用于将患者ID划分为训练集、验证集和测试集
"""

import numpy as np
from typing import List, Tuple, Dict
import os
from sklearn.model_selection import train_test_split


def split_patients(
    patients: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify_by: List = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    划分患者ID为训练集、验证集和测试集
    
    Args:
        patients: 患者ID列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        stratify_by: 分层采样标签（可选）
        
    Returns:
        train_patients: 训练集患者ID
        val_patients: 验证集患者ID
        test_patients: 测试集患者ID
    """
    # 验证比例总和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        raise ValueError(f"比例总和应为1.0，但得到 {total_ratio}")
    
    # 首先划分训练集和临时集（验证+测试）
    temp_ratio = val_ratio + test_ratio
    train_patients, temp_patients = train_test_split(
        patients,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=stratify_by
    )
    
    # 然后从临时集中划分验证集和测试集
    val_test_ratio = val_ratio / temp_ratio if temp_ratio > 0 else 0
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=1 - val_test_ratio,  # 注意：test_size是测试集在temp中的比例
        random_state=random_seed,
        stratify=None  # 第二次划分通常不进行分层
    )
    
    return train_patients, val_patients, test_patients


def save_split_files(
    train_patients: List[str],
    val_patients: List[str],
    test_patients: List[str],
    output_dir: str = "./processed/splits"
) -> Dict[str, str]:
    """
    保存划分结果到文件
    
    Args:
        train_patients: 训练集患者ID
        val_patients: 验证集患者ID
        test_patients: 测试集患者ID
        output_dir: 输出目录
        
    Returns:
        包含文件路径的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 文件路径
    train_file = os.path.join(output_dir, "train_patients.txt")
    val_file = os.path.join(output_dir, "val_patients.txt")
    test_file = os.path.join(output_dir, "test_patients.txt")
    summary_file = os.path.join(output_dir, "split_summary.txt")
    
    # 保存患者ID列表
    def save_patient_list(filepath: str, patients: List[str]):
        with open(filepath, 'w') as f:
            for pid in patients:
                f.write(f"{pid}\n")
    
    save_patient_list(train_file, train_patients)
    save_patient_list(val_file, val_patients)
    save_patient_list(test_file, test_patients)
    
    # 保存划分摘要
    with open(summary_file, 'w') as f:
        f.write("数据划分摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"总患者数: {len(train_patients) + len(val_patients) + len(test_patients)}\n")
        f.write(f"训练集: {len(train_patients)} 患者 ({len(train_patients)/184*100:.1f}%)\n")
        f.write(f"验证集: {len(val_patients)} 患者 ({len(val_patients)/184*100:.1f}%)\n")
        f.write(f"测试集: {len(test_patients)} 患者 ({len(test_patients)/184*100:.1f}%)\n")
        f.write("\n训练集患者ID:\n")
        f.write(", ".join(train_patients[:10]))
        if len(train_patients) > 10:
            f.write(f", ... (共{len(train_patients)}个)")
        f.write("\n\n验证集患者ID:\n")
        f.write(", ".join(val_patients[:10]))
        if len(val_patients) > 10:
            f.write(f", ... (共{len(val_patients)}个)")
        f.write("\n\n测试集患者ID:\n")
        f.write(", ".join(test_patients[:10]))
        if len(test_patients) > 10:
            f.write(f", ... (共{len(test_patients)}个)")
    
    return {
        'train': train_file,
        'val': val_file,
        'test': test_file,
        'summary': summary_file
    }


def load_split_files(split_dir: str = "./processed/splits") -> Tuple[List[str], List[str], List[str]]:
    """
    从文件加载划分结果
    
    Args:
        split_dir: 划分文件目录
        
    Returns:
        train_patients, val_patients, test_patients
    """
    train_file = os.path.join(split_dir, "train_patients.txt")
    val_file = os.path.join(split_dir, "val_patients.txt")
    test_file = os.path.join(split_dir, "test_patients.txt")
    
    def load_patient_list(filepath: str) -> List[str]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"划分文件不存在: {filepath}")
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    train_patients = load_patient_list(train_file)
    val_patients = load_patient_list(val_file)
    test_patients = load_patient_list(test_file)
    
    return train_patients, val_patients, test_patients


def create_splits_from_patient_file(
    patients_file: str,
    output_dir: str = "./processed/splits",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    从患者文件创建划分
    
    Args:
        patients_file: 患者ID列表文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        
    Returns:
        包含文件路径的字典
    """
    # 加载患者ID
    with open(patients_file, 'r') as f:
        patients = [line.strip() for line in f if line.strip()]
    
    print(f"从 {patients_file} 加载了 {len(patients)} 个患者ID")
    
    # 划分患者
    train_patients, val_patients, test_patients = split_patients(
        patients=patients,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    print(f"划分完成:")
    print(f"  训练集: {len(train_patients)} 患者")
    print(f"  验证集: {len(val_patients)} 患者")
    print(f"  测试集: {len(test_patients)} 患者")
    
    # 保存划分结果
    file_paths = save_split_files(
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        output_dir=output_dir
    )
    
    print(f"划分结果已保存到 {output_dir}")
    
    return file_paths


def test_split():
    """测试数据划分功能"""
    import tempfile
    import shutil
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建模拟患者ID
        patients = [f"TCGA-ES-{i:04d}" for i in range(184)]
        patients_file = os.path.join(temp_dir, "patients.txt")
        
        with open(patients_file, 'w') as f:
            for pid in patients:
                f.write(f"{pid}\n")
        
        # 测试划分功能
        print("测试 split_patients 函数...")
        train_patients, val_patients, test_patients = split_patients(
            patients=patients,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        print(f"训练集大小: {len(train_patients)}")
        print(f"验证集大小: {len(val_patients)}")
        print(f"测试集大小: {len(test_patients)}")
        
        # 检查划分是否互斥且覆盖所有患者
        all_split_patients = set(train_patients + val_patients + test_patients)
        assert len(all_split_patients) == len(patients), "划分后患者数量不一致"
        assert len(set(train_patients) & set(val_patients)) == 0, "训练集和验证集有重叠"
        assert len(set(train_patients) & set(test_patients)) == 0, "训练集和测试集有重叠"
        assert len(set(val_patients) & set(test_patients)) == 0, "验证集和测试集有重叠"
        
        # 测试保存和加载功能
        print("\n测试 save_split_files 和 load_split_files 函数...")
        output_dir = os.path.join(temp_dir, "splits")
        file_paths = save_split_files(
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
            output_dir=output_dir
        )
        
        # 检查文件是否存在
        for key, path in file_paths.items():
            if key != 'summary':  # summary文件可能不存在
                assert os.path.exists(path), f"文件 {path} 不存在"
        
        # 加载划分结果
        loaded_train, loaded_val, loaded_test = load_split_files(output_dir)
        
        assert loaded_train == train_patients, "加载的训练集不匹配"
        assert loaded_val == val_patients, "加载的验证集不匹配"
        assert loaded_test == test_patients, "加载的测试集不匹配"
        
        # 测试完整流程
        print("\n测试 create_splits_from_patient_file 函数...")
        file_paths2 = create_splits_from_patient_file(
            patients_file=patients_file,
            output_dir=os.path.join(temp_dir, "splits2"),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        print("\n所有测试通过!")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_split()