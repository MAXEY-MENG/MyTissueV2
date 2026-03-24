#!/usr/bin/env python3
"""
测试TCGA-ESCA预处理脚本
"""

import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加当前目录到路径，以便导入脚本
sys.path.insert(0, str(Path(__file__).parent))

def create_test_data():
    """创建测试数据"""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # 创建输出目录
    output_dir = test_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 创建RNA-seq目录和文件
    rna_dir = test_dir / "rna"
    rna_dir.mkdir(exist_ok=True)
    
    # 创建两个患者的RNA-seq数据
    for i in range(2):
        patient_id = f"TCGA-ES-{i+1:04d}"
        filename = rna_dir / f"{patient_id}_rna.tsv"
        
        # 创建测试数据
        data = {
            'gene_id': ['ENSG00000000003.15', 'ENSG00000000005.6', 'ENSG00000000419.13', 'ENSG00000000457.14'],
            'gene_name': ['TSPAN6', 'TNMD', 'DPM1', 'SCYL3'],
            'tpm_unstranded': [10.5 + i, 20.3 + i, 5.7 + i, 15.2 + i]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, sep='\t', index=False)
    
    # 创建CNV目录和文件
    cnv_dir = test_dir / "cnv"
    cnv_dir.mkdir(exist_ok=True)
    
    # 创建两个患者的CNV数据
    for i in range(2):
        patient_id = f"TCGA-ES-{i+1:04d}"
        filename = cnv_dir / f"{patient_id}_cnv.tsv"
        
        # 创建测试数据
        data = {
            'gene_id': ['ENSG00000000003.15', 'ENSG00000000005.6', 'ENSG00000000419.13', 'ENSG00000000457.14'],
            'copy_number': [2.0 + i*0.1, 1.8 + i*0.1, 2.2 + i*0.1, 1.9 + i*0.1]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, sep='\t', index=False)
    
    # 创建甲基化目录和文件
    meth_dir = test_dir / "meth"
    meth_dir.mkdir(exist_ok=True)
    
    # 创建两个患者的甲基化数据
    for i in range(2):
        patient_id = f"TCGA-ES-{i+1:04d}"
        filename = meth_dir / f"{patient_id}_meth.txt"
        
        # 创建测试数据
        data = {
            'probe_id': ['cg00000029', 'cg00000108', 'cg00000109', 'cg00000165'],
            'beta_value': [0.1 + i*0.05, 0.2 + i*0.05, 0.3 + i*0.05, 0.4 + i*0.05]
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, sep='\t', index=False, header=False)
    
    # 创建探针注释文件
    probe_file = test_dir / "illumina_epic_manifest.csv"
    data = {
        'probe_id': ['cg00000029', 'cg00000108', 'cg00000109', 'cg00000165'],
        'gene_symbol': ['TSPAN6', 'TNMD', 'DPM1', 'SCYL3']
    }
    df = pd.DataFrame(data)
    df.to_csv(probe_file, index=False)
    
    # 创建基因映射文件（可选）
    gene_mapping_file = test_dir / "gene_mapping.csv"
    data = {
        'gene_id': ['ENSG00000000003', 'ENSG00000000005', 'ENSG00000000419', 'ENSG00000000457'],
        'gene_name': ['TSPAN6', 'TNMD', 'DPM1', 'SCYL3']
    }
    df = pd.DataFrame(data)
    df.to_csv(gene_mapping_file, index=False)
    
    return test_dir

def test_basic_functionality():
    """测试基本功能"""
    print("创建测试数据...")
    test_dir = create_test_data()
    
    try:
        # 导入预处理脚本
        from preprocess.preprocess_tcga_esca_improved import TCGAESCAPreprocessor
        
        print("\n测试预处理脚本...")
        
        # 创建预处理器
        preprocessor = TCGAESCAPreprocessor(
            rna_dir=str(test_dir / "rna"),
            cnv_dir=str(test_dir / "cnv"),
            meth_dir=str(test_dir / "meth"),
            probe_annotation_file=str(test_dir / "illumina_epic_manifest.csv"),
            gene_mapping_file=str(test_dir / "gene_mapping.csv"),
            output_dir=str(test_dir / "output")
        )
        
        # 运行预处理
        print("运行预处理流程...")
        tensor = preprocessor.run()
        
        # 验证结果
        print(f"\n验证结果:")
        print(f"张量形状: {tensor.shape}")
        print(f"预期形状: (4, 2, 3)")  # 4个基因，2个患者，3个通道
        
        assert tensor.shape == (4, 2, 3), f"张量形状不正确: {tensor.shape}"
        
        # 检查基因列表
        genes_file = test_dir / "output" / "genes.txt"
        assert genes_file.exists(), "基因列表文件不存在"
        
        with open(genes_file, 'r') as f:
            genes = [line.strip() for line in f]
        
        print(f"基因列表: {genes}")
        assert len(genes) == 4, f"基因数量不正确: {len(genes)}"
        
        # 检查患者列表
        patients_file = test_dir / "output" / "patients.txt"
        assert patients_file.exists(), "患者列表文件不存在"
        
        with open(patients_file, 'r') as f:
            patients = [line.strip() for line in f]
        
        print(f"患者列表: {patients}")
        assert len(patients) == 2, f"患者数量不正确: {len(patients)}"
        
        # 检查张量文件
        tensor_file = test_dir / "output" / "tensor.npy"
        assert tensor_file.exists(), "张量文件不存在"
        
        loaded_tensor = np.load(tensor_file)
        assert np.array_equal(tensor, loaded_tensor), "加载的张量与原始张量不匹配"
        
        print("\n所有测试通过!")
        
        # 显示张量内容
        print("\n张量内容:")
        for g_idx, gene in enumerate(genes):
            print(f"\n基因 {gene}:")
            for p_idx, patient in enumerate(patients):
                print(f"  患者 {patient}: RNA={tensor[g_idx, p_idx, 0]:.2f}, "
                      f"CNV={tensor[g_idx, p_idx, 1]:.2f}, "
                      f"Meth={tensor[g_idx, p_idx, 2]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理测试数据
        print("\n清理测试数据...")
        if test_dir.exists():
            shutil.rmtree(test_dir)

def test_without_gene_mapping():
    """测试没有基因映射文件的情况"""
    print("\n\n测试没有基因映射文件的情况...")
    
    test_dir = create_test_data()
    
    try:
        from preprocess.preprocess_tcga_esca_improved import TCGAESCAPreprocessor
        
        # 创建预处理器（不使用基因映射文件）
        preprocessor = TCGAESCAPreprocessor(
            rna_dir=str(test_dir / "rna"),
            cnv_dir=str(test_dir / "cnv"),
            meth_dir=str(test_dir / "meth"),
            probe_annotation_file=str(test_dir / "illumina_epic_manifest.csv"),
            gene_mapping_file=None,  # 不提供基因映射文件
            output_dir=str(test_dir / "output_no_mapping")
        )
        
        # 运行预处理
        tensor = preprocessor.run()
        
        print(f"张量形状: {tensor.shape}")
        assert tensor.shape[0] > 0, "应该找到一些共同基因"
        
        print("测试通过!")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False
        
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)

def main():
    """主测试函数"""
    print("=" * 60)
    print("TCGA-ESCA预处理脚本测试")
    print("=" * 60)
    
    # 测试1: 基本功能
    success1 = test_basic_functionality()
    
    # 测试2: 没有基因映射文件
    success2 = test_without_gene_mapping()
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  基本功能测试: {'通过' if success1 else '失败'}")
    print(f"  无基因映射测试: {'通过' if success2 else '失败'}")
    print("=" * 60)
    
    if success1 and success2:
        print("\n所有测试通过! 脚本功能正常。")
        return 0
    else:
        print("\n部分测试失败。请检查脚本实现。")
        return 1

if __name__ == "__main__":
    exit(main())