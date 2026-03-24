#!/usr/bin/env python3
"""
创建示例数据脚本

用于生成TCGA-ESCA预处理脚本的示例数据，帮助用户理解数据格式要求。
"""

import os
import pandas as pd
from pathlib import Path
import numpy as np

def create_example_structure(base_dir="example_data"):
    """
    创建示例数据目录结构
    
    Args:
        base_dir: 基础目录路径
    """
    base_path = Path(base_dir)
    
    # 创建目录
    dirs = ["rna", "cnv", "meth", "output"]
    for dir_name in dirs:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    print(f"创建目录结构: {base_path}")
    return base_path

def create_rna_data(base_path, num_patients=3, num_genes=10):
    """
    创建RNA-seq示例数据
    
    Args:
        base_path: 基础目录路径
        num_patients: 患者数量
        num_genes: 基因数量
    """
    rna_dir = base_path / "rna"
    
    # 示例基因数据
    gene_data = [
        {"gene_id": "ENSG00000000003.15", "gene_name": "TSPAN6"},
        {"gene_id": "ENSG00000000005.6", "gene_name": "TNMD"},
        {"gene_id": "ENSG00000000419.13", "gene_name": "DPM1"},
        {"gene_id": "ENSG00000000457.14", "gene_name": "SCYL3"},
        {"gene_id": "ENSG00000000460.17", "gene_name": "C1orf112"},
        {"gene_id": "ENSG00000000938.13", "gene_name": "FGR"},
        {"gene_id": "ENSG00000000971.16", "gene_name": "CFH"},
        {"gene_id": "ENSG00000001036.14", "gene_name": "FUCA2"},
        {"gene_id": "ENSG00000001084.13", "gene_name": "GCLC"},
        {"gene_id": "ENSG00000001167.14", "gene_name": "NFYA"},
    ]
    
    # 限制基因数量
    gene_data = gene_data[:num_genes]
    
    for i in range(num_patients):
        patient_id = f"TCGA-ES-{i+1:04d}"
        filename = rna_dir / f"{patient_id}_rna.tsv"
        
        # 为每个基因生成随机的TPM值
        data = []
        for gene in gene_data:
            tpm = np.random.uniform(0.1, 100.0)  # 随机TPM值
            data.append({
                "gene_id": gene["gene_id"],
                "gene_name": gene["gene_name"],
                "tpm_unstranded": round(tpm, 2)
            })
        
        # 添加统计行（以N_开头）
        data.append({
            "gene_id": "N_unmapped",
            "gene_name": "N_unmapped",
            "tpm_unstranded": 0.0
        })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, sep='\t', index=False)
        print(f"创建RNA文件: {filename}")
    
    return gene_data

def create_cnv_data(base_path, gene_data, num_patients=3):
    """
    创建CNV示例数据
    
    Args:
        base_path: 基础目录路径
        gene_data: 基因数据列表
        num_patients: 患者数量
    """
    cnv_dir = base_path / "cnv"
    
    for i in range(num_patients):
        patient_id = f"TCGA-ES-{i+1:04d}"
        filename = cnv_dir / f"{patient_id}_cnv.tsv"
        
        # 为每个基因生成随机的copy number值
        data = []
        for gene in gene_data:
            copy_number = np.random.uniform(0.0, 4.0)  # 随机copy number
            data.append({
                "gene_id": gene["gene_id"],
                "copy_number": round(copy_number, 2)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, sep='\t', index=False)
        print(f"创建CNV文件: {filename}")

def create_meth_data(base_path, num_patients=3):
    """
    创建甲基化示例数据
    
    Args:
        base_path: 基础目录路径
        num_patients: 患者数量
    """
    meth_dir = base_path / "meth"
    
    # 示例探针数据
    probe_data = [
        {"probe_id": "cg00000029", "gene_symbol": "TSPAN6"},
        {"probe_id": "cg00000108", "gene_symbol": "TNMD"},
        {"probe_id": "cg00000109", "gene_symbol": "DPM1"},
        {"probe_id": "cg00000165", "gene_symbol": "SCYL3"},
        {"probe_id": "cg00000236", "gene_symbol": "C1orf112"},
        {"probe_id": "cg00000289", "gene_symbol": "FGR"},
        {"probe_id": "cg00000321", "gene_symbol": "CFH"},
        {"probe_id": "cg00000363", "gene_symbol": "FUCA2"},
        {"probe_id": "cg00000412", "gene_symbol": "GCLC"},
        {"probe_id": "cg00000483", "gene_symbol": "NFYA"},
    ]
    
    for i in range(num_patients):
        patient_id = f"TCGA-ES-{i+1:04d}"
        filename = meth_dir / f"{patient_id}_meth.txt"
        
        # 为每个探针生成随机的beta值
        data = []
        for probe in probe_data:
            beta = np.random.uniform(0.0, 1.0)  # 随机beta值
            data.append([probe["probe_id"], round(beta, 3)])
        
        df = pd.DataFrame(data)
        df.to_csv(filename, sep='\t', index=False, header=False)
        print(f"创建甲基化文件: {filename}")
    
    return probe_data

def create_probe_annotation(base_path, probe_data):
    """
    创建探针注释文件
    
    Args:
        base_path: 基础目录路径
        probe_data: 探针数据列表
    """
    filename = base_path / "illumina_epic_manifest.csv"
    
    data = []
    for probe in probe_data:
        data.append({
            "probe_id": probe["probe_id"],
            "gene_symbol": probe["gene_symbol"]
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"创建探针注释文件: {filename}")

def create_gene_mapping(base_path, gene_data):
    """
    创建基因映射文件
    
    Args:
        base_path: 基础目录路径
        gene_data: 基因数据列表
    """
    filename = base_path / "gene_mapping.csv"
    
    data = []
    for gene in gene_data:
        # 去掉基因ID的版本号
        gene_id_clean = gene["gene_id"].split('.')[0]
        data.append({
            "gene_id": gene_id_clean,
            "gene_name": gene["gene_name"]
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"创建基因映射文件: {filename}")

def create_usage_script(base_path):
    """
    创建使用示例脚本
    
    Args:
        base_path: 基础目录路径
    """
    filename = base_path / "run_example.sh"
    
    content = f"""#!/bin/bash
# TCGA-ESCA预处理脚本使用示例

echo "运行预处理脚本..."
python ../preprocess_tcga_esca_improved.py \\
  --rna_dir ./rna \\
  --cnv_dir ./cnv \\
  --meth_dir ./meth \\
  --probe_annotation ./illumina_epic_manifest.csv \\
  --gene_mapping ./gene_mapping.csv \\
  --output_dir ./output \\
  --log_level INFO

echo ""
echo "查看输出文件:"
echo "  张量文件: ./output/tensor.npy"
echo "  基因列表: ./output/genes.txt"
echo "  患者列表: ./output/patients.txt"
echo "  处理摘要: ./output/summary.txt"
"""
    
    with open(filename, 'w') as f:
        f.write(content)
    
    # 在Windows上创建批处理文件
    bat_filename = base_path / "run_example.bat"
    bat_content = f"""@echo off
REM TCGA-ESCA预处理脚本使用示例

echo 运行预处理脚本...
python ../preprocess_tcga_esca_improved.py ^
  --rna_dir ./rna ^
  --cnv_dir ./cnv ^
  --meth_dir ./meth ^
  --probe_annotation ./illumina_epic_manifest.csv ^
  --gene_mapping ./gene_mapping.csv ^
  --output_dir ./output ^
  --log_level INFO

echo.
echo 查看输出文件:
echo   张量文件: ./output/tensor.npy
echo   基因列表: ./output/genes.txt
echo   患者列表: ./output/patients.txt
echo   处理摘要: ./output/summary.txt
pause
"""
    
    with open(bat_filename, 'w') as f:
        f.write(bat_content)
    
    print(f"创建使用脚本: {filename}")
    print(f"创建批处理文件: {bat_filename}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='创建TCGA-ESCA示例数据')
    parser.add_argument('--output_dir', type=str, default='example_data',
                       help='输出目录 (默认: example_data)')
    parser.add_argument('--patients', type=int, default=3,
                       help='患者数量 (默认: 3)')
    parser.add_argument('--genes', type=int, default=10,
                       help='基因数量 (默认: 10)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("创建TCGA-ESCA示例数据")
    print("=" * 60)
    
    # 创建目录结构
    base_path = create_example_structure(args.output_dir)
    
    # 创建数据
    print("\n创建数据文件...")
    gene_data = create_rna_data(base_path, args.patients, args.genes)
    create_cnv_data(base_path, gene_data, args.patients)
    probe_data = create_meth_data(base_path, args.patients)
    create_probe_annotation(base_path, probe_data)
    create_gene_mapping(base_path, gene_data)
    
    # 创建使用脚本
    create_usage_script(base_path)
    
    print("\n" + "=" * 60)
    print("示例数据创建完成!")
    print(f"数据目录: {base_path}")
    print("\n使用方法:")
    print(f"  1. 进入目录: cd {base_path}")
    print(f"  2. 运行脚本: ./run_example.sh (Linux/Mac)")
    print(f"             或 run_example.bat (Windows)")
    print("\n生成的文件:")
    print(f"  RNA-seq文件: {base_path}/rna/ (TSV格式)")
    print(f"  CNV文件: {base_path}/cnv/ (TSV格式)")
    print(f"  甲基化文件: {base_path}/meth/ (TXT格式)")
    print(f"  探针注释: {base_path}/illumina_epic_manifest.csv")
    print(f"  基因映射: {base_path}/gene_mapping.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()