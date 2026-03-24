#!/usr/bin/env python3
"""
extract_epic_annotation.py

从Illumina Infinium MethylationEPIC v1.0 B5清单文件（CSV格式）中提取核心注释列，
生成精简版CSV文件，用于后续甲基化数据分析。
"""

import pandas as pd
import numpy as np
import os
import sys


def find_data_start_row(file_path, encoding='utf-8'):
    """
    查找数据行的起始位置（跳过文件头部的公司信息等）。
    数据起始行以 '[Assay]' 开头，其后下一行为列名行。
    返回应跳过的行数（即列名所在的行索引）。
    """
    with open(file_path, 'r', encoding=encoding) as f:
        for i, line in enumerate(f):
            if line.strip().startswith('[Assay]'):
                # 列名在下一行
                return i + 1  # 列名行的索引
    raise ValueError("未找到 '[Assay]' 标识，请检查文件格式。")


def main(input_file, output_file, usecols=None, filter_flagged=False):
    """
    提取核心注释列并保存。

    暂时不用⬇️
    'Relation_to_UCSC_CpG_Island',
    'CHR',
    'MAPINFO',
    'Strand',
    'CHR_hg38',
    'Start_hg38',
    'End_hg38',
    'Strand_hg38'

    'CHR': 'str',  # 染色体可能有 'X','Y' 等
    'MAPINFO': 'Int64',  # 允许缺失值用pd.Int64Dtype()
    'Strand': 'str',
    'Relation_to_UCSC_CpG_Island': 'str',
        'MFG_Change_Flagged': 'str',  # 文件中可能是 'TRUE'/'FALSE'
        'CHR_hg38': 'str',
        'Start_hg38': 'Int64',
        'End_hg38': 'Int64',
        'Strand_hg38': 'str',
    暂时不用⬆️

    Parameters
    ----------
    input_file : str
        输入的B5清单文件路径
    output_file : str
        输出的精简CSV文件路径
    usecols : list, optional
        需要提取的列名列表，若为None则使用默认核心列
    filter_flagged : bool, optional
        是否排除 MFG_Change_Flagged 为 TRUE 的探针（默认False）
    """
    # 默认核心列（根据分析需求可调整）
    if usecols is None:
        usecols = [
            'IlmnID',

            'UCSC_RefGene_Name',
            'UCSC_RefGene_Group',
            'MFG_Change_Flagged'
        ]

    print("正在定位数据起始行...")
    skip_rows = find_data_start_row(input_file)
    print(f"数据起始行（列名行）索引: {skip_rows}")

    # 定义每列的数据类型以节省内存
    dtype_dict = {
        'IlmnID': 'str',

        'UCSC_RefGene_Name': 'str',
        'UCSC_RefGene_Group': 'str',
        'MFG_Change_Flagged': 'str'  # 文件中可能是 'TRUE'/'FALSE'
    }

    # 仅保留需要的列对应的dtype
    dtype_use = {col: dtype_dict[col] for col in usecols if col in dtype_dict}

    print(f"正在读取文件（仅读取指定列）...")
    # 使用usecols和dtype，跳过表头之前的行
    df = pd.read_csv(
        input_file,
        skiprows=skip_rows,  # 跳过头信息，列名行作为表头
        usecols=usecols,
        dtype=dtype_use,
        na_values=['', 'NaN'],  # 将空字符串转为NaN
        low_memory=False  # 避免混合类型警告（实际已指定类型）
    )

    print(f"读取完成，共 {len(df)} 个探针。")

    # 可选：过滤掉制造变更标记的探针
    if filter_flagged:
        if 'MFG_Change_Flagged' in df.columns:
            before = len(df)
            # MFG_Change_Flagged 为 'TRUE' 或 True，转换为布尔值
            df = df[df['MFG_Change_Flagged'] != 'TRUE']
            print(f"过滤掉 {before - len(df)} 个标记探针，剩余 {len(df)} 个。")
        else:
            print("警告：'MFG_Change_Flagged' 列不存在，无法过滤。")

    # 将MAPINFO等整数列中的NaN填充为0（或保留NaN，取决于后续处理）
    # 此处可选处理，但保留原始缺失值更安全
    print(f"正在保存结果到 {output_file} ...")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print("完成。")


if __name__ == '__main__':
    # 命令行参数示例
    if len(sys.argv) < 3:
        print("用法: python extract_mty_epic_annotation.py <输入文件> <输出文件> [--filter]")
        print("示例: python extract_mty_epic_annotation.py F:/Database/TCGA-ESCA-RNA-CNV-SNP-CLINICAL/infinium-methylationepic-v-1-0-b5-manifest-file.csv illumina_epic_v1_B5_manifest.csv --filter")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    filter_flag = '--filter' in sys.argv

    main(input_path, output_path, filter_flagged=filter_flag)