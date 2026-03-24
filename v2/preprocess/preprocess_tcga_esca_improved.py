#!/usr/bin/env python3
"""
TCGA-ESCA 多组学数据预处理脚本 (改进版)

将RNA-seq、CNV、甲基化三种组学数据转换为三维张量 (G, S, 3)
其中：
  G: 共同基因数
  S: 患者数

数据要求：
1. RNA-seq文件：每个患者一个TSV，包含gene_id, gene_name, tpm_unstranded等列
2. CNV文件：每个患者一个TSV，包含gene_id, copy_number等列
3. 甲基化文件：每个患者一个TXT，包含探针ID和beta值
4. 探针注释文件：CSV格式，包含probe_id和gene_symbol列
5. (可选) 基因ID映射文件：包含gene_id和gene_name的映射

输出：
1. 三维张量: tensor.npy (G, S, 3)
2. 基因列表: genes.txt (G个基因名)
3. 患者ID列表: patients.txt (S个患者ID)
4. 处理摘要: summary.txt
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import warnings
import logging
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TCGAESCAPreprocessor:
    """TCGA-ESCA多组学数据预处理类 (改进版)"""
    
    def __init__(self, 
                 rna_dir: str,
                 cnv_dir: str,
                 meth_dir: str,
                 probe_annotation_file: str,
                 gene_mapping_file: Optional[str] = None,
                 output_dir: str = "./output"):
        """
        初始化预处理器
        
        Args:
            rna_dir: RNA-seq文件目录
            cnv_dir: CNV文件目录
            meth_dir: 甲基化文件目录
            probe_annotation_file: 探针注释文件路径
            gene_mapping_file: 基因ID到基因名的映射文件（可选）
            output_dir: 输出目录
        """
        self.rna_dir = Path(rna_dir)
        self.cnv_dir = Path(cnv_dir)
        self.meth_dir = Path(meth_dir)
        self.probe_annotation_file = Path(probe_annotation_file)
        self.gene_mapping_file = Path(gene_mapping_file) if gene_mapping_file else None
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.rna_data: Dict[str, pd.Series] = {}  # 患者ID -> RNA数据Series (gene_name -> value)
        self.cnv_data: Dict[str, pd.Series] = {}   # 患者ID -> CNV数据Series (gene_name -> value)
        self.meth_data: Dict[str, pd.Series] = {}  # 患者ID -> 甲基化数据Series (gene_name -> value)
        
        # 原始CNV数据（gene_id -> value），用于映射
        self.cnv_raw_data: Dict[str, pd.Series] = {}  # 患者ID -> CNV原始数据Series (gene_id -> value)
        
        # 基因集合
        self.rna_genes: Set[str] = set()
        self.cnv_genes: Set[str] = set()
        self.meth_genes: Set[str] = set()
        self.common_genes: List[str] = []
        
        # 患者列表
        self.patient_ids: List[str] = []
        
        # 映射表
        self.probe_to_gene: Dict[str, str] = {}  # 探针ID -> 基因名
        self.gene_id_to_name: Dict[str, str] = {}  # 基因ID -> 基因名
        self.gene_name_to_id: Dict[str, str] = {}  # 基因名 -> 基因ID
        
    def _extract_patient_id(self, filename: str) -> str:
        """
        从文件名中提取患者ID
        
        Args:
            filename: 文件名
            
        Returns:
            患者ID
        """
        # TCGA患者ID通常格式: TCGA-XX-XXXX
        patterns = [
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})',
            r'([A-Z0-9]{2}-[A-Z0-9]{4})',
            r'([A-Z0-9]{4}-[A-Z0-9]{2}-[A-Z0-9]{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        # 如果没有匹配到TCGA模式，返回不带扩展名的文件名
        return Path(filename).stem
    
    def _process_gene_id(self, gene_id: str) -> Optional[str]:
        """
        处理基因ID：去掉版本号，保留ENSG开头的基因
        
        Args:
            gene_id: 原始基因ID
            
        Returns:
            处理后的基因ID，如果不是ENSG开头则返回None
        """
        if not isinstance(gene_id, str):
            return None
        
        # 去掉版本号
        gene_id_clean = gene_id.split('.')[0]
        
        # 只保留ENSG开头的基因
        if gene_id_clean.startswith('ENSG'):
            return gene_id_clean
        return None
    
    def load_gene_mapping(self) -> None:
        """
        加载基因ID到基因名的映射文件
        
        如果提供了映射文件，则从文件中加载；
        否则，尝试从RNA-seq数据中提取映射关系。
        """
        if self.gene_mapping_file and self.gene_mapping_file.exists():
            logger.info(f"加载基因映射文件: {self.gene_mapping_file}")
            try:
                # 尝试不同的分隔符
                for sep in [',', '\t', ';']:
                    try:
                        df = pd.read_csv(self.gene_mapping_file, sep=sep)
                        if 'gene_id' in df.columns and 'gene_name' in df.columns:
                            break
                    except:
                        continue
                else:
                    logger.warning("无法解析基因映射文件，将从RNA数据中提取映射")
                    return
                
                # 清理数据
                df = df.dropna(subset=['gene_id', 'gene_name'])
                
                # 建立映射
                for _, row in df.iterrows():
                    gene_id = str(row['gene_id']).strip()
                    gene_name = str(row['gene_name']).strip()
                    
                    # 处理基因ID
                    gene_id_clean = self._process_gene_id(gene_id)
                    if gene_id_clean and gene_name:
                        self.gene_id_to_name[gene_id_clean] = gene_name
                        self.gene_name_to_id[gene_name] = gene_id_clean
                
                logger.info(f"从映射文件加载了 {len(self.gene_id_to_name)} 个基因映射")
                
            except Exception as e:
                logger.warning(f"加载基因映射文件失败: {e}")
    
    def extract_gene_mapping_from_rna(self, rna_df: pd.DataFrame) -> None:
        """
        从RNA-seq数据中提取基因ID到基因名的映射
        
        Args:
            rna_df: RNA-seq数据DataFrame
        """
        if 'gene_id' not in rna_df.columns or 'gene_name' not in rna_df.columns:
            return
        
        # 过滤统计行
        rna_df = rna_df[~rna_df['gene_name'].astype(str).str.startswith('N_')]
        
        # 处理基因ID并建立映射
        for _, row in rna_df.iterrows():
            gene_id = str(row['gene_id'])
            gene_name = str(row['gene_name'])
            
            gene_id_clean = self._process_gene_id(gene_id)
            if gene_id_clean and gene_name and gene_name != 'NA':
                self.gene_id_to_name[gene_id_clean] = gene_name
                self.gene_name_to_id[gene_name] = gene_id_clean
    
    def load_probe_annotation(self) -> None:
        """
        加载探针注释文件，建立探针到基因的映射
        """
        if not self.probe_annotation_file.exists():
            raise FileNotFoundError(f"探针注释文件不存在: {self.probe_annotation_file}")
        
        logger.info(f"加载探针注释文件: {self.probe_annotation_file}")
        
        try:
            # 尝试不同的分隔符
            for sep in [',', '\t', ';']:
                try:
                    df = pd.read_csv(self.probe_annotation_file, sep=sep)
                    if 'probe_id' in df.columns and 'gene_symbol' in df.columns:
                        break
                except:
                    continue
            else:
                raise ValueError("无法解析探针注释文件，请确保包含'probe_id'和'gene_symbol'列")
            
            # 清理数据
            df = df.dropna(subset=['probe_id', 'gene_symbol'])
            
            # 建立映射
            for _, row in df.iterrows():
                probe_id = str(row['probe_id']).strip()
                gene_symbol = str(row['gene_symbol']).strip()
                
                if gene_symbol and gene_symbol != '' and gene_symbol != 'NA':
                    self.probe_to_gene[probe_id] = gene_symbol
            
            logger.info(f"加载了 {len(self.probe_to_gene)} 个探针到基因的映射")
            
        except Exception as e:
            raise ValueError(f"加载探针注释文件失败: {e}")
    
    def process_rna_file(self, filepath: Path) -> Optional[Tuple[str, pd.Series]]:
        """
        处理单个RNA-seq文件
        
        Args:
            filepath: RNA-seq文件路径
            
        Returns:
            (患者ID, 基因表达Series) 或 None（如果处理失败）
        """
        try:
            logger.debug(f"处理RNA文件: {filepath}")
            
            # 读取TSV文件
            df = pd.read_csv(filepath, sep='\t')
            
            # 检查必要列
            required_cols = ['gene_id', 'gene_name', 'tpm_unstranded']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"RNA文件缺少列 {missing_cols}: {filepath}")
                return None
            
            # 过滤统计行
            df = df[~df['gene_name'].astype(str).str.startswith('N_')]
            
            # 处理基因ID
            df['gene_id_clean'] = df['gene_id'].apply(self._process_gene_id)
            df = df.dropna(subset=['gene_id_clean'])
            
            # 提取基因映射（如果是第一个文件或映射为空）
            if not self.gene_id_to_name:
                self.extract_gene_mapping_from_rna(df)
            
            # 以基因名为索引
            df_unique = df.drop_duplicates(subset=['gene_name'], keep='first')
            
            # 创建Series
            rna_series = pd.Series(
                df_unique['tpm_unstranded'].values,
                index=df_unique['gene_name'].values
            )
            
            # 提取患者ID
            patient_id = self._extract_patient_id(filepath.name)
            
            return patient_id, rna_series
            
        except Exception as e:
            logger.error(f"处理RNA文件失败 {filepath}: {e}")
            return None
    
    def process_cnv_file(self, filepath: Path) -> Optional[Tuple[str, pd.Series, pd.Series]]:
        """
        处理单个CNV文件
        
        Args:
            filepath: CNV文件路径
            
        Returns:
            (患者ID, CNV原始数据Series, CNV映射后数据Series) 或 None（如果处理失败）
            原始数据: gene_id -> copy_number
            映射后数据: gene_name -> copy_number
        """
        try:
            logger.debug(f"处理CNV文件: {filepath}")
            
            # 读取TSV文件
            df = pd.read_csv(filepath, sep='\t')
            
            # 检查必要列
            required_cols = ['gene_id', 'copy_number']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"CNV文件缺少列 {missing_cols}: {filepath}")
                return None
            
            # 处理基因ID
            df['gene_id_clean'] = df['gene_id'].apply(self._process_gene_id)
            df = df.dropna(subset=['gene_id_clean'])
            
            # 创建原始数据Series (gene_id -> copy_number)
            cnv_raw_series = pd.Series(
                df['copy_number'].values,
                index=df['gene_id_clean'].values
            )
            
            # 映射到基因名
            cnv_mapped_data = {}
            for gene_id, copy_number in cnv_raw_series.items():
                if gene_id in self.gene_id_to_name:
                    gene_name = self.gene_id_to_name[gene_id]
                    cnv_mapped_data[gene_name] = copy_number
            
            # 创建映射后数据Series
            cnv_mapped_series = pd.Series(cnv_mapped_data)
            
            # 提取患者ID
            patient_id = self._extract_patient_id(filepath.name)
            
            return patient_id, cnv_raw_series, cnv_mapped_series
            
        except Exception as e:
            logger.error(f"处理CNV文件失败 {filepath}: {e}")
            return None
    
    def process_meth_file(self, filepath: Path) -> Optional[Tuple[str, pd.Series]]:
        """
        处理单个甲基化文件
        
        Args:
            filepath: 甲基化文件路径
            
        Returns:
            (患者ID, 甲基化数据Series) 或 None（如果处理失败）
        """
        try:
            logger.debug(f"处理甲基化文件: {filepath}")
            
            # 读取TXT文件
            df = pd.read_csv(filepath, sep='\t', header=None, names=['probe_id', 'beta_value'])
            
            if df.empty:
                logger.warning(f"甲基化文件为空: {filepath}")
                return None
            
            # 映射探针到基因
            df['gene_symbol'] = df['probe_id'].map(self.probe_to_gene)
            df = df.dropna(subset=['gene_symbol'])
            
            if df.empty:
                logger.warning(f"没有探针映射到基因: {filepath}")
                return None
            
            # 按基因取平均beta值
            gene_beta = df.groupby('gene_symbol')['beta_value'].mean()
            
            # 提取患者ID
            patient_id = self._extract_patient_id(filepath.name)
            
            return patient_id, gene_beta
            
        except Exception as e:
            logger.error(f"处理甲基化文件失败 {filepath}: {e}")
            return None
    
    def load_all_data(self) -> None:
        """
        加载所有患者的数据
        """
        logger.info("开始加载所有数据...")
        
        # 加载基因映射（如果提供了文件）
        self.load_gene_mapping()
        
        # 加载探针注释
        self.load_probe_annotation()
        
        # 处理RNA-seq文件
        logger.info(f"处理RNA-seq文件从: {self.rna_dir}")
        rna_files = list(self.rna_dir.glob("*.tsv")) + list(self.rna_dir.glob("*.txt"))
        
        for rna_file in rna_files:
            result = self.process_rna_file(rna_file)
            if result:
                patient_id, rna_series = result
                self.rna_data[patient_id] = rna_series
                self.rna_genes.update(rna_series.index.tolist())
        
        # 处理CNV文件
        logger.info(f"处理CNV文件从: {self.cnv_dir}")
        cnv_files = list(self.cnv_dir.glob("*.tsv")) + list(self.cnv_dir.glob("*.txt"))
        
        for cnv_file in cnv_files:
            result = self.process_cnv_file(cnv_file)
            if result:
                patient_id, cnv_raw_series, cnv_mapped_series = result
                self.cnv_raw_data[patient_id] = cnv_raw_series
                self.cnv_data[patient_id] = cnv_mapped_series
                self.cnv_genes.update(cnv_mapped_series.index.tolist())
        
        # 处理甲基化文件
        logger.info(f"处理甲基化文件从: {self.meth_dir}")
        meth_files = list(self.meth_dir.glob("*.txt")) + list(self.meth_dir.glob("*.tsv"))
        
        for meth_file in meth_files:
            result = self.process_meth_file(meth_file)
            if result:
                patient_id, meth_series = result
                self.meth_data[patient_id] = meth_series
                self.meth_genes.update(meth_series.index.tolist())
        
        # 记录统计信息
        logger.info(f"加载完成: {len(self.rna_data)} 个RNA样本, "
                   f"{len(self.cnv_data)} 个CNV样本, "
                   f"{len(self.meth_data)} 个甲基化样本")
        logger.info(f"基因统计: RNA={len(self.rna_genes)}, "
                   f"CNV={len(self.cnv_genes)}, "
                   f"Methylation={len(self.meth_genes)}")
        logger.info(f"基因映射: {len(self.gene_id_to_name)} 个基因ID到基因名的映射")
    
    def find_common_patients_and_genes(self) -> None:
        """
        找到所有患者共同拥有的基因集
        """
        logger.info("寻找共同患者和基因...")
        
        # 找到三种数据都有的患者
        common_patients = set(self.rna_data.keys()) & set(self.cnv_data.keys()) & set(self.meth_data.keys())
        self.patient_ids = sorted(list(common_patients))
        
        if not self.patient_ids:
            raise ValueError("没有找到同时具有三种组学数据的患者")
        
        logger.info(f"找到 {len(self.patient_ids)} 个共同患者")
        
        # 对于每个患者，找到三种组学都有的基因
        all_common_genes = set()
        
        for patient_id in self.patient_ids:
            rna_genes = set(self.rna_data[patient_id].index)
            cnv_genes = set(self.cnv_data[patient_id].index)
            meth_genes = set(self.meth_data[patient_id].index)
            
            # 取三种组学都有的基因
            patient_common_genes = rna_genes & cnv_genes & meth_genes
            all_common_genes.update(patient_common_genes)
        
        self.common_genes = sorted(list(all_common_genes))
        
        if not self.common_genes:
            raise ValueError("没有找到共同基因")
        
        logger.info(f"找到 {len(self.common_genes)} 个共同基因")
    
    def build_tensor(self) -> np.ndarray:
        """
        构建三维张量 (G, S, 3)
        
        Returns:
            三维numpy数组，形状为 (基因数, 患者数, 3)
            通道顺序: 0=RNA, 1=CNV, 2=Methylation
        """
        logger.info("构建三维张量...")
        
        G = len(self.common_genes)
        S = len(self.patient_ids)
        
        # 初始化张量
        tensor = np.zeros((G, S, 3), dtype=np.float32)
        
        # 创建基因到索引的映射
        gene_to_idx = {gene: idx for idx, gene in enumerate(self.common_genes)}
        
        # 填充张量
        for patient_idx, patient_id in enumerate(self.patient_ids):
            # RNA数据
            rna_series = self.rna_data[patient_id]
            for gene in self.common_genes:
                if gene in rna_series.index:
                    gene_idx = gene_to_idx[gene]
                    tensor[gene_idx, patient_idx, 0] = rna_series[gene]
            
            # CNV数据
            cnv_series = self.cnv_data[patient_id]
            for gene in self.common_genes:
                if gene in cnv_series.index:
                    gene_idx = gene_to_idx[gene]
                    tensor[gene_idx, patient_idx, 1] = cnv_series[gene]
            
            # 甲基化数据
            meth_series = self.meth_data[patient_id]
            for gene in self.common_genes:
                if gene in meth_series.index:
                    gene_idx = gene_to_idx[gene]
                    tensor[gene_idx, patient_idx, 2] = meth_series[gene]
        
        logger.info(f"张量构建完成: 形状={tensor.shape}, 数据类型={tensor.dtype}")
        return tensor
    
    def save_results(self, tensor: np.ndarray) -> None:
        """
        保存结果文件
        
        Args:
            tensor: 三维张量
        """
        logger.info("保存结果文件...")
        
        # 保存张量
        tensor_path = self.output_dir / "tensor.npy"
        np.save(tensor_path, tensor)
        logger.info(f"张量保存到: {tensor_path}")
        
        # 保存基因列表
        genes_path = self.output_dir / "genes.txt"
        with open(genes_path, 'w') as f:
            for gene in self.common_genes:
                f.write(f"{gene}\n")
        logger.info(f"基因列表保存到: {genes_path}")
        
        # 保存患者ID列表
        patients_path = self.output_dir / "patients.txt"
        with open(patients_path, 'w') as f:
            for patient_id in self.patient_ids:
                f.write(f"{patient_id}\n")
        logger.info(f"患者ID列表保存到: {patients_path}")
        
        # 保存基因映射（用于参考）
        if self.gene_id_to_name:
            mapping_path = self.output_dir / "gene_mapping.txt"
            with open(mapping_path, 'w') as f:
                f.write("gene_id\tgene_name\n")
                for gene_id, gene_name in sorted(self.gene_id_to_name.items()):
                    f.write(f"{gene_id}\t{gene_name}\n")
            logger.info(f"基因映射保存到: {mapping_path}")
        
        # 保存处理摘要
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"TCGA-ESCA多组学数据预处理摘要\n")
            f.write(f"================================\n")
            f.write(f"处理时间: {pd.Timestamp.now()}\n")
            f.write(f"患者数量: {len(self.patient_ids)}\n")
            f.write(f"共同基因数量: {len(self.common_genes)}\n")
            f.write(f"张量形状: ({len(self.common_genes)}, {len(self.patient_ids)}, 3)\n")
            f.write(f"通道顺序: 0=RNA-seq (TPM), 1=CNV (copy number), 2=Methylation (beta)\n")
            f.write(f"\n输入文件统计:\n")
            f.write(f"  RNA-seq样本: {len(self.rna_data)}\n")
            f.write(f"  CNV样本: {len(self.cnv_data)}\n")
            f.write(f"  甲基化样本: {len(self.meth_data)}\n")
            f.write(f"\n基因统计:\n")
            f.write(f"  RNA基因数: {len(self.rna_genes)}\n")
            f.write(f"  CNV基因数: {len(self.cnv_genes)}\n")
            f.write(f"  甲基化基因数: {len(self.meth_genes)}\n")
            f.write(f"  基因映射数: {len(self.gene_id_to_name)}\n")
            f.write(f"\n输出文件:\n")
            f.write(f"  1. tensor.npy - 三维张量数据\n")
            f.write(f"  2. genes.txt - 基因名称列表\n")
            f.write(f"  3. patients.txt - 患者ID列表\n")
            if self.gene_id_to_name:
                f.write(f"  4. gene_mapping.txt - 基因ID到基因名映射\n")
            f.write(f"  5. summary.txt - 处理摘要\n")
        logger.info(f"处理摘要保存到: {summary_path}")
    
    def run(self) -> np.ndarray:
        """
        运行完整的预处理流程
        
        Returns:
            三维张量
        """
        try:
            # 1. 加载所有数据
            self.load_all_data()
            
            # 2. 寻找共同患者和基因
            self.find_common_patients_and_genes()
            
            # 3. 构建张量
            tensor = self.build_tensor()
            
            # 4. 保存结果
            self.save_results(tensor)
            
            logger.info("预处理流程完成!")
            return tensor
            
        except Exception as e:
            logger.error(f"预处理流程失败: {e}")
            raise


def main():
    """
    主函数：示例用法
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='TCGA-ESCA多组学数据预处理 (改进版)')
    parser.add_argument('--rna_dir', type=str, required=True,
                       help='RNA-seq文件目录')
    parser.add_argument('--cnv_dir', type=str, required=True,
                       help='CNV文件目录')
    parser.add_argument('--meth_dir', type=str, required=True,
                       help='甲基化文件目录')
    parser.add_argument('--probe_annotation', type=str, required=True,
                       help='探针注释文件路径')
    parser.add_argument('--gene_mapping', type=str, default=None,
                       help='基因ID到基因名的映射文件（可选）')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录 (默认: ./output)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 创建预处理器并运行
    preprocessor = TCGAESCAPreprocessor(
        rna_dir=args.rna_dir,
        cnv_dir=args.cnv_dir,
        meth_dir=args.meth_dir,
        probe_annotation_file=args.probe_annotation,
        gene_mapping_file=args.gene_mapping,
        output_dir=args.output_dir
    )
    
    try:
        tensor = preprocessor.run()
        print(f"\n预处理成功完成!")
        print(f"张量形状: {tensor.shape}")
        print(f"输出文件保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"\n预处理失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
