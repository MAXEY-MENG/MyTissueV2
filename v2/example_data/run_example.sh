#!/bin/bash
# TCGA-ESCA预处理脚本使用示例

echo "运行预处理脚本..."
python ../preprocess_tcga_esca_improved.py \
  --rna_dir ./rna \
  --cnv_dir ./cnv \
  --meth_dir ./meth \
  --probe_annotation ./illumina_epic_manifest.csv \
  --gene_mapping ./gene_mapping.csv \
  --output_dir ./output \
  --log_level INFO

echo ""
echo "查看输出文件:"
echo "  张量文件: ./output/tensor.npy"
echo "  基因列表: ./output/genes.txt"
echo "  患者列表: ./output/patients.txt"
echo "  处理摘要: ./output/summary.txt"
