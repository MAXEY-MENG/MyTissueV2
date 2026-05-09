"""
多模态数据集类
用于加载组学数据、病理特征和生存数据
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
import os


class MultiModalDataset(Dataset):
    """
    多模态数据集类
    
    加载以下数据：
    1. 组学张量 (tensor.npy): 形状 (G, P, 3)，其中 G=17472(基因数), P=184(患者数)
    2. 患者ID列表 (patients.txt)
    3. 病理特征 (.pt文件): 每个患者一个文件，形状 (N, 1024)
    4. 生存数据 (CSV): 包含 patient_id, time, event 三列
    
    返回: (path_features, omics_features, time, event, patient_id)
    """
    
    def __init__(
        self,
        tensor_path: str,
        patients_path: str,
        path_feature_dir: str,
        survival_path: str,
        patient_ids: Optional[List[str]] = None
    ):
        """
        初始化数据集
        
        Args:
            tensor_path: 组学张量文件路径 (如 ./processed/tensor.npy)
            patients_path: 患者ID列表文件路径 (如 ./processed/patients.txt)
            path_feature_dir: 病理特征文件目录
            survival_path: 生存数据CSV文件路径
            patient_ids: 可选，指定要使用的患者ID子集。如果为None，则使用所有患者
        """
        super().__init__()
        
        # 加载组学张量
        self.tensor = np.load(tensor_path)  # 形状: (G, P, 3)
        
        # 加载患者ID列表
        with open(patients_path, 'r') as f:
            all_patients = [line.strip() for line in f if line.strip()]
        
        # 加载生存数据
        survival_df = pd.read_csv(survival_path)
        
        # 确保生存数据中的patient_id是字符串类型
        survival_df['patient_id'] = survival_df['patient_id'].astype(str)
        
        # 创建患者ID到索引的映射
        self.patient_to_idx = {pid: idx for idx, pid in enumerate(all_patients)}
        
        # 确定要使用的患者ID
        if patient_ids is None:
            self.patient_ids = all_patients
        else:
            # 确保指定的患者ID都存在
            valid_patients = []
            for pid in patient_ids:
                if pid in self.patient_to_idx:
                    valid_patients.append(pid)
                else:
                    print(f"警告: 患者ID {pid} 不在患者列表中，已跳过")
            self.patient_ids = valid_patients
        
        # 过滤生存数据，只保留我们需要的患者
        self.survival_data = survival_df[survival_df['patient_id'].isin(self.patient_ids)].copy()
        
        # 创建患者ID到生存数据的映射
        self.survival_dict = {}
        for _, row in self.survival_data.iterrows():
            pid = str(row['patient_id'])
            self.survival_dict[pid] = {
                'time': float(row['time']),
                'event': int(row['event'])
            }
        
        # 检查是否有患者缺少生存数据
        missing_survival = [pid for pid in self.patient_ids if pid not in self.survival_dict]
        if missing_survival:
            print(f"警告: {len(missing_survival)} 个患者缺少生存数据: {missing_survival[:5]}...")
        
        # 病理特征目录
        self.path_feature_dir = path_feature_dir
        
        # 构建病理特征文件到患者ID的映射
        self.path_file_mapping = self._build_path_file_mapping()
        
        # 构建患者索引映射：确保 patient_ids 和 tensor 列索引一一对应
        # 只保留有生存数据的患者
        self.valid_patient_ids = []
        self.indices = []
        for pid in self.patient_ids:
            if pid in self.survival_dict:  # 只包含有生存数据的患者
                self.valid_patient_ids.append(pid)
                self.indices.append(self.patient_to_idx[pid])
        
        # 用 valid_patient_ids 替换 patient_ids
        self.patient_ids = self.valid_patient_ids
    
    def _build_path_file_mapping(self) -> Dict[str, str]:
        """
        构建病理特征文件到患者ID的映射
        
        支持以下文件名格式:
        - 简单格式: TCGA-2H-A9GF-01Z.pt
        - 复杂格式: TCGA-2H-A9GF-01Z-00-DX1.FA1016AF-3FE3-45DC-A77B-F1ACC2B33B2A.pt
        
        提取患者ID部分: TCGA-2H-A9GF-01Z
        
        策略: 优先选择复杂格式的文件，如果存在的话
        """
        mapping = {}
        
        if not os.path.exists(self.path_feature_dir):
            print(f"警告: 病理特征目录不存在: {self.path_feature_dir}")
            return mapping
        
        # 首先收集所有可能的映射
        all_mappings = {}
        
        # 扫描目录中的所有.pt文件
        for filename in os.listdir(self.path_feature_dir):
            if filename.endswith('.pt'):
                # 提取患者ID部分
                # 文件名格式: TCGA-2H-A9GF-01Z-00-DX1.FA1016AF-3FE3-45DC-A77B-F1ACC2B33B2A.pt
                # 患者ID: TCGA-2H-A9GF-01Z
                
                # 方法1: 按'-'分割，取前4部分
                parts = filename.split('-')
                if len(parts) >= 4:
                    # 前4部分: TCGA, 2H, A9GF, 01Z
                    patient_id = '-'.join(parts[:4])
                    
                    # 检查第四部分是否以数字开头（如01Z）
                    if len(parts[3]) >= 3 and parts[3][:2].isdigit():
                        patient_id = '-'.join(parts[:4])
                    else:
                        # 如果第四部分不是样本类型代码，可能需要调整
                        patient_id = '-'.join(parts[:4])
                    
                    # 存储映射，但可能有多对一的情况
                    if patient_id not in all_mappings:
                        all_mappings[patient_id] = []
                    all_mappings[patient_id].append(filename)
        
        # 为每个患者选择最佳文件
        for patient_id, filenames in all_mappings.items():
            if patient_id in self.patient_ids:
                # 优先选择复杂格式的文件（文件名更长的）
                if len(filenames) > 1:
                    # 选择最长的文件名（通常是复杂格式）
                    selected = max(filenames, key=len)
                    mapping[patient_id] = selected
                    # print(f"多个文件，选择: {patient_id} -> {selected}")
                else:
                    mapping[patient_id] = filenames[0]
        
        # 检查映射完整性
        missing_patients = []
        for pid in self.patient_ids:
            if pid not in mapping and pid in self.survival_dict:
                missing_patients.append(pid)
        
        if missing_patients:
            print(f"警告: {len(missing_patients)} 个患者缺少病理特征文件: {missing_patients[:5]}...")
            print("尝试其他匹配方法...")
            
            # 尝试更宽松的匹配
            for pid in missing_patients:
                # 尝试匹配文件名中包含患者ID的文件
                for filename in os.listdir(self.path_feature_dir):
                    if filename.endswith('.pt') and pid in filename:
                        mapping[pid] = filename
                        print(f"找到匹配: {pid} -> {filename}")
                        break
        
        print(f"病理特征文件映射构建完成: {len(mapping)}/{len(self.patient_ids)} 个患者有病理特征")
        
        # 显示前几个映射
        print("前5个映射:")
        for i, (pid, filename) in enumerate(list(mapping.items())[:5]):
            print(f"  {pid} -> {filename}")
        
        return mapping
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, int, str]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            path_features: 病理特征张量，形状 (N, 1024)
            omics_features: 组学特征张量，形状 (17472, 3)
            time: 生存时间
            event: 事件指示器 (1=死亡, 0=删失)
            patient_id: 患者ID
        """
        # 获取患者索引
        patient_idx = self.indices[idx]
        
        # 获取患者ID
        patient_id = self.patient_ids[idx]
        
        # 提取组学特征: 所有基因，当前患者，所有通道
        omics_features = self.tensor[:, patient_idx, :]  # 形状: (17472, 3)
        omics_features = torch.from_numpy(omics_features).float()
        
        # 加载病理特征
        if patient_id in self.path_file_mapping:
            path_filename = self.path_file_mapping[patient_id]
            path_file = os.path.join(self.path_feature_dir, path_filename)
            try:
                path_features = torch.load(path_file)  # 形状: (N, 1024)
                # 确保是浮点类型
                path_features = path_features.float()
            except Exception as e:
                print(f"警告: 加载病理特征文件 {path_file} 失败: {e}，使用零张量")
                path_features = torch.zeros((1, 1024)).float()
        else:
            # 如果文件不存在，创建零张量作为占位符
            print(f"警告: 患者 {patient_id} 没有病理特征文件，使用零张量")
            path_features = torch.zeros((1, 1024)).float()
        
        # 获取生存数据
        survival_info = self.survival_dict[patient_id]
        time = survival_info['time']
        event = survival_info['event']
        
        return path_features, omics_features, time, event, patient_id
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, float, int, str]]
                  ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        自定义collate函数，处理变长的病理特征
        
        Args:
            batch: 批次数据列表
            
        Returns:
            path_list: 病理特征列表，每个元素形状 (N_i, 1024)
            omics_batch: 组学特征张量，形状 (batch_size, 17472, 3)
            time_batch: 生存时间张量，形状 (batch_size,)
            event_batch: 事件指示器张量，形状 (batch_size,)
            patient_ids: 患者ID列表
        """
        path_list = []
        omics_list = []
        time_list = []
        event_list = []
        patient_ids = []
        
        for path_features, omics_features, time, event, pid in batch:
            path_list.append(path_features)
            omics_list.append(omics_features)
            time_list.append(time)
            event_list.append(event)
            patient_ids.append(pid)
        
        # 组学特征可以堆叠，因为维度固定
        omics_batch = torch.stack(omics_list, dim=0)  # (batch_size, 17472, 3)
        
        # 生存数据转换为张量
        time_batch = torch.tensor(time_list, dtype=torch.float32)
        event_batch = torch.tensor(event_list, dtype=torch.float32)
        
        return path_list, omics_batch, time_batch, event_batch, patient_ids


def test_dataset():
    """测试数据集类"""
    import tempfile
    import shutil
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建模拟数据
        G, P = 100, 10  # 简化尺寸用于测试
        tensor = np.random.randn(G, P, 3).astype(np.float32)
        tensor_path = os.path.join(temp_dir, "tensor.npy")
        np.save(tensor_path, tensor)
        
        # 使用真实的TCGA患者ID格式
        patients = [
            "TCGA-2H-A9GF-01Z",
            "TCGA-2H-A9GG-01Z", 
            "TCGA-2H-A9GH-01Z",
            "TCGA-2H-A9GI-01Z",
            "TCGA-2H-A9GJ-01Z",
            "TCGA-2H-A9GK-01Z",
            "TCGA-2H-A9GL-01Z",
            "TCGA-2H-A9GM-01Z",
            "TCGA-2H-A9GN-01Z",
            "TCGA-2H-A9GO-01Z"
        ]
        patients_path = os.path.join(temp_dir, "patients.txt")
        with open(patients_path, 'w') as f:
            for pid in patients:
                f.write(f"{pid}\n")
        
        path_feature_dir = os.path.join(temp_dir, "path_features")
        os.makedirs(path_feature_dir, exist_ok=True)
        
        # 创建模拟病理特征文件 - 使用复杂文件名格式
        for i, pid in enumerate(patients):
            N = np.random.randint(10, 100)  # 随机patch数
            path_features = torch.randn(N, 1024)
            
            # 使用复杂文件名格式
            complex_filename = f"{pid}-00-DX1.FA1016AF-3FE3-45DC-A77B-F1ACC2B33B2A.pt"
            torch.save(path_features, os.path.join(path_feature_dir, complex_filename))
            
            # 也创建一个简单格式的文件用于测试
            simple_filename = f"{pid}.pt"
            torch.save(path_features, os.path.join(path_feature_dir, simple_filename))
        
        # 创建模拟生存数据
        survival_data = []
        for i, pid in enumerate(patients):
            survival_data.append({
                'patient_id': pid,
                'time': np.random.uniform(10, 100),
                'event': np.random.choice([0, 1])
            })
        
        survival_df = pd.DataFrame(survival_data)
        survival_path = os.path.join(temp_dir, "survival.csv")
        survival_df.to_csv(survival_path, index=False)
        
        print("=" * 60)
        print("测试数据集类 - 复杂文件名格式")
        print("=" * 60)
        
        # 创建数据集
        dataset = MultiModalDataset(
            tensor_path=tensor_path,
            patients_path=patients_path,
            path_feature_dir=path_feature_dir,
            survival_path=survival_path
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试单个样本
        path_features, omics_features, time, event, pid = dataset[0]
        print(f"\n样本测试:")
        print(f"病理特征形状: {path_features.shape}")
        print(f"组学特征形状: {omics_features.shape}")
        print(f"生存时间: {time}, 事件: {event}, 患者ID: {pid}")
        
        # 测试collate函数
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=MultiModalDataset.collate_fn
        )
        
        batch = next(iter(dataloader))
        path_list, omics_batch, time_batch, event_batch, patient_ids = batch
        print(f"\n批次数据测试:")
        print(f"病理特征列表长度: {len(path_list)}")
        print(f"组学批次形状: {omics_batch.shape}")
        print(f"时间批次形状: {time_batch.shape}")
        print(f"事件批次形状: {event_batch.shape}")
        print(f"患者ID: {patient_ids}")
        
        # 测试文件映射
        print(f"\n文件映射测试:")
        print(f"映射数量: {len(dataset.path_file_mapping)}")
        for pid, filename in list(dataset.path_file_mapping.items())[:3]:
            print(f"  {pid} -> {filename}")
        
        print("\n" + "=" * 60)
        print("测试通过! 数据集类支持复杂文件名格式。")
        print("=" * 60)
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_dataset()