#!/usr/bin/env python3
"""
测试数据集加载
"""

import torch
from data.dataset import MultiModalDataset

def test_dataset():
    """测试数据集加载"""
    print("测试数据集加载...")
    
    try:
        # 只测试前5个患者
        with open('./processed/patients.txt', 'r') as f:
            patients = [line.strip() for line in f if line.strip()]
        
        print(f'患者总数: {len(patients)}')
        test_patients = patients[:5]
        print(f'测试前5个患者: {test_patients}')
        
        dataset = MultiModalDataset(
            tensor_path='./processed/tensor.npy',
            patients_path='./processed/patients.txt',
            path_feature_dir='./path_features/',
            survival_path='./data/survival.csv',
            patient_ids=test_patients
        )
        
        print(f'数据集大小: {len(dataset)}')
        
        if len(dataset) > 0:
            path_features, omics_features, time, event, patient_id = dataset[0]
            print(f'样本测试成功:')
            print(f'  患者ID: {patient_id}')
            print(f'  病理特征形状: {path_features.shape}')
            print(f'  组学特征形状: {omics_features.shape}')
            print(f'  生存时间: {time}, 事件: {event}')
            
            # 检查是否有零张量警告
            if torch.all(path_features == 0):
                print('警告: 病理特征为零张量')
            else:
                print('病理特征正常')
        
        print('数据集测试通过!')
        return True
        
    except Exception as e:
        print(f'数据集测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset()
    exit(0 if success else 1)