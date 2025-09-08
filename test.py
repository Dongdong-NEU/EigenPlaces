
"""
EigenPlaces测试/评估模块

该模块实现了视觉地点识别任务的评估功能，主要步骤：
1. 提取数据库图像和查询图像的特征描述符
2. 使用FAISS进行高效的最近邻搜索
3. 计算Recall@K指标评估模型性能

Recall@K指标说明：
- Recall@1: 最相似的1个检索结果中包含正确位置的查询比例
- Recall@5: 最相似的5个检索结果中包含正确位置的查询比例
- 以此类推...
"""

import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset


# 计算的召回率指标：R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module, batchify : bool = False) -> Tuple[np.ndarray, str]:
    """
    在给定数据集上评估模型性能
    
    该函数执行完整的视觉地点识别评估流程：
    1. 提取所有图像的特征描述符
    2. 构建FAISS索引进行高效相似度搜索
    3. 计算Recall@K指标
    
    Args:
        args (Namespace): 包含评估参数的命名空间
        eval_ds (Dataset): 评估数据集，包含数据库图像和查询图像
        model (torch.nn.Module): 要评估的模型
        batchify (bool): 是否对查询图像进行批处理，默认为False
        
    Returns:
        Tuple[np.ndarray, str]: (召回率数组, 格式化的召回率字符串)
    """
    
    # 设置模型为评估模式
    model = model.eval()
    
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        
        # 1. 提取数据库图像的特征描述符
        # 数据库图像用于构建检索索引
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        
        # 预分配存储所有描述符的数组
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        
        # 批量处理数据库图像
        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model(images.to(args.device))  # 前向传播获取特征
            descriptors = descriptors.cpu().numpy()      # 转换为numpy数组
            all_descriptors[indices.numpy(), :] = descriptors  # 存储到对应位置
        
        logging.debug("Extracting queries descriptors for evaluation/testing")
        
        # 2. 提取查询图像的特征描述符
        # 根据batchify参数决定批处理大小
        if batchify:
            queries_infer_batch_size = args.infer_batch_size
        else:
            queries_infer_batch_size = 1  # 逐个处理查询图像
            
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        
        # 批量处理查询图像
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))  # 前向传播获取特征
            descriptors = descriptors.cpu().numpy()      # 转换为numpy数组
            all_descriptors[indices.numpy(), :] = descriptors  # 存储到对应位置
    
    # 3. 分离查询描述符和数据库描述符
    queries_descriptors = all_descriptors[eval_ds.database_num:]  # 查询图像的描述符
    database_descriptors = all_descriptors[:eval_ds.database_num] # 数据库图像的描述符
    
    # 4. 使用FAISS构建高效的最近邻搜索索引
    # IndexFlatL2使用L2距离进行精确搜索
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)  # 将数据库描述符添加到索引中
    del database_descriptors, all_descriptors  # 释放内存
    
    logging.debug("Calculating recalls")
    # 5. 执行最近邻搜索
    # 对每个查询，返回最相似的max(RECALL_VALUES)个数据库图像的索引
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))
    
    # 6. 计算Recall@K指标
    # 对于每个查询，检查预测结果是否包含正确的位置
    positives_per_query = eval_ds.get_positives()  # 获取每个查询的正确位置列表
    recalls = np.zeros(len(RECALL_VALUES))
    
    for query_index, preds in enumerate(predictions):
        # 对于当前查询，检查不同K值下的召回情况
        for i, n in enumerate(RECALL_VALUES):
            # 检查前n个预测结果中是否包含正确位置
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1  # 如果找到，则该K及更大K值的召回都+1
                break             # 找到后就跳出，避免重复计算
                
    # 7. 将召回率转换为百分比并格式化输出
    recalls = recalls / eval_ds.queries_num * 100  # 除以查询总数得到百分比
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    return recalls, recalls_str
