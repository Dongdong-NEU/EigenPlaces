
"""
EigenPlaces评估脚本

该脚本用于评估训练好的EigenPlaces模型在测试数据集上的性能。
支持两种模型加载方式：
1. 从torch.hub加载预训练模型（设置--resume_model torchhub）
2. 从本地路径加载自训练模型（设置--resume_model 模型路径）

评估指标包括Recall@1, Recall@5, Recall@10, Recall@20等。
"""

import sys
import torch
import logging
import multiprocessing
from datetime import datetime

import test
import parser
import commons
from datasets.test_dataset import TestDataset
from eigenplaces_model import eigenplaces_network

torch.backends.cudnn.benchmark = True  # 启用cudnn优化以提升推理速度

# 解析命令行参数和初始化
args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)           # 设置随机种子确保可重复性
commons.setup_logging(output_folder, console="info")  # 设置日志级别为info
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")
logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

#### 模型加载和初始化
if args.resume_model == "torchhub":
    # 从PyTorch Hub加载预训练的EigenPlaces模型
    logging.info("Loading pretrained model from PyTorch Hub")
    model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                           backbone=args.backbone, fc_output_dim=args.fc_output_dim)
else:
    # 创建模型并加载本地权重
    model = eigenplaces_network.GeoLocalizationNet_(args.backbone, args.fc_output_dim)
    
    if args.resume_model is not None:
        logging.info(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)
    else:
        # 警告：使用随机初始化的权重进行评估
        logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                     "Evaluation will be computed using randomly initialized weights.")

# 将模型移动到指定设备
model = model.to(args.device)

# 加载测试数据集
test_ds = TestDataset(args.test_dataset_folder, queries_folder="queries",
                      positive_dist_threshold=args.positive_dist_threshold)

# 执行模型评估
recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str}")

