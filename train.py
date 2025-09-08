
"""
EigenPlaces训练脚本

该脚本实现了EigenPlaces的训练过程，主要特点：
1. 使用多个数据组（groups），每组包含不同视角的图像
2. 每个epoch轮换使用不同的数据组进行训练
3. 使用CosFace损失函数增强特征的判别性
4. 支持混合精度训练以提高效率
"""

import sys
import torch
import logging
import torchmetrics
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as tfm

import test
import util
import parser
import commons
import cosface_loss
import augmentations
from eigenplaces_model import eigenplaces_network
from datasets.test_dataset import TestDataset
from datasets.eigenplaces_dataset import EigenPlacesDataset

torch.backends.cudnn.benchmark = True  # 启用cudnn优化以提升训练速度

# 解析命令行参数和初始化
args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)           # 设置随机种子确保可重复性
commons.setup_logging(output_folder, console="debug")  # 设置日志
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### 模型初始化
# 创建EigenPlaces网络模型
model = eigenplaces_network.GeoLocalizationNet_(args.backbone, args.fc_output_dim)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

# 如果指定了预训练模型路径，则加载模型权重
if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

# 将模型移动到指定设备并设置为训练模式
model = model.to(args.device).train()

#### 优化器和损失函数
# 使用交叉熵损失函数（与CosFace分类器配合使用）
criterion = torch.nn.CrossEntropyLoss()
# 使用Adam优化器优化模型参数
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#### 数据集和分类器初始化
# 创建多个数据组，每组包含不同视角（0度和90度）的图像
# EigenPlaces的核心思想：使用多视角数据增强模型的视角鲁棒性
groups = [EigenPlacesDataset(
        args.train_dataset_folder, M=args.M, N=args.N, focal_dist=args.focal_dist,
        current_group=n//2,                    # 数据组编号
        min_images_per_class=args.min_images_per_class, 
        angle=[0, 90][n % 2],                  # 交替使用0度和90度视角
        visualize_classes=args.visualize_classes)
    for n in range(args.groups_num * 2)  # 每个组有两个视角，所以总数是groups_num * 2
]

# 为每个数据组创建独立的CosFace分类器
# 每个分类器的输出类别数等于对应数据组中的类别数
classifiers = [cosface_loss.MarginCosineProduct(
    args.fc_output_dim, len(group), s=args.s, m=args.m) for group in groups]

# 为每个分类器创建独立的优化器
classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

# GPU数据增强管道
# 在GPU上进行数据增强以提高效率
gpu_augmentation = tfm.Compose([
    # 设备无关的颜色抖动，增加模型对光照变化的鲁棒性
    augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                            contrast=args.contrast,
                                            saturation=args.saturation,
                                            hue=args.hue),
    # 随机裁剪和缩放，增加模型对尺度变化的鲁棒性
    augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                  scale=[1-args.random_resized_crop, 1]),
    # ImageNet标准化
    tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following "
             f"number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following "
             f"number of images {[g.get_images_num() for g in groups]}")

logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

val_ds = TestDataset(f"{args.val_dataset_folder}")
logging.info(f"Validation set: {val_ds}")

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, \
        best_val_recall1, start_epoch_num = \
            util.resume_train(args, output_folder, model, model_optimizer, 
                              classifiers, classifiers_optimizers)

    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} " +
                 f"from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
logging.info("Start training ...")

scaler = torch.cuda.amp.GradScaler()

for epoch_num in range(start_epoch_num, args.epochs_num):
    
    #### Train
    epoch_start_time = datetime.now()

    def get_iterator(groups, classifiers, classifiers_optimizers, batch_size, g_num):
        assert len(groups) == len(classifiers) == len(classifiers_optimizers)
        classifiers[g_num] = classifiers[g_num].to(args.device)
        util.move_to_device(classifiers_optimizers[g_num], args.device)
        return commons.InfiniteDataLoader(groups[g_num], num_workers=args.num_workers,
                                          batch_size=batch_size, shuffle=True,
                                          pin_memory=(args.device == "cuda"), drop_last=True)
    
    # Select classifier and dataloader according to epoch
    current_dataset_num = (epoch_num % args.groups_num) * 2
    
    iterators = []
    for i in range(2):
        iterators.append(get_iterator(groups, classifiers, classifiers_optimizers,
                                         args.batch_size, current_dataset_num + i))
    lateral_loss = torchmetrics.MeanMetric()
    frontal_loss = torchmetrics.MeanMetric()
    
    model = model.train()
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
        model_optimizer.zero_grad()
    
        #### EigenPlace ITERATION ####
        for i in range(2):
            classifiers_optimizers[current_dataset_num + i].zero_grad()
            
            images, targets, _ = next(iterators[i])
            images, targets = images.to(args.device), targets.to(args.device)
            with torch.cuda.amp.autocast():
                images = gpu_augmentation(images)
                descriptors = model(images)
                output = classifiers[current_dataset_num + i](descriptors, targets)
                loss = criterion(output, targets)
                if i == 0:
                    loss *= args.lambda_lat
                else:
                    loss *= args.lambda_front
            del images, output
            scaler.scale(loss).backward()
            scaler.step(classifiers_optimizers[current_dataset_num + i])
            if i == 0:
                lateral_loss.update(loss.detach().cpu())
            else:
                frontal_loss.update(loss.detach().cpu())
            del loss
            #######################
    
        scaler.step(model_optimizer)
        scaler.update()
    
    for i in range(2):
        classifiers[current_dataset_num + i] = classifiers[current_dataset_num + i].cpu()
        util.move_to_device(classifiers_optimizers[current_dataset_num + i], "cpu")
    
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]} - "
                  f"group {current_dataset_num} lateral_loss = {lateral_loss.compute():.4f} - "
                  f"group {current_dataset_num + 1} frontal_loss = {frontal_loss.compute():.4f}")
    
    #### Evaluation
    recalls, recalls_str = test.test(args, val_ds, model, batchify=True)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, output_folder)

logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model_ on test set v1
best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")
model.load_state_dict(best_model_state_dict)

test_ds = TestDataset(f"{args.test_dataset_folder}")
recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str}")

logging.info("Experiment finished (without any errors)")


