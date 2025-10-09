import torch
import os
import sys
import argparse
import logging
from datetime import datetime
from pmodel.runner.tracer import BaseTracer

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons
from eigenplaces_model import eigenplaces_network
from eigenplaces_model.eigenplaces_network import convert_to_no_sqrt_dla_compatible

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Trace EigenPlaces model")
    
    # 模型骨架参数
    parser.add_argument("--backbone", type=str, default="ResNet50",
                       choices=["ResNet18", "ResNet50", "ResNet101", "ResNet152", "VGG16"],
                       help="Backbone architecture")
    
    # 模型输出维度
    parser.add_argument("--fc_output_dim", type=int, default=2048,
                       help="Output dimension of descriptors")
    
    # 模型路径
    parser.add_argument("--resume_model", type=str, required=True,
                       help="Path to model checkpoint or 'torchhub' for pretrained model")
    
    # 输入图像尺寸
    parser.add_argument("--input_size", type=int, nargs=2, default=[512, 512],
                       help="Input image size [height, width]")
    
    # batch size 大小
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for tracing")
    
    # trace输出目录
    parser.add_argument("--trace_dir", type=str, default="./results",
                       help="Directory to save trace results")
    
    # 模型名称
    parser.add_argument("--model_name", type=str, default="eigenplaces",
                       help="Name for the traced model")
    
    # 指定设备
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device for model tracing")
    
    # 完全DLA兼容选项 (无Sqrt算子)
    parser.add_argument("--no_sqrt", action="store_true",
                       help="Convert model to completely DLA compatible version (avoids ReduceSum/ReduceL2/Sqrt operators)")
    
    parser.add_argument("--use_reciprocal", action="store_true", default=True,
                       help="Use reciprocal method (pow(-0.5)) instead of pow(0.5) for sqrt replacement")
    
    return parser.parse_args()

def load_model(args):
    """加载EigenPlaces模型"""
    if args.resume_model == "torchhub":
        logging.info("从PyTorch Hub加载预训练模型")
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                              backbone=args.backbone, fc_output_dim=args.fc_output_dim)
    else:
        logging.info(f"从 {args.resume_model} 加载模型")
        model = eigenplaces_network.GeoLocalizationNet_(args.backbone, args.fc_output_dim)
        
        if not os.path.exists(args.resume_model):
            raise FileNotFoundError(f"模型文件未找到: {args.resume_model}")
            
        # 加载模型权重
        checkpoint = torch.load(args.resume_model, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
            else:
                model_state_dict = checkpoint
        else:
            model_state_dict = checkpoint
            
        model.load_state_dict(model_state_dict)
    
    # 如果需要DLA兼容，转换模型
    # if args.no_sqrt:
    #     logging.info("转换模型为完全DLA兼容版本 (无Sqrt)...")
    #     model = convert_to_no_sqrt_dla_compatible(model, use_reciprocal=args.use_reciprocal)
    #     logging.info("无Sqrt模型转换完成")
    
    return model

class EigenPlacesModel(torch.nn.Module):
    """
    EigenPlaces模型的包装类 用于trace
    由于BaseTracer需要nn.Module 所以需要将模型包装一下
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def main():

    args = parse_arguments()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'trace_eigenplaces_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    try:
        # 加载模型
        logging.info("开始加载模型...")
        model = load_model(args)
        logging.info(f"成功加载模型: {args.backbone} 输出维度 {args.fc_output_dim}")
        
        # 移动模型到指定设备
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()  # 设置为评估模式
        
        # 包装模型
        # wrapped_model = EigenPlacesModel(model)
        
        # 创建输入数据
        logging.info(f"创建输入数据: batch_size={args.batch_size}, size={args.input_size}")
        input_tensor = torch.randn(args.batch_size, 3, args.input_size[0], args.input_size[1], device=device)
        
        # 测试模型前向传播
        logging.info("测试模型前向传播...")
        with torch.no_grad():
            output = model(input_tensor)
            logging.info(f"模型输出形状: {output.shape}")
            logging.info(f"模型输出类型: {output.dtype}")
        
        # 创建tracer并执行trace
        logging.info("开始trace模型...")
        tracer = BaseTracer(model)
        
        # 确保trace目录存在
        os.makedirs(args.trace_dir, exist_ok=True)
        
        # 执行trace
        tracer.trace(
            trace_data=(input_tensor,), 
            trace_dir=args.trace_dir, 
            model_name=args.model_name
        )
        
        logging.info(f"成功完成模型trace！结果保存在: {args.trace_dir}")
        logging.info(f"模型名称: {args.model_name}")
        
    except Exception as e:
        logging.error(f"Trace过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
