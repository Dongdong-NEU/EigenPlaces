import os
import torch
import logging
import argparse
from datetime import datetime

import sys
# os.path.abspath(__file__)) 表示当前文件的绝对路径
# os.path.dirname(os.path.abspath(__file__)) 表示当前文件的父目录
# os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 表示当前文件的父目录的父目录
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 表示将当前文件的父目录的父目录添加到系统路径中
# 这样就可以在当前文件中导入父目录中的模块

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons
from eigenplaces_model import eigenplaces_network
from eigenplaces_model.eigenplaces_network import  convert_to_no_sqrt_dla_compatible

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Export EigenPlaces model to ONNX format")
    
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
    
    # 导出路径
    parser.add_argument("--output_path", type=str, default="eigenplaces_model.onnx",
                       help="Output path for ONNX model")
    # 输入图像尺寸
    parser.add_argument("--input_size", type=int, nargs=2, default=[512, 512],
                       help="Input image size [height, width]")
    # batch size 大小
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for ONNX model")
    
    # 导出ONNX版本
    parser.add_argument("--opset_version", type=int, default=12,
                       help="ONNX opset version")
    # 动态轴
    parser.add_argument("--dynamic_axes", action="store_true",
                       help="Enable dynamic batch size")
    # 是否简化ONNX模型
    parser.add_argument("--simplify", action="store_true",
                       help="Simplify ONNX model using onnx-simplifier")
    
    # 验证选项
    parser.add_argument("--verify", action="store_true",
                       help="Verify ONNX model by comparing outputs")
    # 指定设备
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for model inference")
    
    # 完全DLA兼容选项 (无Sqrt算子)
    parser.add_argument("--no_sqrt", action="store_true",
                       help="Convert model to completely DLA compatible version (avoids ReduceSum/ReduceL2/Sqrt operators)")
    
    parser.add_argument("--use_reciprocal", action="store_true", default=True,
                       help="Use reciprocal method (pow(-0.5)) instead of pow(0.5) for sqrt replacement")
    
    # BHWC输入格式支持
    parser.add_argument("--bhwc_input", action="store_true",
                       help="Export model with BHWC input format instead of BCHW")
    
    return parser.parse_args()

def load_model(args):
    """加载EigenPlaces模型"""
    if args.resume_model == "torchhub":
        logging.info("Loading pretrained model from PyTorch Hub")
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                              backbone=args.backbone, fc_output_dim=args.fc_output_dim)
    else:
        logging.info(f"Loading model from {args.resume_model}")
        model = eigenplaces_network.GeoLocalizationNet_(args.backbone, args.fc_output_dim)
        
        if not os.path.exists(args.resume_model):
            raise FileNotFoundError(f"Model file not found: {args.resume_model}")
            
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
    if args.no_sqrt:
        logging.info("Converting model to completely DLA compatible version (no Sqrt)...")
        model = convert_to_no_sqrt_dla_compatible(model, use_reciprocal=args.use_reciprocal)
        logging.info("No-Sqrt model conversion completed")

    
    return model

class ModelWithBHWCInput(torch.nn.Module):
    """
    包装模型以支持BHWC输入格式
    
    该包装类在模型前添加一个transpose操作，将BHWC格式的输入转换为BCHW格式，
    然后再送入原始模型进行推理。这样可以在不修改原模型结构的情况下，
    支持BHWC格式的输入数据。
    
    Args:
        model: 原始的PyTorch模型（期望BCHW输入）
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，格式为BHWC (batch, height, width, channels)
            
        Returns:
            模型输出（通常是描述符向量）
        """
        # 将BHWC转换为BCHW: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # 调用原始模型
        return self.model(x)

def export_to_onnx(model, args):
    """
    导出模型为ONNX格式
    把一个已经在 eval() 模式下的 PyTorch 模型,用一个"假输入"（dummy input）跑一遍前向,并把得到的静态计算图 + 权重导出成 .onnx 文件。
    
    Returns:
        model: 导出使用的模型（如果使用bhwc_input，则返回包装后的模型）
    """
    # 如果需要BHWC输入格式，包装模型
    if args.bhwc_input:
        logging.info("Wrapping model to support BHWC input format")
        model = ModelWithBHWCInput(model)
    
    # 切到推理模式（冻结 BN 统计、关闭 Dropout 等）。导出 ONNX 前必须做，保证导出的行为与推理一致
    model.eval()
    
    # dummy_input：一个"假输入"，只用于跟踪模型的前向，构建图用。
    # 根据是否使用BHWC格式创建不同形状的输入
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    
    if args.bhwc_input:
        # BHWC格式: [batch, height, width, channels]
        dummy_input = torch.randn(args.batch_size, args.input_size[0], args.input_size[1], 3, device=device, dtype=dtype)
        logging.info(f"Using BHWC input format: {dummy_input.shape}")
    else:
        # BCHW格式: [batch, channels, height, width]
        dummy_input = torch.randn(args.batch_size, 3, args.input_size[0], args.input_size[1], device=device, dtype=dtype)
        logging.info(f"Using BCHW input format: {dummy_input.shape}")
    
    # 设置动态轴，dynamic_axes：告诉 ONNX 哪些维度是可变的（"动态形状"）
    dynamic_axes = None
    if args.dynamic_axes:
        if args.bhwc_input:
            # BHWC格式: 输入形状为 [batch, height, width, channels]
            # 可变维度: 0: batch_size, 1: height, 2: width
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'height', 2: 'width'},
                'output': {0: 'batch_size'}
            }
        else:
            # BCHW格式: 输入形状为 [batch, channels, height, width]
            # 可变维度: 0: batch_size, 2: height, 3: width
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
    
    # 导出ONNX
    logging.info(f"Exporting model to {args.output_path}")
    logging.info(f"Input shape: {dummy_input.shape}")
    logging.info(f"ONNX opset version: {args.opset_version}")
    
    torch.onnx.export(
        model,
        dummy_input,
        args.output_path,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    logging.info(f"Successfully exported ONNX model to {args.output_path}")
    
    # 获取文件大小
    file_size = os.path.getsize(args.output_path) / (1024 * 1024)  # MB
    logging.info(f"ONNX model size: {file_size:.2f} MB")
    
    # 返回用于导出的模型（可能已被包装）
    return model

def simplify_onnx(onnx_path):
    """使用onnx-simplifier简化ONNX模型"""
    try:
        import onnx
        import onnxsim
        
        logging.info("Simplifying ONNX model...")
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        
        # 简化模型
        simplified_model, check = onnxsim.simplify(onnx_model)
        
        if check:
            # 保存简化后的模型
            simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
            onnx.save(simplified_model, simplified_path)
            logging.info(f"Simplified model saved to {simplified_path}")
            
            # 比较文件大小
            original_size = os.path.getsize(onnx_path) / (1024 * 1024)
            simplified_size = os.path.getsize(simplified_path) / (1024 * 1024)
            logging.info(f"Original size: {original_size:.2f} MB")
            logging.info(f"Simplified size: {simplified_size:.2f} MB")
            logging.info(f"Size reduction: {((original_size - simplified_size) / original_size * 100):.1f}%")
        else:
            logging.warning("Model simplification failed")
            
    except ImportError:
        logging.warning("onnx-simplifier not installed. Install with: pip install onnx-simplifier")

def verify_onnx_model(pytorch_model, onnx_path, args):
    """验证ONNX模型输出与PyTorch模型是否一致
    
    注意：pytorch_model在导出时可能已经被ModelWithBHWCInput包装过，
    因此这里需要使用相同格式的输入进行验证
    """
    try:
        import onnxruntime as ort
        import numpy as np
        
        logging.info("Verifying ONNX model...")
        
        # 创建测试输入
        # 注意：无论是否使用bhwc_input，这里的test_input格式都应该与
        # pytorch_model期望的输入格式一致（即ONNX模型的输入格式）
        if args.bhwc_input:
            # BHWC格式：pytorch_model已经是ModelWithBHWCInput包装后的模型
            test_input = torch.randn(args.batch_size, args.input_size[0], args.input_size[1], 3)
        else:
            # BCHW格式：pytorch_model是原始模型
            test_input = torch.randn(args.batch_size, 3, args.input_size[0], args.input_size[1])
        
        # PyTorch推理
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # ONNX推理
        ort_session = ort.InferenceSession(onnx_path)
        onnx_output = ort_session.run(None, {'input': test_input.numpy()})[0]
        
        # 比较输出
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        
        logging.info(f"Max difference: {max_diff:.6f}")
        logging.info(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            logging.info("✓ ONNX model verification PASSED")
        else:
            logging.warning(f"⚠ ONNX model verification FAILED (max diff: {max_diff})")
            
    except ImportError:
        logging.warning("onnxruntime not installed. Install with: pip install onnxruntime")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 检查输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 加载模型
        model = load_model(args)
        logging.info(f"Loaded model: {args.backbone} with output dim {args.fc_output_dim}")
        
        # 移动到CPU进行导出（ONNX导出通常在CPU上进行）
        model = model.cpu()
        
        # 导出ONNX（返回可能被包装过的模型）
        exported_model = export_to_onnx(model, args)
        
        # 简化模型（可选）
        if args.simplify:
            simplify_onnx(args.output_path)
        
        # 验证模型（可选）
        # 注意：这里使用exported_model而不是model，因为如果启用了bhwc_input，
        # exported_model是被ModelWithBHWCInput包装过的模型
        if args.verify:
            verify_onnx_model(exported_model, args.output_path, args)
            
        logging.info("ONNX export completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during ONNX export: {str(e)}")
        raise

if __name__ == "__main__":
    main()
