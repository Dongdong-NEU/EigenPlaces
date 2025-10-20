import torch
import os
import sys
import argparse
import logging
import drinfer
from datetime import datetime
from pmodel.runner.tracer import BaseTracer

# 添加父目录到系统路径，以便导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons
from eigenplaces_model import eigenplaces_network
from eigenplaces_model.eigenplaces_network import convert_to_no_sqrt_dla_compatible

def parse_arguments():
    parser = argparse.ArgumentParser(description="Trace EigenPlaces Head (aggregation) only")
    
    parser.add_argument("--backbone", type=str, default="ResNet50",
                       choices=["ResNet18", "ResNet50", "ResNet101", "ResNet152", "VGG16"],
                       help="Backbone architecture")
    
    parser.add_argument("--fc_output_dim", type=int, default=2048,
                       help="Output dimension of descriptors")
    
    parser.add_argument("--resume_model", type=str, required=True,
                       help="Path to model checkpoint or 'torchhub' for pretrained model")
    
    parser.add_argument("--feature_size", type=int, nargs=2, default=[16, 16],
                       help="Feature map size [height, width] from backbone")
    
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for tracing")
    
    parser.add_argument("--trace_dir", type=str, default="./results",
                       help="Directory to save trace results")
    
    parser.add_argument("--model_name", type=str, default="eigenplaces_head",
                       help="Name for the traced model")
    
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device for model tracing")
    
    parser.add_argument("--no_sqrt", action="store_true",
                       help="Convert model to completely DLA compatible version (avoids ReduceSum/ReduceL2/Sqrt operators)")
    
    parser.add_argument("--use_reciprocal", action="store_true", default=True,
                       help="Use reciprocal method (pow(-0.5)) instead of pow(0.5) for sqrt replacement")
        
    parser.add_argument("--input_format", type=str, default="BCHW",
                       choices=["BCHW", "BHWC"],
                       help="Input tensor format: BCHW (default) or BHWC")
    
    return parser.parse_args()

def load_model(args):
    if args.resume_model == "torchhub":
        logging.info("从PyTorch Hub加载预训练模型")
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                              backbone=args.backbone, fc_output_dim=args.fc_output_dim)
    else:
        logging.info(f"从 {args.resume_model} 加载模型")
        model = eigenplaces_network.GeoLocalizationNet_(args.backbone, args.fc_output_dim)
        
        if not os.path.exists(args.resume_model):
            raise FileNotFoundError(f"模型文件未找到: {args.resume_model}")
            
        checkpoint = torch.load(args.resume_model, map_location='cpu')
        
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
    
    # if args.no_sqrt:
    #     logging.info("转换模型为完全DLA兼容版本 (无Sqrt)...")
    #     model = convert_to_no_sqrt_dla_compatible(model, use_reciprocal=args.use_reciprocal)
    #     logging.info("无Sqrt模型转换完成")
    
    return model

class EigenPlacesHead(torch.nn.Module):

    def __init__(self, aggregation_module):
        super().__init__()
        self.aggregation = aggregation_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.aggregation(x)

def verify_head_output(model, head, input_tensor, device):

    logging.info("\n" + "="*60)
    logging.info("验证head输出正确性...")
    logging.info("="*60)
    
    model.eval()
    head.eval()
    
    with torch.no_grad():
        backbone_output = model.backbone(input_tensor)
        logging.info(f"Backbone输出形状: {backbone_output.shape}")
        
        full_model_output = model.aggregation(backbone_output)
        logging.info(f"完整模型aggregation输出形状: {full_model_output.shape}")
        
        head_output = head(backbone_output)
        logging.info(f"独立Head输出形状: {head_output.shape}")
        
        diff = torch.abs(full_model_output - head_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        logging.info(f"\n输出差异:")
        logging.info(f"  最大差异: {max_diff:.10f}")
        logging.info(f"  平均差异: {mean_diff:.10f}")
        
        is_equal = torch.allclose(full_model_output, head_output, rtol=1e-5, atol=1e-7)
        
        if is_equal:
            logging.info(" 验证通过: Head输出与完整模型的aggregation输出一致!")
        else:
            logging.warning("警告: Head输出与完整模型的aggregation输出存在差异")
            logging.warning(f"   最大差异 {max_diff} 可能超出容差范围")
        
        return is_equal, backbone_output

def main():
    args = parse_arguments()
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'trace_eigenplaces_head_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    try:
        logging.info("开始加载完整模型...")
        model = load_model(args)
        logging.info(f"成功加载模型: {args.backbone} 输出维度 {args.fc_output_dim}")
        
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()
        
        logging.info("\n提取Head (aggregation) 模块...")
        # head = EigenPlacesHead(model.aggregation)
        head = model.aggregation
        head = head.to(device)
        head.eval()
        
        logging.info(f"\nHead结构:")
        logging.info(f"{head}")
        
        features_dim = eigenplaces_network.CHANNELS_NUM_IN_LAST_CONV[args.backbone]
        logging.info(f"\n特征图通道数: {features_dim}")
        
        logging.info(f"\n创建验证输入: batch_size={args.batch_size}")
        full_input = torch.randn(args.batch_size, 3, 512, 512, device=device)
        
        is_valid, backbone_output = verify_head_output(model, head, full_input, device)
        
        if not is_valid:
            logging.error("Head验证失败，停止trace")
            return
        
        logging.info(f"\n创建trace输入数据:")
        logging.info(f"  batch_size={args.batch_size}")
        logging.info(f"  channels={features_dim}")
        logging.info(f"  feature_size={args.feature_size}")
        
        head_input_tensor = torch.randn(
            args.batch_size, 
            features_dim, 
            args.feature_size[0], 
            args.feature_size[1], 
            device=device
        )
        
        logging.info("\n测试Head前向传播...")
        with torch.no_grad():
            output = head(head_input_tensor)
            logging.info(f"Head输出形状: {output.shape}")
            logging.info(f"Head输出类型: {output.dtype}")
            logging.info(f"Head输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        logging.info("\n开始trace Head模块...")
        tracer = BaseTracer(head)
        
        os.makedirs(args.trace_dir, exist_ok=True)
        
        tracer.trace(
            trace_data=(head_input_tensor,), 
            trace_dir=args.trace_dir, 
            model_name=args.model_name,
            runtime_dtype="half",
            trace_runtime_dtype=drinfer.MODEL_DATA_TYPE.MODEL_HALF,
            input_layouts={"input_data_0": "NHWC"}
            )
        
        logging.info(f"\n 成功完成Head trace! ")
        logging.info(f"结果保存在: {args.trace_dir}")
        logging.info(f"模型名称: {args.model_name}")
        
        logging.info("\n" + "="*60)
        logging.info("Trace总结:")
        logging.info("="*60)
        logging.info(f"模型类型: EigenPlaces Head (aggregation only)")
        logging.info(f"Backbone: {args.backbone}")
        logging.info(f"输入形状: (batch_size={args.batch_size}, channels={features_dim}, H={args.feature_size[0]}, W={args.feature_size[1]})")
        logging.info(f"输出形状: (batch_size={args.batch_size}, features={args.fc_output_dim})")
        logging.info(f"是否使用无Sqrt版本: {args.no_sqrt}")
        if args.no_sqrt:
            logging.info(f"Sqrt替代方法: {'pow(-0.5) 倒数法' if args.use_reciprocal else 'pow(0.5)'}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Trace过程中发生错误: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()

