#!/usr/bin/env python3
"""
EigenPlaces模型管理工具
功能: 下载、保存、列出、删除本地模型
"""

import torch
import os
import argparse
import logging
from datetime import datetime
import glob

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="EigenPlaces Model Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 下载命令
    download_parser = subparsers.add_parser('download', help='Download model from PyTorch Hub')
    download_parser.add_argument("--backbone", type=str, default="ResNet50",
                               choices=["ResNet18", "ResNet50", "ResNet101", "ResNet152", "VGG16"])
    download_parser.add_argument("--fc_output_dim", type=int, default=2048)
    download_parser.add_argument("--output_dir", type=str, default="./models")
    
    # 列出命令
    list_parser = subparsers.add_parser('list', help='List local models')
    list_parser.add_argument("--models_dir", type=str, default="./models")
    
    # 信息命令
    info_parser = subparsers.add_parser('info', help='Show model info')
    info_parser.add_argument("model_path", type=str, help="Path to model file")
    
    # 删除命令
    delete_parser = subparsers.add_parser('delete', help='Delete model')
    delete_parser.add_argument("model_path", type=str, help="Path to model file")
    
    return parser.parse_args()

def download_model(args):
    """下载并保存模型"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # 从PyTorch Hub加载模型
        logging.info(f"从PyTorch Hub下载: {args.backbone}, 输出维度: {args.fc_output_dim}")
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                              backbone=args.backbone, fc_output_dim=args.fc_output_dim)
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 生成模型文件名
        model_name = f"eigenplaces_{args.backbone.lower()}_{args.fc_output_dim}.pth"
        model_path = os.path.join(args.output_dir, model_name)
        
        # 保存模型
        logging.info(f"保存模型到: {model_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'backbone': args.backbone,
            'fc_output_dim': args.fc_output_dim,
            'save_time': datetime.now().isoformat(),
            'source': 'pytorch_hub_eigenplaces'
        }, model_path)
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ 模型下载完成: {model_path} ({file_size:.2f} MB)")
        
    except Exception as e:
        logging.error(f"下载模型失败: {str(e)}")
        raise

def list_models(args):
    """列出本地模型"""
    if not os.path.exists(args.models_dir):
        print(f"❌ 模型目录不存在: {args.models_dir}")
        return
    
    model_files = glob.glob(os.path.join(args.models_dir, "eigenplaces_*.pth"))
    
    if not model_files:
        print(f"📂 {args.models_dir} 目录中没有找到EigenPlaces模型")
        return
    
    print(f"📂 在 {args.models_dir} 中找到 {len(model_files)} 个模型:")
    print("-" * 80)
    
    for model_file in sorted(model_files):
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            file_size = os.path.getsize(model_file) / (1024 * 1024)
            
            print(f"📄 {os.path.basename(model_file)}")
            print(f"   路径: {model_file}")
            print(f"   骨架: {checkpoint.get('backbone', 'Unknown')}")
            print(f"   输出维度: {checkpoint.get('fc_output_dim', 'Unknown')}")
            print(f"   大小: {file_size:.2f} MB")
            print(f"   保存时间: {checkpoint.get('save_time', 'Unknown')}")
            print()
            
        except Exception as e:
            print(f"❌ 无法读取 {model_file}: {str(e)}")

def show_model_info(args):
    """显示模型信息"""
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        file_size = os.path.getsize(args.model_path) / (1024 * 1024)
        
        print("📋 模型信息:")
        print("-" * 40)
        print(f"文件路径: {args.model_path}")
        print(f"文件大小: {file_size:.2f} MB")
        print(f"骨架网络: {checkpoint.get('backbone', 'Unknown')}")
        print(f"输出维度: {checkpoint.get('fc_output_dim', 'Unknown')}")
        print(f"保存时间: {checkpoint.get('save_time', 'Unknown')}")
        print(f"数据源: {checkpoint.get('source', 'Unknown')}")
        print(f"参数数量: {len(checkpoint.get('model_state_dict', {}))}")
        
        # 显示使用示例
        print("\n🚀 使用示例:")
        print(f"ONNX导出: python convert_onnx/export_onnx.py --resume_model {args.model_path}")
        print(f"Trace模型: python trace_model/trace_eigenplaces.py --resume_model {args.model_path} --no_sqrt")
        
    except Exception as e:
        print(f"❌ 无法读取模型信息: {str(e)}")

def delete_model(args):
    """删除模型"""
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    # 显示模型信息
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        file_size = os.path.getsize(args.model_path) / (1024 * 1024)
        
        print(f"准备删除模型:")
        print(f"  文件: {args.model_path}")
        print(f"  骨架: {checkpoint.get('backbone', 'Unknown')}")
        print(f"  输出维度: {checkpoint.get('fc_output_dim', 'Unknown')}")
        print(f"  大小: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"⚠️ 无法读取模型信息: {str(e)}")
    
    # 确认删除
    confirm = input("确定要删除吗? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        os.remove(args.model_path)
        print(f"✅ 模型已删除: {args.model_path}")
    else:
        print("❌ 取消删除")

def main():
    """主函数"""
    args = parse_arguments()
    
    if args.command == 'download':
        download_model(args)
    elif args.command == 'list':
        list_models(args)
    elif args.command == 'info':
        show_model_info(args)
    elif args.command == 'delete':
        delete_model(args)
    else:
        print("请指定命令: download, list, info, delete")
        print("使用 --help 查看详细帮助")

if __name__ == "__main__":
    main()
