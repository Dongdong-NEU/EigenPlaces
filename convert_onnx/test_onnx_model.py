"""
ONNX模型测试脚本

该脚本用于测试导出的ONNX模型，验证其正确性和性能。
"""

import numpy as np
import torch
import time
import argparse
import logging

def test_onnx_model(onnx_path, input_shape=(1, 3, 512, 512), num_tests=10):
    """
    测试ONNX模型
    
    Args:
        onnx_path (str): ONNX模型路径
        input_shape (tuple): 输入形状
        num_tests (int): 测试次数
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        import onnxruntime as ort
        
        logging.info(f"Loading ONNX model from {onnx_path}")
        
        # 创建ONNX Runtime会话
        session = ort.InferenceSession(onnx_path)
        
        # 获取模型信息
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        logging.info(f"Model input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
        logging.info(f"Model output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
        
        # 创建测试输入
        test_input = np.random.randn(*input_shape).astype(np.float32)
        logging.info(f"Test input shape: {test_input.shape}")
        
        # 预热运行
        logging.info("Warming up...")
        for _ in range(3):
            _ = session.run(None, {input_info.name: test_input})
        
        # 性能测试
        logging.info(f"Running {num_tests} inference tests...")
        times = []
        
        for i in range(num_tests):
            start_time = time.time()
            output = session.run(None, {input_info.name: test_input})
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms
            times.append(inference_time)
            
            if i == 0:
                # 打印第一次推理的结果信息
                output_array = output[0]
                logging.info(f"Output shape: {output_array.shape}")
                logging.info(f"Output range: [{output_array.min():.6f}, {output_array.max():.6f}]")
                logging.info(f"Output mean: {output_array.mean():.6f}, std: {output_array.std():.6f}")
                
                # 检查L2归一化
                l2_norm = np.linalg.norm(output_array, axis=1)
                logging.info(f"L2 norm: {l2_norm[0]:.6f} (should be close to 1.0)")
        
        # 统计推理时间
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        logging.info(f"Inference time statistics ({num_tests} runs):")
        logging.info(f"  Average: {avg_time:.2f} ms")
        logging.info(f"  Min: {min_time:.2f} ms")
        logging.info(f"  Max: {max_time:.2f} ms")
        logging.info(f"  Std: {std_time:.2f} ms")
        logging.info(f"  FPS: {1000/avg_time:.1f}")
        
        logging.info("✓ ONNX model test completed successfully!")
        
    except ImportError:
        logging.error("onnxruntime is not installed. Install with: pip install onnxruntime")
    except Exception as e:
        logging.error(f"Error during ONNX model test: {str(e)}")
        raise

def compare_pytorch_onnx(onnx_path, input_shape=(1, 3, 512, 512)):
    """
    比较PyTorch模型和ONNX模型的输出
    
    Args:
        onnx_path (str): ONNX模型路径
        input_shape (tuple): 输入形状
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        import onnxruntime as ort
        from onnx_compatible_model import create_onnx_compatible_model
        
        logging.info("Comparing PyTorch and ONNX model outputs...")
        
        # 创建PyTorch模型
        pytorch_model = create_onnx_compatible_model("ResNet50", 2048)
        pytorch_model.eval()
        
        # 创建ONNX Runtime会话
        onnx_session = ort.InferenceSession(onnx_path)
        input_name = onnx_session.get_inputs()[0].name
        
        # 创建测试输入
        test_input_np = np.random.randn(*input_shape).astype(np.float32)
        test_input_torch = torch.from_numpy(test_input_np)
        
        # PyTorch推理
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input_torch).numpy()
        
        # ONNX推理
        onnx_output = onnx_session.run(None, {input_name: test_input_np})[0]
        
        # 比较输出
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        rel_diff = max_diff / (np.max(np.abs(pytorch_output)) + 1e-8)
        
        logging.info(f"Output comparison:")
        logging.info(f"  PyTorch output shape: {pytorch_output.shape}")
        logging.info(f"  ONNX output shape: {onnx_output.shape}")
        logging.info(f"  Max absolute difference: {max_diff:.8f}")
        logging.info(f"  Mean absolute difference: {mean_diff:.8f}")
        logging.info(f"  Relative difference: {rel_diff:.8f}")
        
        if max_diff < 1e-5:
            logging.info("✓ PyTorch and ONNX outputs match well!")
        elif max_diff < 1e-3:
            logging.info("⚠ PyTorch and ONNX outputs have small differences (acceptable)")
        else:
            logging.warning(f"⚠ PyTorch and ONNX outputs have significant differences!")
            
    except Exception as e:
        logging.error(f"Error during comparison: {str(e)}")
        raise

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Test ONNX model")
    
    parser.add_argument("--onnx_path", type=str, required=True,
                       help="Path to ONNX model file")
    parser.add_argument("--input_height", type=int, default=512,
                       help="Input image height")
    parser.add_argument("--input_width", type=int, default=512,
                       help="Input image width")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--num_tests", type=int, default=10,
                       help="Number of inference tests")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with PyTorch model")
    
    args = parser.parse_args()
    
    input_shape = (args.batch_size, 3, args.input_height, args.input_width)
    
    # 测试ONNX模型
    test_onnx_model(args.onnx_path, input_shape, args.num_tests)
    
    # 比较PyTorch和ONNX模型（可选）
    if args.compare:
        compare_pytorch_onnx(args.onnx_path, input_shape)

if __name__ == "__main__":
    main()
