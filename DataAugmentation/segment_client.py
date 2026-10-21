#!/usr/bin/env python3
"""
测试BiRefNet服务器的客户端示例
用法: python test_server.py --image <图片路径> --url <服务器URL>
"""

import argparse
import base64
import os
import requests


def test_predict_base64(image_path: str, server_url: str = "http://localhost:8000", resolution: str = None, output_dir: str = "outputs"):
    """
    使用base64编码方式预测mask
    
    Args:
        image_path: 图片路径
        server_url: 服务器URL
        resolution: 推理分辨率
        output_dir: 输出目录路径（相对于脚本所在目录）
    """
    print(f"Testing /predict_base64 endpoint...")
    print(f"Image: {image_path}")
    print(f"Server: {server_url}")
    
    # 读取图片并编码为base64
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    url = f"{server_url}/predict_base64"
    
    payload = {
        "image_base64": image_base64,
    }
    if resolution:
        payload["resolution"] = resolution
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        mask_base64 = result['mask_base64']
        
        # 获取脚本所在目录，确保输出目录在代码目录下
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 如果 output_dir 是相对路径，则相对于脚本目录；如果是绝对路径，则直接使用
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(script_dir, output_dir)
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 解码并保存mask
        mask_bytes = base64.b64decode(mask_base64)
        # 获取原图片的文件名（不含路径）
        image_filename = os.path.basename(image_path)
        image_name_without_ext = os.path.splitext(image_filename)[0]
        output_filename = f"{image_name_without_ext}_mask.png"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'wb') as f:
            f.write(mask_bytes)
        print(f"✓ Success! Mask saved to: {output_path}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def test_health(server_url: str = "http://localhost:8000"):
    """测试健康检查端点"""
    print(f"\nTesting /health endpoint...")
    url = f"{server_url}/health"
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Health check passed:")
        print(f"  Status: {result['status']}")
        print(f"  Model loaded: {result['model_loaded']}")
        print(f"  Device: {result['device']}")
    else:
        print(f"✗ Error: {response.status_code}")


def main():
    parser = argparse.ArgumentParser(description='测试BiRefNet服务器')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--url', type=str, default='http://localhost:8000', help='服务器URL（默认: http://localhost:8000）')
    parser.add_argument('--resolution', type=str, default=None, help='推理分辨率，格式: WxH')
    parser.add_argument('--output', type=str, default='outputs', help='输出目录路径（默认: outputs）')
    
    args = parser.parse_args()
    
    # 健康检查
    test_health(args.url)
    
    # 使用base64方式预测
    test_predict_base64(args.image, args.url, args.resolution, args.output)


if __name__ == '__main__':
    main()
