"""
Depth Anything 3 Client
发送2张图片到服务器，接收并保存处理结果
"""

import base64
import requests
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import json


def image_to_base64(image_path: str) -> str:
    """读取图片并转换为 base64"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def base64_to_image(base64_str: str) -> np.ndarray:
    """将 base64 转换为图像数组"""
    image_data = base64.b64decode(base64_str)
    import io
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)


def save_image(base64_str: str, output_path: str):
    """保存 base64 图像到文件"""
    img_array = base64_to_image(base64_str)
    Image.fromarray(img_array).save(output_path)
    print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Depth Anything 3 Client")
    parser.add_argument("--input_images_dir", type=str, help="输入图片目录路径")
    parser.add_argument("--align_cam_intrinsics", type=str, help="内参矩阵")
    parser.add_argument("--align_cam_extrinsics", type=str, help="外参矩阵")
    parser.add_argument("--tgt_cam_extrinsics", type=str, help="目标外参矩阵")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="服务器地址 (默认: http://localhost:8000)")
    parser.add_argument("--output-dir", type=str, default="./client_output",
                        help="输出目录 (默认: ./client_output)")
    parser.add_argument("--src-idx", type=int, default=0,
                        help="源图像索引 (默认: 0)")
    # parser.add_argument("--tgt-idx", type=int, default=1,
    #                     help="目标图像索引 (默认: 1)")
    parser.add_argument("--save-comparison", action="store_true",
                        help="是否保存对比图 (默认: 不保存)")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Depth Anything 3 Client")
    print("=" * 60)
    
    # 1. 检查服务器健康状态
    print("\n[1/5] Checking server health...")
    try:
        response = requests.get(f"{args.server}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"  ✓ Server status: {health['status']}")
            print(f"  ✓ Model loaded: {health['model_loaded']}")
            print(f"  ✓ Device: {health['device']}")
        else:
            print(f"  ✗ Server health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"  ✗ Cannot connect to server: {e}")
        print(f"  Make sure server is running at {args.server}")
        return
    
    # 2. 读取并编码图片,内参矩阵,外参矩阵,目标外参矩阵
    print("\n[2/5] Reading images and camera parameters...")
    try:
        # 从输入目录读取图片
        input_images_path = Path(args.input_images_dir)
        if not input_images_path.exists():
            raise FileNotFoundError(f"Input images directory not found: {args.input_images_dir}")
        
        # 获取目录中的所有图片文件（按文件名排序）
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_files = sorted([
            f for f in input_images_path.iterdir() 
            if f.is_file() and f.suffix in image_extensions
        ])
        
        if len(image_files) < 3:
            raise ValueError(f"Expected at least 3 images in {args.input_images_dir}, found {len(image_files)}")
        
        # 读取前3张图片
        print(f"  Reading images from: {args.input_images_dir}")
        images_base64 = []
        for i, img_path in enumerate(image_files[:3]):
            print(f"  Reading: {img_path.name}")
            img_base64 = image_to_base64(str(img_path))
            images_base64.append(img_base64)
            print(f"  ✓ Image {i+1} encoded ({len(img_base64)} bytes)")
        
        image1_base64, image2_base64, image3_base64 = images_base64

        # 读取相机内参 JSON 文件
        print(f"  Reading: {args.align_cam_intrinsics}")
        with open(args.align_cam_intrinsics, 'r') as f:
            intrinsics_data = json.load(f)
        # 提取内参矩阵 (3x3)，为3张图像复制相同的内参
        img_intrinsic_3x3 = intrinsics_data['img_intrinsic_3x3']
        align_cam_intrinsics = [img_intrinsic_3x3] * 3  # [N, 3, 3]
        print(f"  ✓ Align cam intrinsics loaded: {len(align_cam_intrinsics)} matrices (3x3 each)")

        # 读取相机外参 JSON 文件
        print(f"  Reading: {args.align_cam_extrinsics}")
        with open(args.align_cam_extrinsics, 'r') as f:
            extrinsics_data = json.load(f)
        # 提取外参矩阵 (3x4)，按顺序组成列表
        align_cam_extrinsics = [
            extrinsics_data[f'img_{i}_extrinsic_3x4'] 
            for i in range(3)
        ]  # [N, 3, 4]
        print(f"  ✓ Align cam extrinsics loaded: {len(align_cam_extrinsics)} matrices (3x4 each)")

        # 读取目标相机外参 JSON 文件
        print(f"  Reading: {args.tgt_cam_extrinsics}")
        with open(args.tgt_cam_extrinsics, 'r') as f:
            tgt_extrinsics_data = json.load(f)
        # 提取目标外参矩阵 (3x4)
        tgt_cam_extrinsics = tgt_extrinsics_data['img_tgt_extrinsic_3x4']  # [3, 4]
        print(f"  ✓ Tgt cam extrinsics loaded: 1 matrix (3x4)")

    except Exception as e:
        print(f"  ✗ Failed to read images or camera parameters: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 发送推理请求
    print("\n[3/5] Sending inference request...")
    payload = {
        "images": [image1_base64, image2_base64, image3_base64],
        "src_idx": args.src_idx,
        "align_cam_intrinsics": align_cam_intrinsics,
        "align_cam_extrinsics": align_cam_extrinsics,
        "tgt_cam_extrinsics": tgt_cam_extrinsics
    }
    
    try:
        print(f"  Uploading to {args.server}/inference...")
        print("  (This may take a while, please wait...)")
        
        response = requests.post(
            f"{args.server}/inference",
            json=payload,
            timeout=300  # 5分钟超时
        )
        
        if response.status_code != 200:
            print(f"  ✗ Request failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return
        
        print("  ✓ Inference completed!")
        
    except requests.exceptions.Timeout:
        print("  ✗ Request timeout (>5 minutes)")
        return
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return
    
    # 4. 解析响应
    print("\n[4/5] Parsing response...")
    try:
        result = response.json()
        
        if not result.get('success'):
            print(f"  ✗ Inference failed: {result.get('message')}")
            return
        
        print(f"  ✓ Success: {result['message']}")
        
        # 打印统计信息
        if 'stats' in result and result['stats']:
            print("\n  Statistics:")
            stats = result['stats']
            print(f"    - Valid pixels: {stats['valid_pixels']:,} / {stats['total_pixels']:,}")
            print(f"    - Valid ratio: {stats['valid_ratio']*100:.2f}%")
            print(f"    - Mean Absolute Error: {stats['mean_absolute_error']:.4f}")
            print(f"    - Max Absolute Error: {stats['max_absolute_error']:.4f}")
            print(f"    - PSNR: {stats['psnr_db']:.2f} dB")
        
    except Exception as e:
        print(f"  ✗ Failed to parse response: {e}")
        return
    
    # 5. 保存结果
    print(f"\n[5/5] Saving results to {output_dir}/...")
    try:
        # 保存 warped 图像（必需）
        if result.get('warped_image'):
            save_image(result['warped_image'], 
                      output_dir / "warped_image.png")
        else:
            print("  ✗ Warning: No warped_image in response")
        
        # 保存对比图（可选）
        if args.save_comparison and result.get('comparison_image'):
            save_image(result['comparison_image'], 
                      output_dir / "comparison.png")
        
        print("\n" + "=" * 60)
        print("✓ All done! Results saved to:")
        print(f"  {output_dir.absolute()}")
        print("=" * 60)
        print("\nSaved files:")
        print("  - warped_image.png : Warping 结果")
        if args.save_comparison:
            print("  - comparison.png   : 对比图 [src|warped|tgt|diff]")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"  ✗ Failed to save results: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

