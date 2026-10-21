import os
import pickle
import base64
import requests
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenPIClient:
    def __init__(self, server_url):
        """
        初始化客户端
        
        Args:
            server_url: 服务器地址，如 'http://localhost:20020'
        """
        self.server_url = server_url.rstrip('/')
        
    def health_check(self):
        """检查服务器健康状态"""
        try:
            response = requests.get(f'{self.server_url}/health', timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return None
    
    def test_infer(self):
        """使用服务器内置的测试数据进行推理"""
        try:
            logger.info("发送测试推理请求...")
            response = requests.get(f'{self.server_url}/test', timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.info("测试推理完成")
            return result
        except Exception as e:
            logger.error(f"测试推理失败: {e}")
            return None
    
    def infer(self, example):
        """
        发送example数据进行推理
        
        Args:
            example: 要推理的数据
            
        Returns:
            推理结果，如果失败则返回None
        """
        try:
            # 序列化example数据
            logger.info("序列化数据...")
            example_bytes = pickle.dumps(example)
            example_b64 = base64.b64encode(example_bytes).decode('utf-8')
            
            # 准备请求数据
            data = {
                'example': example_b64
            }
            
            logger.info("发送推理请求...")
            # 发送POST请求
            response = requests.post(
                f'{self.server_url}/infer',
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=60  # 推理可能需要较长时间
            )
            response.raise_for_status()
            
            # 解析响应
            result_json = response.json()
            
            if result_json.get('success'):
                # 反序列化结果
                result_b64 = result_json['result']
                result_bytes = base64.b64decode(result_b64)
                result = pickle.loads(result_bytes)
                
                logger.info("推理完成")
                return result
            else:
                logger.error(f"服务器返回错误: {result_json.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"推理请求失败: {e}")
            return None

def main():
    """示例使用方法"""
    # 服务器地址 - 请修改为你的服务器IP地址
    SERVER_URL = os.getenv("VLA_SERVER_URL", "http://localhost:20020")
    
    # 创建客户端
    client = OpenPIClient(SERVER_URL)
    #import pdb;pdb.set_trace()
    
    # 1. 检查服务器健康状态
    print("=== 检查服务器状态 ===")
    health = client.health_check()
    if health:
        print(f"服务器状态: {health}")
    else:
        print("无法连接到服务器")
        return
    
    # 2. 使用测试接口
    print("\n=== 测试推理 ===")
    test_result = client.test_infer()
    if test_result:
        print(f"测试结果: {test_result}")
    else:
        print("测试推理失败")
        return
    
    # 3. 发送自定义数据进行推理
    print("\n=== 自定义数据推理 ===")

    import pickle
    import pdb;pdb.set_trace()
    example = pickle.load(open("example.pkl", "rb"))
    result = client.infer(example)
    print("result action:")
    print(result)
    
    # 这里你需要准备你的example数据
    # 例如，如果你有自己的数据，可以这样使用：
    # your_example = your_data_preparation_function()
    # result = client.infer(your_example)
    
    # 为了演示，我们可以使用一个简单的示例数据
    # 注意：这里需要根据你的实际数据格式来调整

if __name__ == '__main__':
    main() 