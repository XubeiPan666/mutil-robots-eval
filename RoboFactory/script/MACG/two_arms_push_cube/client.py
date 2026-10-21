import os
import pickle
import base64
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenPIClient:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')
        
    def health_check(self):
        try:
            response = requests.get(f'{self.server_url}/health', timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return None
    
    def test_infer(self):
        try:
            logger.info("Sending test inference request...")
            response = requests.get(f'{self.server_url}/test', timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.info("Test inference complete")
            return result
        except Exception as e:
            logger.error(f"Test inference failed: {e}")
            return None
    
    def infer(self, example):
        try:
            logger.info("Serializing data...")
            example_bytes = pickle.dumps(example)
            example_b64 = base64.b64encode(example_bytes).decode('utf-8')
            
            data = {
                'example': example_b64
            }
            
            logger.info("Sending inference request...")
            response = requests.post(
                f'{self.server_url}/infer',
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            response.raise_for_status()
            
            result_json = response.json()
            
            if result_json.get('success'):
                result_b64 = result_json['result']
                result_bytes = base64.b64decode(result_b64)
                result = pickle.loads(result_bytes)
                
                logger.info("Inference complete")
                return result
            else:
                logger.error(f"Server returned error: {result_json.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"Inference request failed: {e}")
            return None

if __name__ == '__main__':
    SERVER_URL = os.getenv("VLA_SERVER_URL", "http://localhost:20020")
    client = OpenPIClient(SERVER_URL)
    
    print("=== Checking server status ===")
    health = client.health_check()
    if health:
        print(f"Server status: {health}")
    else:
        print("Cannot connect to server")
