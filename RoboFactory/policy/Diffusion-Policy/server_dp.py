#!/usr/bin/env python3
import sys
sys.path.append('./') 
sys.path.insert(0, './policy/Diffusion-Policy') 

import argparse
import base64
import logging
import os
import pickle
import torch
import dill
import numpy as np
import hydra
from pathlib import Path

from flask import Flask
from flask import jsonify
from flask import request

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量保存加载的模型和runner
policies = {}  # {agent_id: policy}
runners = {}   # {agent_id: DPRunner}


def get_policy(checkpoint, output_dir, device):
    """加载 Diffusion Policy 模型
    
    Args:
        checkpoint: checkpoint 文件路径
        output_dir: 输出目录（可以为 None）
        device: 设备（如 'cuda:0', 'cpu'）
    
    Returns:
        policy: 加载的模型
    """
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy


def load_model(task_name: str, checkpoint_num: int, data_num: int, 
               agent_ids: list = [0], device: str = "cuda:0", 
               checkpoint_base_dir: str = "checkpoints"):
    """加载模型（支持多 agent）
    
    Args:
        task_name: 任务名称
        checkpoint_num: checkpoint 编号
        data_num: 数据编号
        agent_ids: agent ID 列表，默认为 [0]
        device: 设备（如 'cuda:0', 'cpu'）
        checkpoint_base_dir: checkpoint 基础目录，默认为 'checkpoints'
    
    Returns:
        bool: 是否加载成功
    """
    global policies, runners
    
    try:
        logger.info("开始加载模型...")
        
        for agent_id in agent_ids:
            checkpoint_path = os.path.join(
                checkpoint_base_dir, 
                f"{task_name}_Agent{agent_id}_{data_num}", 
                f"{checkpoint_num}.ckpt"
            )
            
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint 文件不存在: {checkpoint_path}")
                return False
            
            logger.info(f"加载 Agent {agent_id} 的模型: {checkpoint_path}")
            policy = get_policy(checkpoint_path, None, device)
            policies[agent_id] = policy
            runners[agent_id] = DPRunner(output_dir=None)
            logger.info(f"Agent {agent_id} 模型加载成功!")
        
        logger.info("所有模型加载成功!")
        return True
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy", 
        "models_loaded": len(policies) > 0,
        "loaded_agents": list(policies.keys())
    })


@app.route("/infer", methods=["POST"])
def infer():
    """推理接口
    
    请求格式:
    {
        "agent_id": 0,  # agent ID
        "observation": {  # 观察数据
            "head_cam": ...,  # numpy array, shape (C, H, W), 值范围 [0, 1]
            "agent_pos": ...  # numpy array, shape (n,)
        },
        "update_obs": true  # 是否更新观察历史（默认 true）
    }
    """
    global policies, runners

    if len(policies) == 0:
        return jsonify({"error": "模型未加载"}), 500

    try:
        # 获取请求数据
        data = request.get_json()

        if "agent_id" not in data:
            return jsonify({"error": "缺少 agent_id"}), 400
        
        agent_id = data["agent_id"]
        if agent_id not in policies:
            return jsonify({"error": f"Agent {agent_id} 的模型未加载"}), 400

        if "observation" not in data:
            return jsonify({"error": "缺少 observation 数据"}), 400

        # 解码 observation 数据 (假设客户端会发送base64编码的pickle数据)
        obs_b64 = data["observation"]
        if isinstance(obs_b64, str):
            obs_bytes = base64.b64decode(obs_b64)
            observation = pickle.loads(obs_bytes)
        else:
            # 如果直接是字典，尝试转换
            observation = obs_b64
            # 确保 head_cam 和 agent_pos 是 numpy array
            if "head_cam" in observation and not isinstance(observation["head_cam"], np.ndarray):
                observation["head_cam"] = np.array(observation["head_cam"])
            if "agent_pos" in observation and not isinstance(observation["agent_pos"], np.ndarray):
                observation["agent_pos"] = np.array(observation["agent_pos"])

        # 检查数据格式
        if "head_cam" not in observation or "agent_pos" not in observation:
            return jsonify({"error": "observation 必须包含 head_cam 和 agent_pos"}), 400

        update_obs = data.get("update_obs", True)
        
        logger.info(f"收到 Agent {agent_id} 的推理请求")
        logger.debug(f"Observation keys: {observation.keys()}")

        # 获取 runner 和 policy
        runner = runners[agent_id]
        policy = policies[agent_id]

        # 运行推理
        # 如果 update_obs 为 True，则更新观察历史；否则只使用已有历史进行推理
        action = runner.get_action(policy, observation if update_obs else None)

        # 将结果序列化
        result = {"action": action}
        result_bytes = pickle.dumps(result)
        result_b64 = base64.b64encode(result_bytes).decode("utf-8")

        logger.info(f"Agent {agent_id} 推理完成")

        return jsonify(
            {
                "success": True,
                "result": result_b64,
                "action_shape": list(action.shape) if isinstance(action, np.ndarray) else None,
            }
        )

    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"推理失败: {e!s}"}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """重置指定 agent 的观察历史
    
    请求格式:
    {
        "agent_id": 0  # agent ID，如果不提供则重置所有
    }
    """
    global runners
    
    try:
        data = request.get_json() or {}
        agent_id = data.get("agent_id")
        
        if agent_id is not None:
            if agent_id not in runners:
                return jsonify({"error": f"Agent {agent_id} 的 runner 不存在"}), 400
            runners[agent_id].reset_obs()
            logger.info(f"重置 Agent {agent_id} 的观察历史")
        else:
            # 重置所有
            for aid, runner in runners.items():
                runner.reset_obs()
            logger.info("重置所有 Agent 的观察历史")
        
        return jsonify({"success": True, "message": "观察历史已重置"})
        
    except Exception as e:
        logger.error(f"重置观察历史出错: {e}")
        return jsonify({"error": f"重置失败: {e!s}"}), 500


@app.route("/test", methods=["GET"])
def test():
    """测试接口，使用示例数据进行推理"""
    global policies, runners

    if len(policies) == 0:
        return jsonify({"error": "模型未加载"}), 500

    try:
        # 使用第一个可用的 agent
        agent_id = list(policies.keys())[0]
        policy = policies[agent_id]
        runner = runners[agent_id]
        
        # 创建示例数据（需要根据实际情况调整）
        # 这里创建一个简单的示例
        example_obs = {
            "head_cam": np.zeros((3, 224, 224), dtype=np.float32),  # 示例图像
            "agent_pos": np.zeros(7, dtype=np.float32),  # 示例位置（7维，根据实际情况调整）
        }
        
        # 重置观察历史
        runner.reset_obs()
        
        # 运行推理
        action = runner.get_action(policy, example_obs)

        return jsonify(
            {
                "success": True,
                "message": f"测试推理成功 (Agent {agent_id})",
                "action_shape": list(action.shape) if isinstance(action, np.ndarray) else None,
            }
        )

    except Exception as e:
        logger.error(f"测试推理出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"测试失败: {e!s}"}), 500


if __name__ == "__main__":
    logger.info("启动 Diffusion Policy 推理服务器...")

    parser = argparse.ArgumentParser(description="Diffusion Policy inference server")
    parser.add_argument(
        "--task-name", 
        type=str, 
        required=True,
        help="任务名称（用于构建 checkpoint 路径）"
    )
    parser.add_argument(
        "--checkpoint-num", 
        type=int, 
        required=True,
        help="Checkpoint 编号"
    )
    parser.add_argument(
        "--data-num", 
        type=int, 
        required=True,
        help="数据编号"
    )
    parser.add_argument(
        "--agent-ids",
        type=str,
        default="0",
        help="Agent ID 列表，用逗号分隔（如 '0,1,2'），默认为 '0'"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备（如 'cuda:0', 'cpu'），默认为 'cuda:0'"
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint 基础目录，默认为 'checkpoints'"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=20020, 
        help="Flask 服务器端口，默认为 20020"
    )
    args = parser.parse_args()

    # 解析 agent_ids
    agent_ids = [int(x.strip()) for x in args.agent_ids.split(",")]

    # 启动时加载模型
    if not load_model(
        task_name=args.task_name,
        checkpoint_num=args.checkpoint_num,
        data_num=args.data_num,
        agent_ids=agent_ids,
        device=args.device,
        checkpoint_base_dir=args.checkpoint_base_dir
    ):
        logger.error("无法启动服务器: 模型加载失败")
        exit(1)

    # 启动Flask服务器
    logger.info(f"服务器启动在端口 {args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)

