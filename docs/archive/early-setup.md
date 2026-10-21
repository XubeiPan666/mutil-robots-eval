> Historical development notes. Commands may target unreleased configurations. Use the [current installation guide](../installation.md) and [evaluation guide](../evaluation.md).

# 2025 7.4 xjl 想润色一下readme.

#  环境搭建
## 创建并激活 Conda 环境（Python 3.11）
conda create -n pi0 python=3.11 -y  
conda activate pi0
## 安装 uv 和依赖项
pip install uv  
GIT_LFS_SKIP_SMUDGE=1 uv sync  
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
## 安装RoboFactory依赖  
pip install -r requirements.txt
## 下载模型和资源文件  
python script/download_assets.py

# 数据生成
## 生成示例任务数据（以 lift_barrier.yaml 为例，生成 150 条数据并保存视频）  
python script/generate_data.py configs/table/lift_barrier.yaml 150 --save-video
## 转换数据格式（进入 Pi0 策略目录）
cd policy/Pi0  
python convert.py

# 训练
## 安装系统依赖（Ubuntu/Debian 系统）  
sudo apt update  
sudo apt install -y libgl1
## 计算归一化统计量  
uv run scripts/compute_norm_stats.py --config-name pi0_fast_droid_train_lora
## 启动训练任务（设置 XLA 显存占用比例为 1）  
XLA_PYTHON_CLIENT_MEM_FRACTION=1 uv run scripts/train.py pi0_fast_droid_train_lora --exp-name=train_300steps --overwrite

# 我们是在服务器上部署pi0，需要在服务器运行：
python server.py
# 然后在本地运行
python client.py
