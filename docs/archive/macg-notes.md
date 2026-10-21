> Historical development notes. Commands may target unreleased configurations. Use the [current installation guide](../installation.md) and [evaluation guide](../evaluation.md).

# Place factory
```bash
cd RoboFactory

# 生成 123 顺序
python script/generate_data.py configs/table/place_factory_object.yaml 100 --status place_factory_object_123 --record-dir demo

# 生成 132 顺序
python script/generate_data.py configs/table/place_factory_object.yaml 100 --status place_factory_object_132 --record-dir demo

# 生成 213 顺序
python script/generate_data.py configs/table/place_factory_object.yaml 100 --status place_factory_object_213 --record-dir demo


# 测试一下视频看看效果
script/generate_data.py configs/table/place_factory_object.yaml 1 --status place_factory_object_123 --record-dir demo --save-video

#评测
bash script/MACG/eval_placefactory_object.sh script/MACG/fun.yaml 100 300 place_factory_object_132

```

# Three arms stack cubes
```bash
python script/generate_data.py configs/table/three_robots_stack_cube.yaml 1 --status three_robots_stack_cube_wrist --record-dir demo --save-video
```