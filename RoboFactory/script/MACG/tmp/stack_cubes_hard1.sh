PERTURBATION_CFG=configs/perturbation/visual_perturbation_stackcube.yaml
bash script/MACG/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_21 script/MACG/stack_cubes_reorder.yaml "$PERTURBATION_CFG"
mv eval_video/TwoArmsStackCube-rf eval_video/TwoArmsStackCube-rf-reorder-perturbed-pi0
bash script/MACG/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_21 script/MACG/stack_cubes_syncreorder.yaml "$PERTURBATION_CFG"
mv eval_video/TwoArmsStackCube-rf eval_video/TwoArmsStackCube-rf-syncreorder-perturbed-pi0