# --- Visual perturbation evaluation (RoboTwin 2.0 style) ---
PERTURBATION_CFG=configs/perturbation/visual_perturbation.yaml
bash script/MACG/eval_stackbowl_20022.sh configs/table/two_arms_stack_bowl.yaml 1 300 0 two_arms_stack_bowl_12 script/MACG/stack_bowls_indomain.yaml "$PERTURBATION_CFG"
mv eval_video/TwoArmsStackBowl-rf eval_video/TwoArmsStackBowl-rf-in-domain-perturbed-pi0
bash script/MACG/eval_stackbowl_20022.sh configs/table/two_arms_stack_bowl.yaml 1 300 0 two_arms_stack_bowl_12 script/MACG/stack_bowls_sync.yaml "$PERTURBATION_CFG"
mv eval_video/TwoArmsStackBowl-rf eval_video/TwoArmsStackBowl-rf-sync-perturbed-pi0
bash script/MACG/eval_stackbowl_20022.sh configs/table/two_arms_stack_bowl.yaml 1 300 0 two_arms_stack_bowl_21 script/MACG/stack_bowls_reorder.yaml "$PERTURBATION_CFG"
mv eval_video/TwoArmsStackBowl-rf eval_video/TwoArmsStackBowl-rf-reorder-perturbed-pi0
bash script/MACG/eval_stackbowl_20022.sh configs/table/two_arms_stack_bowl.yaml 1 300 0 two_arms_stack_bowl_21 script/MACG/stack_bowls_syncreorder.yaml "$PERTURBATION_CFG"
mv eval_video/TwoArmsStackBowl-rf eval_video/TwoArmsStackBowl-rf-syncreorder-perturbed-pi0

PERTURBATION_CFG=configs/perturbation/visual_perturbation_stackcube.yaml
bash script/MACG/eval_two_arms_stackcube.sh configs/table/two_arms_stack_cube.yaml 1 300 0 two_arms_stack_cube_12 script/MACG/stack_cubes_sync.yaml "$PERTURBATION_CFG"
mv eval_video/TwoArmsStackCube-rf eval_video/TwoArmsStackCube-rf-sync-perturbed-pi0