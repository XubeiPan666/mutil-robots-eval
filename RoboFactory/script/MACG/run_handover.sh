bash script/MACG/eval_handover_bottles.sh 1 300 script/MACG/phase_config_handover_bottles_indomain.yaml

mv eval_video/TwoArmsHandoverBottles-rf eval_video/TwoArmsHandoverBottles-rf-in-domain-lct

bash script/MACG/eval_handover_bottles.sh 1 300 script/MACG/phase_config_handover_bottles_outdomain.yaml

mv eval_video/TwoArmsHandoverBottles-rf eval_video/TwoArmsHandoverBottles-rf-out-domain-lct