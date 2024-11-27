#!/bin/bash

python3 -u launch_experiment.py \
--algo morld \
--env-id MOLunarLanderDR-v0 \
--seed 5 \
--num-timesteps 3000000 \
--gamma 0.99 \
--auto-tag False \
--wandb-group 'domain_randomization' \
--ref-point '-101' '-1001' '-101' '-101' \
--test-generalization True \
--init-hyperparams "policy_name:'MOSACDiscrete'" "shared_buffer:True" "exchange_every:25000" "pop_size:10" "weight_adaptation_method:'PSA'" "policy_args:{'target_net_freq':2000, 'batch_size':128, 'buffer_size':1000000, 'net_arch':[256, 256, 256, 256], 'update_frequency': 1, 'target_entropy_scale':0.3}"  \
--train-hyperparams eval_mo_freq:50000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 "algo_suffix:'_PSA'" \
--test-envs "MOLunarLanderDefault-v0,MOLunarLanderHighGravity-v0,MOLunarLanderWindy-v0,MOLunarLanderTurbulent-v0,MOLunarLanderLowMainEngine-v0,MOLunarLanderLowSideEngine-v0,MOLunarLanderStartRight-v0,MOLunarLanderStartLow-v0,MOLunarLanderHard-v0" \
# --record-video True \