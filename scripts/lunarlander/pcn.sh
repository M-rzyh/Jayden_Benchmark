#!/bin/bash

python3 -u launch_experiment.py \
--algo pcn \
--env-id MOLunarLanderDR-v0 \
--seed 92 \
--num-timesteps 3000000 \
--gamma 0.99 \
--wandb-group 'domain_randomization' \
--ref-point '-101' '-1001' '-101' '-101' \
--test-generalization True \
--init-hyperparams "learning_rate:0.0003" "scaling_factor:[0.1, 0.1, 0.1, 0.1, 0.1]" "net_arch:[256, 256, 256, 256]" \
--train-hyperparams eval_mo_freq:50000 max_buffer_size:100000 num_er_episodes:1000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 "generalization_algo:'dr_state_action_history'" history_len:2 \
--test-envs "MOLunarLanderDefault-v0,MOLunarLanderHighGravity-v0,MOLunarLanderWindy-v0,MOLunarLanderTurbulent-v0,MOLunarLanderLowMainEngine-v0,MOLunarLanderLowSideEngine-v0,MOLunarLanderStartRight-v0,MOLunarLanderHard-v0" \
# --record-video True \