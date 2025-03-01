#!/bin/bash

python3 -u launch_experiment.py \
--algo pcn \
--env-id MOHumanoidDR-v5 \
--seed 5 \
--num-timesteps 10000000 \
--gamma 0.99 \
--wandb-group 'domain_randomization' \
--ref-point '-100.0' '-100.0' \
--test-generalization True \
--init-hyperparams "scaling_factor:[0.1, 0.1, 0.1]" "max_return:[5200, 1000]" "net_arch:[256, 256, 256, 256]" \
--train-hyperparams eval_mo_freq:100000 max_buffer_size:100000 num_er_episodes:1000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 "generalization_algo:'dr_state_action_history'" history_len:2 \
--test-envs "MOHumanoidDefault-v5,MOHumanoidLight-v5,MOHumanoidHeavy-v5,MOHumanoidLowDamping-v5,MOHumanoidHard-v5" \
# --record-video True \