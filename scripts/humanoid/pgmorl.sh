#!/bin/bash

python3 -u launch_experiment.py \
--algo pgmorl \
--env-id MOHumanoidDR-v5 \
--seed 5 \
--num-timesteps 10000000 \
--gamma 0.99 \
--wandb-group 'domain_randomization' \
--ref-point '-100.0' '-100.0' \
--test-generalization True \
--init-hyperparams num_envs:4 pop_size:6 steps_per_iteration:2048 warmup_iterations:200 evolutionary_iterations:40 num_performance_buffer:100 sparsity_coef:-1 delta_weight:0.2 "net_arch:[256, 256, 256, 256]" \
--train-hyperparams eval_mo_freq:100000  \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 "generalization_algo:'dr_state_action_history'" history_len:2 \
--test-envs "MOHumanoidDefault-v5,MOHumanoidLight-v5,MOHumanoidHeavy-v5,MOHumanoidLowDamping-v5,MOHumanoidHard-v5" \
# --record-video True \
# --record_video_w_freq:297