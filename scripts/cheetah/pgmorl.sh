#!/bin/bash

python3 -u launch_experiment.py \
--algo pgmorl \
--env-id MOHalfCheetahDR-v5 \
--seed 5 \
--num-timesteps 5000000 \
--gamma 0.99 \
--wandb-group 'domain_randomization' \
--ref-point '-100.0' '-500.0' \
--test-generalization True \
--init-hyperparams num_envs:4 steps_per_iteration:2048 warmup_iterations:80 evolutionary_iterations:20 num_performance_buffer:100 delta_weight:0.2 sparsity_coef:-1 "net_arch:[256, 256, 256, 256]" \
--train-hyperparams eval_mo_freq:100000  \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 "generalization_algo:'dr_state_action_history'" history_len:2 \
--test-envs "MOHalfCheetahDefault-v5,MOHalfCheetahLight-v5,MOHalfCheetahHeavy-v5,MOHalfCheetahSlippery-v5,MOHalfCheetahHard-v5" \
# --record-video True \