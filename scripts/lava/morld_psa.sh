#!/bin/bash

python3 -u launch_experiment.py \
--algo morld \
--env-id MOLavaGridDR-v0 \
--seed 76 \
--num-timesteps 5000000 \
--gamma 0.995 \
--wandb-group 'domain_randomization' \
--ref-point '-1000.0' '-500.0' \
--test-generalization True \
--init-hyperparams "policy_name:'MOSACDiscrete'" "shared_buffer:True" "exchange_every:10000" "pop_size:5" "weight_adaptation_method:'PSA'" "policy_args:{'target_net_freq':200, 'batch_size':128, 'buffer_size':1000000, 'net_arch':[256, 256, 256, 256], 'update_frequency': 1}"  \
--train-hyperparams eval_mo_freq:50000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 "algo_suffix:'_PSA'" \
--test-envs "MOLavaGridCorridor-v0,MOLavaGridIslands-v0,MOLavaGridMaze-v0,MOLavaGridSnake-v0,MOLavaGridRoom-v0,MOLavaGridLabyrinth-v0,MOLavaGridSmiley-v0,MOLavaGridCheckerBoard-v0" \
# --record-video True \
# --record_video_w_freq:201