#!/bin/bash

python3 -u launch_experiment.py \
--algo sac_discrete \
--env-id MOLavaGridDR-v0 \
--seed 92 \
--num-timesteps 5000000 \
--gamma 0.995 \
--ref-point '-1000.0' '-500.0' \
--wandb-group 'domain_randomization' \
--test-generalization True \
--init-hyperparams alpha:0.2 "net_arch:[256, 256, 256, 256]" target_net_freq:200 batch_size:128 buffer_size:1000000 \
--train-hyperparams eval_mo_freq:50000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 record_video_w_freq:201 \
--test-envs "MOLavaGridCorridor-v0,MOLavaGridIslands-v0,MOLavaGridMaze-v0,MOLavaGridSnake-v0,MOLavaGridRoom-v0,MOLavaGridLabyrinth-v0,MOLavaGridSmiley-v0,MOLavaGridCheckerBoard-v0" \
# --record-video True \