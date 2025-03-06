#!/bin/bash

python3 -u launch_experiment.py \
--algo envelope \
--env-id MOLavaGridDR-v0 \
--seed 5 \
--num-timesteps 5000000 \
--gamma 0.995 \
--wandb-group 'domain_randomization' \
--ref-point '-1000.0' '-500.0' \
--test-generalization True \
--init-hyperparams initial_epsilon:1.0 final_epsilon:0.05 epsilon_decay_steps:5000000 initial_homotopy_lambda:0.95 "batch_size:128" "buffer_size:1000000" "net_arch:[256, 256, 256, 256]" \
--train-hyperparams num_eval_episodes_for_front:1 eval_mo_freq:50000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 \
--test-envs "MOLavaGridCorridor-v0,MOLavaGridIslands-v0,MOLavaGridMaze-v0,MOLavaGridSnake-v0,MOLavaGridRoom-v0,MOLavaGridLabyrinth-v0,MOLavaGridSmiley-v0,MOLavaGridCheckerBoard-v0" \
# --record-video True \
# --record_video_w_freq:201