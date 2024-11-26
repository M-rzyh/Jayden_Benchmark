#!/bin/bash

python3 -u launch_experiment.py \
--algo pcn \
--env-id MOLavaGridDR-v0 \
--seed 5 \
--num-timesteps 5000000 \
--gamma 0.995 \
--wandb-group 'domain_randomization' \
--ref-point '-1000.0' '-100.0' \
--test-generalization True \
--init-hyperparams "scaling_factor:[0.1, 0.1, 0.1]" "max_return:[300, 300]" "net_arch:[256, 256, 256, 256]" \
--train-hyperparams eval_mo_freq:50000 max_buffer_size:4000 num_er_episodes:2000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 \
--test-envs "MOLavaGridCorridor-v0,MOLavaGridIslands-v0,MOLavaGridMaze-v0,MOLavaGridSnake-v0,MOLavaGridRoom-v0,MOLavaGridLabyrinth-v0,MOLavaGridSmiley-v0,MOLavaGridCheckerBoard-v0" \