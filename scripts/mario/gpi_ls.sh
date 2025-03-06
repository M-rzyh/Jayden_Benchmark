#!/bin/bash

python3 -u launch_experiment.py \
--algo gpi_ls_discrete \
--env-id MOSuperMarioBrosZeroShot-v2 \
--seed 47 \
--num-timesteps 3000000 \
--gamma 0.99 \
--ref-point '-100.0' '-100.0' '-100.0' \
--wandb-group 'domain_randomization' \
--wandb-tags 'mario zero shot' \
--test-generalization True \
--init-hyperparams "net_arch:[512, 512]" "learning_rate:0.0003" "batch_size:64" "learning_starts:25000" "initial_epsilon:0.2" "final_epsilon:0.01" "epsilon_decay_steps:1000000" "target_net_update_freq:1000" "gradient_updates:1" "layer_norm:False" "buffer_size:100000" \
--train-hyperparams num_eval_episodes_for_front:3 num_eval_weights_for_front:32 timesteps_per_iter:50000 eval_mo_freq:100000 \
--generalization-hyperparams num_eval_weights:32 num_eval_episodes:1 \
--test-envs "MOSuperMarioBros-1-2-v2,MOSuperMarioBros-3-2-v2,MOSuperMarioBros-3-3-v2,MOSuperMarioBros-4-3-v2,MOSuperMarioBros-5-2-v2,MOSuperMarioBros-5-3-v2,MOSuperMarioBros-7-3-v2,MOSuperMarioBros-8-1-v2" \
# --record-video True \
# --record_video_w_freq:23