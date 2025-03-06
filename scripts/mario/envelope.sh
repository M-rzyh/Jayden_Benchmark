#!/bin/bash

python3 -u launch_experiment.py \
--algo envelope \
--env-id MOSuperMarioBrosZeroShot-v2 \
--seed 5 \
--num-timesteps 3000000 \
--gamma 0.99 \
--ref-point '-100.0' '-100.0' '-100.0' \
--wandb-group 'domain_randomization' \
--wandb-tags 'mario zero shot' \
--test-generalization True \
--init-hyperparams "net_arch:[512, 512]" batch_size:64 learning_starts:25000 num_sample_w:8 per:True initial_epsilon:0.2 final_epsilon:0.01 epsilon_decay_steps:1000000 initial_homotopy_lambda:0.95 buffer_size:100000 gradient_updates:1 target_net_update_freq:1000 \
--train-hyperparams eval_mo_freq:100000 \
--generalization-hyperparams num_eval_weights:32 num_eval_episodes:1 \
--test-envs "MOSuperMarioBros-1-2-v2,MOSuperMarioBros-3-2-v2,MOSuperMarioBros-3-3-v2,MOSuperMarioBros-4-3-v2,MOSuperMarioBros-5-2-v2,MOSuperMarioBros-5-3-v2,MOSuperMarioBros-7-3-v2,MOSuperMarioBros-8-1-v2" \
# --record-video True \
# --record_video_w_freq:23