#!/bin/bash

python3 -u launch_experiment.py \
--algo gpi_ls_discrete \
--env-id MOLunarLanderDR-v0 \
--seed 92 \
--num-timesteps 3000000 \
--gamma 0.99 \
--ref-point '-101' '-1001' '-101' '-101' \
--wandb-group 'domain_randomization' \
--test-generalization True \
--init-hyperparams "initial_epsilon:1.0" "final_epsilon:0.05" "epsilon_decay_steps:1000000" "target_net_update_freq:1000" "gradient_updates:1" "batch_size:128" "buffer_size:1000000" "net_arch:[256, 256, 256, 256]" \
--train-hyperparams num_eval_episodes_for_front:2 timesteps_per_iter:50000 eval_mo_freq:50000 \
--generalization-hyperparams num_eval_weights:100 num_eval_episodes:1 record_video_w_freq:230 \
--test-envs "MOLunarLanderDefault-v0,MOLunarLanderHighGravity-v0,MOLunarLanderWindy-v0,MOLunarLanderTurbulent-v0,MOLunarLanderLowMainEngine-v0,MOLunarLanderLowSideEngine-v0,MOLunarLanderStartRight-v0,MOLunarLanderStartLow-v0,MOLunarLanderHard-v0" \
# --record-video True \