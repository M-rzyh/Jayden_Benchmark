#!/bin/bash


python3 -u launch_experiment.py \
--algo capql \
--env-id MOHopperUED-v5 \
--num-timesteps 20000000 \
--gamma 0.99 \
--auto-tag False \
--ref-point '0.0' '-20.0' '-20.0' \
--init-hyperparams alpha:0.2 \
--train-hyperparams test_generalization:True eval_freq:5 \
--test-envs "MOHopperUED-v5,MOHopperLight-v5,MOHopperHeavy-v5,MOHopperSlippery-v5,MOHopperHighDamping-v5" \