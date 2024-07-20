#!/bin/bash


python3 -u launch_experiment.py \
--algo gpi_ls_continuous \
--env-id MOHopperDR-v5 \
--num-timesteps 20000000 \
--gamma 0.99 \
--auto-tag False \
--ref-point '0.0' '-20.0' '-20.0' \
--init-hyperparams per:False \
--train-hyperparams test_generalization:True eval_mo_freq:5000 eval_freq:10000000000000  \
--test-envs "MOHopperDefault-v5,MOHopperLight-v5,MOHopperHeavy-v5,MOHopperSlippery-v5,MOHopperLowDamping-v5" \