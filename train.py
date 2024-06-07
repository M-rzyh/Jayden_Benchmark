"""Launches an experiment on a given environment and algorithm.

Many parameters can be given in the command line, see the help for more infos.

Examples:
    python benchmark/launch_experiment.py --algo pcn --env-id deep-sea-treasure-v0 --num-timesteps 1000000 --gamma 0.99 --ref-point 0 -25 --auto-tag True --wandb-entity openrlbenchmark --seed 0 --init-hyperparams "scaling_factor:np.array([1, 1, 1])"
"""

import argparse
import os
import subprocess
import timeit
import torch
from distutils.util import strtobool

import mo_gymnasium as mo_gym
import numpy as np
import matplotlib.pyplot as plt
import requests
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers.record_video import RecordVideo
from mo_gymnasium.utils import MORecordEpisodeStatistics

from mo_utils.evaluation import seed_everything
from mo_utils.experiments import (
    ALGOS,
    ENVS_WITH_KNOWN_PARETO_FRONT,
    StoreDict,
)
from ued_utils import make_agent, FileWriter, safe_checkpoint, create_parallel_env, make_plr_args, save_images
from runners.experiment_runner import ExperimentRunner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="Name of the algorithm to run", choices=ALGOS.keys(), required=True)
    parser.add_argument("--env-id", type=str, help="MO-Gymnasium id of the environment to run", required=True)
    parser.add_argument("--num-timesteps", type=int, help="Number of timesteps to train for", required=True)
    parser.add_argument("--gamma", type=float, help="Discount factor to apply to the environment and algorithm", required=True)
    parser.add_argument(
        "--ref-point", type=float, nargs="+", help="Reference point to use for the hypervolume calculation", required=True
    )
    parser.add_argument("--seed", type=int, help="Random seed to use", default=42)
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity to use", required=False)
    parser.add_argument(
        "--record-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, the runs will be recorded with RecordVideo wrapper.",
    )
    parser.add_argument("--record-video-ep-freq", type=int, default=5, help="Record video frequency (in episodes).")
    parser.add_argument(
        "--init-hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Override hyperparameters to use for the initiation of the algorithm. Example: --init-hyperparams learning_rate:0.001 final_epsilon:0.1",
        default={},
    )

    parser.add_argument(
        "--train-hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Override hyperparameters to use for the train method algorithm. Example: --train-hyperparams num_eval_weights_for_front:10 timesteps_per_iter:10000",
        default={},
    )

    return parser.parse_args()



def main():
    args = parse_args()
    print(args)

    # === Determine device ====
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if 'cuda' in device.type:
        torch.backends.cudnn.benchmark = True
        print('Using CUDA\n')

    # === Configure checkpointing ===
    timer = timeit.default_timer
    initial_update_count = 0
    last_logged_update_at_restart = -1
    checkpoint_path = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, "model.tar"))
    )
    ## This is only used for the first iteration of finetuning
    if args.xpid_finetune:
        model_fname = f'{args.model_finetune}.tar'
        base_checkpoint_path = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid_finetune, model_fname))
        )

    def checkpoint(index=None):
        if args.disable_checkpoint:
            return
        safe_checkpoint({'runner_state_dict': train_runner.state_dict()},
                        checkpoint_path,
                        index=index,
                        archive_interval=args.archive_interval)
        logging.info("Saved checkpoint to %s", checkpoint_path)


    # if args.algo == "pgmorl":
    #     # PGMORL creates its own environments because it requires wrappers
    #     print(f"Instantiating {args.algo} on {args.env_id}")
    #     eval_env = mo_gym.make(args.env_id)
    #     algo = ALGOS[args.algo](
    #         env_id=args.env_id,
    #         origin=np.array(args.ref_point),
    #         gamma=args.gamma,
    #         log=True,
    #         seed=args.seed,
    #         wandb_entity=args.wandb_entity,
    #         **args.init_hyperparams,
    #     )
    #     print(algo.get_config())

    #     print("Training starts... Let's roll!")
    #     algo.train(
    #         total_timesteps=args.num_timesteps,
    #         eval_env=eval_env,
    #         ref_point=np.array(args.ref_point),
    #         known_pareto_front=None,
    #         **args.train_hyperparams,
    #     )

    # else:
    # env = mo_gym.make(args.env_id)
    # eval_env = mo_gym.make(args.env_id, render_mode="rgb_array" if args.record_video else None)
    # env = MORecordEpisodeStatistics(env, gamma=args.gamma)

    # if "highway" in args.env_id:
    #     env = FlattenObservation(env)
    #     eval_env = FlattenObservation(eval_env)

    # if args.record_video:
    #     eval_env = RecordVideo(
    #         eval_env,
    #         video_folder=f"videos/{args.algo}-{args.env_id}",
    #         episode_trigger=lambda ep: ep % args.record_video_ep_freq == 0,
    #     )

    print(f"Instantiating {args.algo} on {args.env_id}")
    if args.algo == "ols":
        args.init_hyperparams["experiment_name"] = "MultiPolicy MO Q-Learning (OLS)"
    elif args.algo == "gpi-ls":
        args.init_hyperparams["experiment_name"] = "MultiPolicy MO Q-Learning (GPI-LS)"

    # === Create parallel envs ===
    venv, ued_venv = create_parallel_env(args)

    seed_everything(args.seed)

    algo = ALGOS[args.algo](
        env=venv,
        gamma=args.gamma,
        log=True,
        seed=args.seed,
        wandb_entity=args.wandb_entity,
        **args.init_hyperparams,
    )
    # if args.env_id in ENVS_WITH_KNOWN_PARETO_FRONT:
    #     known_pareto_front = env.unwrapped.pareto_front(gamma=args.gamma)
    # else:
    #     known_pareto_front = None

    print(algo.get_config())

    # === Create runner ===
    train_runner = ExperimentRunner(
        args=args,
        venv=venv,
        agent=agent,
        ued_venv=ued_venv,
        adversary_agent=adversary_agent,
        adversary_env=adversary_env,
        flexible_protagonist=False,
        train=True,
        plr_args=plr_args,
        device=device)
    
    # === Set up Evaluator ===
    evaluator = None
    if args.test_env_names:
        evaluator = Evaluator(
            args.test_env_names.split(','),
            num_processes=args.test_num_processes,
            num_episodes=args.test_num_episodes,
            frame_stack=args.frame_stack,
            grayscale=args.grayscale,
            num_action_repeat=args.num_action_repeat,
            use_global_critic=args.use_global_critic,
            use_global_policy=args.use_global_policy,
            device=device)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    initial_update_count = 0

    # === Train ===
    print("Training starts... Let's roll!")
    for j in range(initial_update_count, num_updates):
        stats = train_runner.run()

        # === Perform logging ===
        if train_runner.num_updates <= last_logged_update_at_restart:
            continue

        log = (j % args.log_interval == 0) or j == num_updates - 1
        save_screenshot = \
            args.screenshot_interval > 0 and \
                (j % args.screenshot_interval == 0)

        if log:
            # Eval
            test_stats = {}
            if evaluator is not None and (j % args.test_interval == 0 or j == num_updates - 1):
                test_stats = evaluator.evaluate(train_runner.agents['agent'])
                stats.update(test_stats)
            else:
                stats.update({k:None for k in evaluator.get_stats_keys()})

            update_end_time = timer()
            num_incremental_updates = 1 if j == 0 else args.log_interval
            sps = num_incremental_updates*(args.num_processes * args.num_steps) / (update_end_time - update_start_time)
            update_start_time = update_end_time
            stats.update({'sps': sps})
            stats.update(test_stats) # Ensures sps column is always before test stats
            log_stats(stats)

        checkpoint_idx = getattr(train_runner, args.checkpoint_basis)

        if checkpoint_idx != last_checkpoint_idx:
            is_last_update = j == num_updates - 1
            if is_last_update or \
                (train_runner.num_updates > 0 and checkpoint_idx % args.checkpoint_interval == 0):
                checkpoint(checkpoint_idx)
                logging.info(f"\nSaved checkpoint after update {j}")
                logging.info(f"\nLast update: {is_last_update}")
            elif train_runner.num_updates > 0 and args.archive_interval > 0 \
                and checkpoint_idx % args.archive_interval == 0:
                checkpoint(checkpoint_idx)
                logging.info(f"\nArchived checkpoint after update {j}")

        if save_screenshot:
            level_info = train_runner.sampled_level_info
            venv.reset_agent()
            images = venv.get_images()
            if args.use_editor and level_info:
                save_images(
                    images[:args.screenshot_batch_size],
                    os.path.join(
                        screenshot_dir,
                        f"update{j}-replay{level_info['level_replay']}-n_edits{level_info['num_edits'][0]}.png"),
                    normalize=True, channels_first=False)
            else:
                save_images(
                    images[:args.screenshot_batch_size],
                    os.path.join(screenshot_dir, f'update{j}.png'),
                    normalize=True, channels_first=False)
            plt.close()
    # algo.train(
    #     total_timesteps=args.num_timesteps,
    #     eval_env=eval_env,
    #     ref_point=np.array(args.ref_point),
    #     known_pareto_front=known_pareto_front,
    #     **args.train_hyperparams,
    # )
    evaluator.close()
    venv.close()


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
