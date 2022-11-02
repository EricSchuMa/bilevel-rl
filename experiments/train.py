import argparse
import os
import sys

import mlflow

from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.monitor import Monitor

from utils.logger import MLflowOutputFormat, log_agent_params, log_env_params, log_meta_params, log_net_params, log_rew_params
from utils.callbacks import SUMOEvalCallback
from utils.configuration import generate_agent, generate_env_from_config, parse_config

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def main(args):
    agent_config, env_config, reward_config, net_arch, train_config, meta_config = parse_config(args)
    env = generate_env_from_config(env_config, reward_config)
    eval_env = Monitor(generate_env_from_config(env_config, reward_config))
    agent = generate_agent(env, agent_config, env_config, net_arch, reward_config, meta_config)

    loggers = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()])

    mlflow.set_experiment(experiment_name=train_config.get('experiment_name'))
    with mlflow.start_run(run_name=train_config.get('run_name')):
        log_agent_params(agent_config)
        log_env_params(env_config)
        log_meta_params(meta_config)
        log_rew_params(reward_config)
        log_net_params(net_arch)

        eval_callback = SUMOEvalCallback(eval_env,
                                         best_model_save_path='./best_models/',
                                         eval_freq=train_config.getint('eval_freq'),  # log every episode
                                         n_eval_episodes=1  # currently only one eval_episode is supported
                                         )

        agent.set_logger(loggers)
        agent.learn(total_timesteps=train_config.getint('total_timesteps'),
                    log_interval=1,
                    callback=eval_callback)

        if train_config.getboolean('save_model'):
            agent.save(train_config.get('model_save_path'))
            mlflow.log_artifact(train_config.get('model_save_path') + '.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help="experiment config path")
    parser.add_argument('--queue', type=str, required=False, help="initial weight for queue")
    parser.add_argument('--brake', type=str, required=False, help="initial weight for brake")
    arguments = parser.parse_args()

    main(arguments)
