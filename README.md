# Learning intrinsic rewards with bilevel reinforcement learning
This repository holds code for a bilevel meta-gradient reinforcement learning variant of DQN: **Intrinsic Reward Deep Q-Network (IRDQN)**.

## Abstract
Reinforcement learning teaches intelligent agents how to act in dynamic environments through scalar rewards. There are many feasible ways to define a reward for solving a given task. However, designing rewards requires extensive domain knowledge and engineering efforts.

A popular application for the reinforcement learning approach is traffic signal control, where agents control traffic lights at intersections. Works in this field predominantly define rewards that minimize vehicle travel times. Because travel times are inefficient as a reward – they are delayed, sparse, and influenced by factors outside the agent’s control – researchers use combinations of other traffic metrics as the reward.

This approach can be problematic because specific weight choices might lead to considerable differences in agent performance and learning efficiency. This work aims to mitigate the difficulty of choosing weightings. We investigate this problem with the sustainability-oriented goal of reducing CO2 emissions in traffic.

We studied how sensitive agents are to pre-defined weightings of combined rewards. Our insight for mitigating this sensitivity was to relax the assumption that the weightings are pre-defined by an agent designer. We propose a method that treats the weightings as intrinsic to the agent. Our implementation builds on the famous Deep Q-Network; hence, we call our algorithm the Intrinsic Reward Deep Q-Network (IRDQN). Based on a meta-learning assumption, we consider that the agent’s experience contains knowledge about learning itself. In this work, the IRDQN agent meta-learns reward weights online through gradient descent.

Our results indicate that some task objectives (e.g., a CO2 emission reward) are inefficient for learning, agents are sensitive to combined reward weightings, and meta-learning these weightings can benefit agent performance and learning efficiency. The proposed IRDQN agent learned reward weights that lead to the desired behavior of reducing CO2 emissions.

Our study revealed some interesting limitations of the IRDQN algorithm, such as a lack of exploration and its sensitivity to imbalanced weights. In the future, we plan to investigate these issues further to see if we can improve the algorithm’s performance.

## Setup
1. Clone this repository (including submodules): `git clone --recurse_submodules https://github.com/EricSchuMa/bilevel-rl.git`.
2. Follow the intructions in `sumo_rl/README.md` for installing the SUMO traffic simulator.
3. Create a conda envrionment with python 3.8: `conda create -n bilevel-rl python=3.8`.
4. Activate the conda environment: `conda activate bilevel-rl`.
5. Add your local repository path to the python PATH variable: `export PYTHONPATH="${PYTHONPATH}:{/path/to/bilevel-rl}`.
6. Install the requirements with pip: `pip install -r requirements.txt`.

## Running
From the project root, run the following command to train a DQN or IRDQN agent:
```bash
python experiments/train.py --config-path experiments/configs/{config}
```
where {config} should be replaced by a config file. Available config files are `experiments/configs/DQN.ini` and `experiments/configs/IRDQN.ini`.

The training logs are saved to the folder `mlruns`. You can access the logs by running a MLflow server:
```bash
mlflow ui
```

## Examples
### 1: DQN trained with brake and queue reward
[Video of DQN with brake and queue weights of 0.5 controlling an intersection in SUMO](https://www.youtube.com/watch?v=UjkyiCG75ms)

### 2: IRDQN trained with brake and queue reward
[Video of IRDQN with initial brake and queue weights of 0.5 controlling an intersection in SUMO](https://www.youtube.com/watch?v=Cu0ZR0lyRnw)
