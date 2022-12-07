# Learning intrinsic rewards with bilevel reinforcement learning
This repository holds code for a bilevel meta-gradient reinforcement learning variant of DQN: **Intrinsic Reward Deep Q-Network (IRDQN)**.

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
[![DQN-Agent with brake and queue reward](https://img.youtube.com/vi/UjkyiCG75ms/0.jpg)](https://www.youtube.com/watch?v=UjkyiCG75ms)

### 2: IRDQN trained with brake and queue reward
[![IRDQN-Agent with brake and queue reward](https://img.youtube.com/vi/Cu0ZR0lyRnw/0.jpg)](https://www.youtube.com/watch?v=Cu0ZR0lyRnw)
