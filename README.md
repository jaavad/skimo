# Skill-based Model-based Reinforcement learning (SkiMo)

[[Project website](https://clvrai.com/skimo)] [[Paper](https://openreview.net/forum?id=iVxy2eO601U)] [[arXiv](https://arxiv.org/abs/2207.07560)]

This project is a PyTorch implementation of [Skill-based Model-based Reinforcement Learning](https://clvrai.com/skimo), published in CoRL 2022.


## Files and Directories
* `run.py`: launches an appropriate trainer based on algorithm
* `skill_trainer.py`: trainer for skill-based approaches
* `skimo_agent.py`: model and training code for SkiMo
* `skimo_rollout.py`: rollout with SkiMo agent
* `spirl_tdmpc_agent.py`: model and training code for SPiRL+TD-MPC
* `spirl_tdmpc_rollout.py`: rollout with SPiRL+TD-MPC
* `spirl_dreamer_agent.py`: model and training code for SPiRL+Dreamer
* `spirl_dreamer_rollout.py`: rollout with SPiRL+Dreamer
* `spirl_trainer.py`: trainer for SPiRL
* `spirl_agent.py`: model for SPiRL
* `config/`: default hyperparameters
* `calvin/`: CALVIN environments
* `d4rl/`: [D4RL](https://github.com/kpertsch/d4rl) environments forked by Karl Pertsch. The only change from us is in the [installation](d4rl/setup.py#L15) command
* `envs/`: environment wrappers
* `spirl/`: [SPiRL code](https://github.com/clvrai/spirl)
* `data/`: offline data directory
* `rolf/`: implementation of RL algorithms from [robot-learning](https://github.com/youngwoon/robot-learning) by Youngwoon Lee
* `log/`: training log, evaluation results, checkpoints


## Prerequisites
* Ubuntu 20.04
* Python 3.9
* MuJoCo 2.1


## Installation

1. Clone this repository.
```bash
git clone --recursive git@github.com:clvrai/skimo.git
cd skimo
```

2. Create a virtual environment
```bash
conda create -n skimo_venv python=3.9
conda activate skimo_venv
```

3. Install MuJoCo 2.1
* Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
* Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

4. Install packages
```bash
sh install.sh
```

# Pre-trained Models

We provide pre-trained model checkpoints in the kitchen, maze, and calvin environments. Using these models you can skip the pre-training step and directly run SkiMo on downstream tasks.

To download model checkpoints, run:
```bash
# Maze
mkdir -p log/maze.skimo.pretrain.test.0/ckpt
cd log/maze.skimo.pretrain.test.0/ckpt
gdown 162cmAz9E9D3DyfUSihItI5gae9Z_DdoY
cd ../../..


Now, for downstream RL, you can simply run
```bash
# Maze
python run.py --config-name skimo_maze run_prefix=test gpu=0 wandb=true rolf.phase=rl rolf.pretrain_ckpt_path=log/maze.skimo.pretrain.test.0/ckpt/maze_ckpt_00000140000.pt


## Usage
Commands for SkiMo and all baselines. Results will be logged to [WandB](https://wandb.ai/site). Before running the commands below, **please change the wandb entity** in [run.py#L36](run.py#L36) to match your account.


## Troubleshooting

### Failed building wheel for mpi4py
Solution: install `mpi4py` with conda instead, which requires a lower version of python.
```bash
conda install python==3.8
conda install mpi4py
```
Now you can re-run `sh install.sh`.

### MacOS mujoco-py compilation error
See [this](https://github.com/openai/mujoco-py#youre-on-macos-and-you-see-clang-error-unsupported-option--fopenmp). In my case, I needed to change `/usr/local/` to `/opt/homebrew/` in all paths.


## Citation

If you find our code useful for your research, please cite:
```
@inproceedings{shi2022skimo,
  title={Skill-based Model-based Reinforcement Learning},
  author={Lucy Xiaoyang Shi and Joseph J. Lim and Youngwoon Lee},
  booktitle={Conference on Robot Learning},
  year={2022}
}
```


## References
* This code is based on Youngwoon's robot-learning repo: https://github.com/youngwoon/robot-learning
* SPiRL: https://github.com/clvrai/spirl
* TD-MPC: https://github.com/nicklashansen/tdmpc
* Dreamer: https://github.com/danijar/dreamer
* D4RL: https://github.com/rail-berkeley/d4rl
* CALVIN: https://github.com/mees/calvin
