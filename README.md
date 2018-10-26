# HyperA2C
Distributional Atari agents generated from a HyperGAN - WIP

HyperA2C implements the A2C algorithm, and samples a new batch of agents at every time step. A distribution of policies is generated, and evaluated on the environment. Whatever reward is gained is used to update the agent generators. 

## Usage
There is a list of arguments in `args.py`, and right now its cofigured to use my server with the `--scratch=True` argument. But in general the usage goes like this. 

`python3 ga3c.py --exp state --env PongDeterministic-v0`
