# Simple Example of RL in 0 AD
This repository contains a simple example training an RL agent to learn to control an army of cavalry in a small scenario.

## Quick Start
First, install the required python dependencies  and run an instance of 0 AD ([built with RL interface support](https://code.wildfiregames.com/D2199)) locally:

```
pip install -r requirements.txt
pyrogenesis --rpc-server=0.0.0.0:50050 --autostart-nonvisual
```

Next, train an agent using
```
python train.py --env CavalryVsInfantry --run PPO --checkpoint-freq 50000 --steps 500000
```

Finally, generate some rollouts. (First, you may want to shutdown 0 AD and run it w/o the `--autostart-nonvisual` command.) To run an agent from a given checkpoint, use the following command:
```
python rollout.py ~/ray_results/path/to/checkpoint/file --env CavalryVsInfantry --run PPO --steps 5000
```
