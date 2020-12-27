# RL Explorations in 0 AD
This repository contains experimental code exploring different applications of RL in 0 AD.

## Quick Start
First, clone the repo and install the python package locally.
```bash
pip install -e .
```

Next, install 0 AD (version > 0.23) and the [mod](https://trac.wildfiregames.com/wiki/Modding_Guide#Howtoinstallmods) containing the custom maps. Next, run an instance of the latest version of 0 AD (post 0.23) locally:

```bash
pyrogenesis --rl-interface=0.0.0.0:6000 --autostart-nonvisual
```

Next, train an agent using
```bash
python -m zero_ad_rl.train --env CavalryVsInfantry --run PPO --checkpoint-freq 100
```

Finally, generate some rollouts. (First, you may want to shutdown 0 AD and run it w/o the `--autostart-nonvisual` command.) To run an agent from a given checkpoint, use the following command:
```
python -m zero_ad_rl.rollout ~/ray_results/path/to/checkpoint/file --env CavalryVsInfantry --run PPO --steps 5000
```

## Environments
This contains a few different gym environments which use slightly different observation and action spaces:
- CavalryVsInfantry:
- MinimapCavVsInf:
- SimpleMinimapCavVsInf:
