# RL Explorations in 0 AD
This repository contains experimental code exploring different applications of RL in 0 AD.

## Quick Start
First, install the required python dependencies and the custom map from `maps/`. (If you built 0 AD from source, you can simply copy the map files to `binaries/data/mods/public/maps/scenarios/`.) Next, run an instance of the latest version of 0 AD (post 0.23) locally:

```
pip install -r requirements.txt
pyrogenesis --rl-interface=0.0.0.0:6000 --autostart-nonvisual
```

Next, train an agent using
```bash
python train.py --env CavalryVsInfantry --run PPO --checkpoint-freq 100
```

Finally, generate some rollouts. (First, you may want to shutdown 0 AD and run it w/o the `--autostart-nonvisual` command.) To run an agent from a given checkpoint, use the following command:
```
python rollout.py ~/ray_results/path/to/checkpoint/file --env CavalryVsInfantry --run PPO --steps 5000
```

## Environments
This contains a few different gym environments which use slightly different observation and action spaces:
- CavalryVsInfantry:
- MinimapCavVsInf:
- SimpleMinimapCavVsInf:
