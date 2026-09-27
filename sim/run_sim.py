"""
Run the stationary ai simulation

Usage:
python3 -m sim.run_sim --visualizer kit --enable_cameras
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac-dependent imports must follow application startup
import torch
from isaaclab.envs import ManagerBasedEnv

from .config.tasks.stack_cube_env_cfg import StackCubeEnvCfg


def main():
    cfg = StackCubeEnvCfg()
    cfg.sim.device = args.device

    env = ManagerBasedEnv(cfg=cfg)

    env.reset()

    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim),device=env.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(actions)


if __name__ == "__main__":
    main()
    