"""
Run teleop in sim for the stationary ai setup.

Keyboard controls...
W/S: X
A/D: Y
Q/E: Z
Z/X, T/G, C/V: rotation
K: gripper
B: toggle arm
R: reset


Usage:
python3 -m sim.run_sim_teleop
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(
    visualizer="kit",
    enable_cameras=True,
    device="cuda:0",
    rendering_mode="balanced",
)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac-dependent imports must follow application startup
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from .config.tasks.stack_cube_env_cfg import StackCubeEnvCfg


class SimTeleop:
    def __init__(self):
        """
        Setup the simulation teleop.
        """

        self.active_arm_ = 0  # 0=left, 1=right
        self.grippers_ = [1.0, 1.0]  # [left, right], 1=open, -1=closed

        cfg = StackCubeEnvCfg()
        cfg.sim.device = args.device

        self.env_ = ManagerBasedEnv(cfg=cfg)
        self.env_.reset()

        self.keyboard_ = Se3Keyboard(
            Se3KeyboardCfg(
                    pos_sensitivity=0.002,
                    rot_sensitivity=0.01,
                )
            )
        self.keyboard_.add_callback("B", self.swap_arm)
        self.keyboard_.add_callback("K", self.toggle_gripper)
        self.keyboard_.add_callback("R", self.reset)

       
    def swap_arm(self):
        """
        Swap which arm is being currently controlled.
        """
        self.active_arm_ = 1 - self.active_arm_
        print(f"Controlling {'left' if self.active_arm_ == 0 else 'right'} arm")


    def toggle_gripper(self):
        """
        Close gripper (of active arm) if open or open if closed. 
        """
        self.grippers_[self.active_arm_] *= -1.0


    def reset(self):
        """
        Rest the sim and all variables.
        """
        self.env_.reset()
        self.keyboard_.reset()
        self.grippers_ = [1.0, 1.0]


    def run(self):
        """
        Start the sim loop.
        """
        actions = torch.zeros(
            (self.env_.num_envs, self.env_.action_manager.total_action_dim),
            device=self.env_.device,
        )
        while simulation_app.is_running():
            with torch.inference_mode():

                command = self.keyboard_.advance().to(self.env_.device)


                start = 0 if self.active_arm_ == 0 else 7
                actions[:, start:start + 6] = command[:6]
                actions[:, 6] = self.grippers_[0]
                actions[:, 13] = self.grippers_[1]

                self.env_.step(actions)


if __name__ == "__main__":
    sim = SimTeleop()

    sim.run()
    