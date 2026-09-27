"""
Displays stationary ai setup.

Usage:
python3 -m sim.view_scene --visualizer kit
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac-dependent imports must follow application startup.
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

from .config.stationary_scene_cfg import StationarySceneCfg


def main():
    sim = SimulationContext(
        SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    sim.set_camera_view(
        eye=[2.0, 0.0, 1.5],
        target=[0.0, 0.0, 1.0],
    )

    scene = InteractiveScene(
        StationarySceneCfg(num_envs=1, env_spacing=2.5)
    )
    sim.reset()

    robot = scene["robot"]
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_position_to_sim_index(position=joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
    scene.reset()

    while simulation_app.is_running():
        robot.set_joint_position_target_index(target=joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == '__main__':
    main()