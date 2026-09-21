"""
Stacking cube teleop.

Keyboard controls...
W/S: X
A/D: Y
Q/E: Z
Z/X, T/G, C/V: rotation
K: gripper
R: reset
"""

import os
import sys

from isaacsim import SimulationApp

# Must initialize SimulationApp before importing other Isaac Sim modules
simulation_app = SimulationApp({"headless": False})

from trossen_ai_isaac.controllers import RobotType, TrossenAIController

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.timeline
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg


# Default configuration constants
CUBE_SIZE = np.array([0.05, 0.05, 0.05])
CUBE_POSITION = np.array([0.35, -0.15, 0.03])
CUBE_ORIENTATION = np.array([1, 0, 0, 0])

HOME_POSITION = np.array([0.2, 0.0, 0.3])


DOWNWARD_ORIENTATION = np.array([[0.7071068, 0.0, 0.7071068, 0.0]])  

# Scene configuration
ROBOT_USD_PATH = "./trossen_ai_isaac/assets/robots/wxai/wxai_base.usd"
ROBOT_SCENE_PATH = "/World/wxai_robot"
GROUND_SCENE_PATH = "/World/ground"
CUBE_SCENE_PATH = "/World/Cube"

# Robot controller configuration
WXAI_ARM_DOF_INDICES = [0, 1, 2, 3, 4, 5]
WXAI_GRIPPER_DOF_INDEX = 6
WXAI_DEFAULT_DOF_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044]


class StackCubeTeleop:
    """Teleop for stacking cubes"""

    def __init__(self):
        self.control_position = HOME_POSITION.astype(float).copy()
        self.control_orientation = DOWNWARD_ORIENTATION.reshape(4).copy()
        self.keyboard = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=0.002,
                rot_sensitivity=0.01,
            )
        )
        self.keyboard.add_callback("R", self.reset)


    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, cube, and environment."""
        stage_utils.create_new_stage(template="sunlight")

        # Spawn robot in scene
        stage_utils.add_reference_to_stage(
            usd_path=ROBOT_USD_PATH,
            path=ROBOT_SCENE_PATH,
        )

        self.robot = TrossenAIController(
            robot_path=ROBOT_SCENE_PATH,
            robot_type=RobotType.WXAI,
            arm_dof_indices=WXAI_ARM_DOF_INDICES,
            gripper_dof_index=WXAI_GRIPPER_DOF_INDEX,
            default_dof_positions=WXAI_DEFAULT_DOF_POSITIONS,
        )

        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path()
            + "/Isaac/Environments/Grid/default_environment.usd",
            path=GROUND_SCENE_PATH,
        )

        visual_material = PreviewSurfaceMaterial("/Visual_materials/blue")
        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])

        cube_shape = Cube(
            paths=CUBE_SCENE_PATH,
            positions=CUBE_POSITION,
            orientations=CUBE_ORIENTATION,
            sizes=[1.0],
            scales=CUBE_SIZE,
            reset_xform_op_properties=True,
        )

        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
        self.cube = RigidPrim(paths=cube_shape.paths)
        cube_shape.apply_visual_materials(visual_material)

    def step(self):
        """Execute one simulation step.
        """
        command = self.keyboard.advance().detach().cpu().numpy()
        self.control_position += command[:3]

        self.robot.set_end_effector_pose(position=self.control_position,
            orientation=self.control_orientation)

        if command[-1] > 0:
            self.robot.open_gripper()
        else:
            self.robot.close_gripper()
            

    def reset(self) -> None:
        """Reset task to initial state."""
        self.keyboard.reset()
        self.control_position = HOME_POSITION.astype(float).copy()
        self.control_orientation = DOWNWARD_ORIENTATION.reshape(4).copy()

        self.robot.reset_to_default_pose()
        self.cube.set_world_poses(
            positions=CUBE_POSITION.reshape(1, -1),
            orientations=CUBE_ORIENTATION.reshape(1, -1),
        )




def main():
    simulation_app.update()

    sim = StackCubeTeleop()
    sim.setup_scene()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    task_completed = False
    sim.reset()

    while simulation_app.is_running():
        if SimulationManager.is_simulating():
            sim.step()


        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
