
from ..stationary_scene_cfg import StationarySceneCfg

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnvCfg, mdp, ViewerCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

############################################# HELPERS ##############################################
def arm_action(side: str) -> mdp.DifferentialInverseKinematicsActionCfg:
    """
    Template for cartesian arm controller.
    Args:
        side: 'left' or 'right'.
    Returns:
        IK action config for the arm.
    """
    return mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[f"follower_{side}_joint_[0-5]"],
        body_name=f"follower_{side}_link_5",
        scale=1.0,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
    )


def gripper_action(side: str) -> mdp.BinaryJointPositionActionCfg:
    """
    Template for gripper controller.
    Args:
        side: 'left' or 'right'.
    Returns:
        Action config for the gripper.
    """
    joint = f"follower_{side}_left_carriage_joint"
    return mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[joint],
        open_command_expr={joint: 0.044},
        close_command_expr={joint: 0.0},
    )
####################################################################################################

@configclass
class StackCubeSceneCfg(StationarySceneCfg):

    red_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube1",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.4,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.1, 0.55),
        ),
    )
    blue_cube = red_cube.replace(
        prim_path="{ENV_REGEX_NS}/Cube2",
        spawn=red_cube.spawn.replace(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.1, 0.55),
        ),
    )


@configclass
class ActionsCfg:
    left_arm = arm_action("left")
    left_gripper = gripper_action("left")
    right_arm = arm_action("right")
    right_gripper = gripper_action("right")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class StackCubeEnvCfg(ManagerBasedEnvCfg):
    scene: StackCubeSceneCfg = StackCubeSceneCfg(
        num_envs=1,
        env_spacing=2.5,
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()

    viewer: ViewerCfg = ViewerCfg(
        eye=(-2.0, 0.0, 1.5),
        lookat=(0.0, 0.0, 1.0),
    )

    def __post_init__(self):
        self.sim.dt = 1.0 / 60.0
        self.decimation = 2  # One action every 2 physics steps: 30 Hz
        self.sim.render_interval = 4 # Render every 4 physics steps: 15 Hz.