
from ..stationary_scene_cfg import StationarySceneCfg

from isaaclab.envs import ManagerBasedEnvCfg, mdp, ViewerCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass



@configclass
class StackCubeSceneCfg(StationarySceneCfg):

    red_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube1",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.1, -0.1, 0.55),
        ),
    )
    blue_cube = red_cube.replace(
        prim_path="{ENV_REGEX_NS}/Cube2",
        spawn=red_cube.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.1, 0.1, 0.55),
        ),
    )


@configclass
class ActionsCfg:
    arm_positions = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["follower_(left|right)_joint_[0-5]"],
        scale=1.0,
        use_default_offset=False,
    )


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