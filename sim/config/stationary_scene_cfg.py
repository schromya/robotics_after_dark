from .stationary_cfg import STATIONARY_CFG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import GroundPlaneCfg, UsdFileCfg, DomeLightCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

D405_RGB_CFG = CameraCfg(
    spawn=None,
    update_period=1.0 / 30.0,
    width=1280,
    height=720,
    data_types=["rgb"],
)

@configclass
class StationarySceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = STATIONARY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
        init_state=STATIONARY_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.5),
        ),
    )


    ##################### Cameras #####################
    high_camera = D405_RGB_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/cam_high_link/cam_high_color_frame/cam_high",
    )

    low_camera = D405_RGB_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/cam_low_link/cam_low_color_frame/cam_low",
    )

    left_follower_camera = D405_RGB_CFG.replace(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/follower_left_camera_link/"
            "follower_left_camera_color_frame/cam_left"
        ),
    )

    right_follower_camera = D405_RGB_CFG.replace(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/follower_right_camera_link/"
            "follower_right_camera_color_frame/cam_right"
        ),
    )


    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
