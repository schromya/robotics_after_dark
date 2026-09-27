from .stationary_cfg import STATIONARY_CFG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import GroundPlaneCfg, UsdFileCfg, DomeLightCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


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

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
