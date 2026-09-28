import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_USD_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "trossen_ai_isaac", "assets", "robots", "stationary_ai", "stationary_ai.usd"
    )
)

STATIONARY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # TODO (@schromya): re-enable when sagging figured out
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "follower_left_joint_[0-5]": 0.0,
            "follower_left_left_carriage_joint": 0.0,
            "follower_right_joint_[0-5]": 0.0,
            "follower_right_left_carriage_joint": 0.0,
        },
    ),
    # TODO (@schromya): Figure out why stiffness/damping is so high
    actuators={
        "arm_actuators": ImplicitActuatorCfg(
            joint_names_expr=["follower_(left|right)_joint_[0-5]"],
            friction=0.01,
            armature=0.01,
            stiffness={
                "follower_(left|right)_joint_0": 150.0,
                "follower_(left|right)_joint_1": 2000.0,
                "follower_(left|right)_joint_2": 2000.0,
                "follower_(left|right)_joint_3": 500.0,
                "follower_(left|right)_joint_4": 500.0,
                "follower_(left|right)_joint_5": 500.0,
            },
            damping={
                "follower_(left|right)_joint_0": 5.0,
                "follower_(left|right)_joint_1": 5.0,
                "follower_(left|right)_joint_2": 5.0,
                "follower_(left|right)_joint_3": 5.0,
                "follower_(left|right)_joint_4": 5.0,
                "follower_(left|right)_joint_5": 5.0,
            },
        ),

        # right_carriage_joint is a mimic joint specified in USD file
        "gripper_actuators": ImplicitActuatorCfg(
            joint_names_expr=["follower_(left|right)_left_carriage_joint"],
            friction=0.01,
            armature=0.01,
            stiffness=1000.0,
            damping=5.0,
        ),
    },

)



__all__ = ["STATIONARY_CFG"]
