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
python3 -m sim.run_sim_teleop --record-dir data/top_camera --record-dt 1.0
"""

import argparse
import math
from datetime import datetime
from pathlib import Path
import time
from PIL import Image

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--record-dir", type=Path, default=None,
    help="Save top-camera PNGs in a new session folder.",
)
parser.add_argument(
    "--record-dt", type=float, default=1.0,
    help="Wall-clock interval between saved images in seconds (default: 0.5).",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(
    visualizer="kit",
    enable_cameras=True,
    device="cuda:0",
    rendering_mode="balanced",
)
args = parser.parse_args()
if not math.isfinite(args.record_dt) or args.record_dt <= 0:
    parser.error("--record-dt must be a finite number greater than zero")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac-dependent imports must follow application startup
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from .config.tasks.stack_cube_env_cfg import StackCubeEnvCfg


class SimTeleop:
    def __init__(
        self, device: str, record_dir: str = None, record_dt: float = 0.5
    ):
        """
        Setup the simulation teleop.
        Args:
            device: Device used by the simulation, such as "cuda:0" or "cpu".
            record_dir: Parent directory for recording sessions, or None to disable recording.
            record_dt: Positive, finite wall-clock interval between saved images in seconds.
                Defaults to 0.5; this does not change the simulation timestep.
        """
        if not math.isfinite(record_dt) or record_dt <= 0:
            raise ValueError("record_dt must be a finite number greater than zero")
        self.record_dt_ = record_dt

        self.active_arm_ = 0  # 0=left, 1=right
        self.grippers_ = [1.0, 1.0]  # [left, right], 1=open, -1=closed

        cfg = StackCubeEnvCfg()
        cfg.sim.device = device

        self.env_ = ManagerBasedEnv(cfg=cfg)
        self.env_.reset()

        self.record_dir_ = None
        self.capture_index_ = 0
        if record_dir is not None:
            session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.record_dir_ = Path(record_dir) / session
            self.record_dir_.mkdir(parents=True, exist_ok=False)
            print(f"Recording top-camera images to {self.record_dir_.resolve()}")

        self.keyboard_ = Se3Keyboard(
            Se3KeyboardCfg(
                    pos_sensitivity=0.005,
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


    def save_image(self, image: torch.Tensor):
        """
        Save an image as a timestamped PNG in the current recording directory.
        Args:
            image: uint8 RGB or RGBA tensor shaped [height, width, channels].
                Alpha is discarded; resolution and RGB pixel values are preserved.
        Returns:
            Path to the saved image, or None if recording is disabled.
        """
        if self.record_dir_ is None:
            return None

        frame = image[..., :3].detach().cpu().numpy()
        path = self.record_dir_ / f"{self.capture_index_:06d}.png"
        Image.fromarray(frame).save(path)
        self.capture_index_ += 1
        return path


    def run(self):
        """
        Start the sim loop.
        """
        actions = torch.zeros(
            (self.env_.num_envs, self.env_.action_manager.total_action_dim),
            device=self.env_.device,
        )
        next_capture = time.monotonic() + self.record_dt_
        while simulation_app.is_running():
            with torch.inference_mode():

                command = self.keyboard_.advance().to(self.env_.device)


                start = 0 if self.active_arm_ == 0 else 7
                actions[:, start:start + 6] = command[:6]
                actions[:, 6] = self.grippers_[0]
                actions[:, 13] = self.grippers_[1]

                observations, _ = self.env_.step(actions)

                if self.record_dir_ is not None and time.monotonic() >= next_capture:
                    self.save_image(observations["images"]["high_camera"][0])
                    # Skip missed intervals instead of saving catch-up duplicates.
                    next_capture = time.monotonic() + self.record_dt_


if __name__ == "__main__":
    sim = SimTeleop(
        device=args.device,
        record_dir=args.record_dir,
        record_dt=args.record_dt,
    )

    sim.run()
    