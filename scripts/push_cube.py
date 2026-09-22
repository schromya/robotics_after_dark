import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard

model = mujoco.MjModel.from_xml_path("model/push_cube/push_cube_env.xml")
data = mujoco.MjData(model)

# control input is a target position for the position actuators: ctrl[0] = x_slider, ctrl[1] = y_slider
# speed in (m/s)
SPEED = 0.25

held = set()
listener = keyboard.Listener(on_press=held.add, on_release=held.discard)
listener.start()

print("***Use arrow keys to control the end effector x and y position***")

with mujoco.viewer.launch_passive(model, data) as viewer:

    # specifing this in xml might get reconfigure by simple mouse click
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -90
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0, 0, 0.5] # point it at the table area

    while viewer.is_running():
        step_start = time.time()

        dt = model.opt.timestep
        if keyboard.Key.up in held:
            data.ctrl[1] += SPEED * dt
        if keyboard.Key.down in held:
            data.ctrl[1] -= SPEED * dt
        if keyboard.Key.right in held:
            data.ctrl[0] += SPEED * dt
        if keyboard.Key.left in held:
            data.ctrl[0] -= SPEED * dt

        # keep the target inside ctrlrange.
        data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

        mujoco.mj_step(model, data)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)