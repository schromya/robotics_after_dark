import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("model/pendulum/pendulum.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)