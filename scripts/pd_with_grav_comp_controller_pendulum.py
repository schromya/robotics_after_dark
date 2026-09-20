import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("model/pendulum/pendulum.xml")
data = mujoco.MjData(model)

# 6 second as time step 0.002 second per steps
n_steps = 3000

kp = 5.0
kv = 3.0
target_angle = np.pi/2

def pd_controller(model, data, target_angle):
    current_angle = data.qpos[0]
    current_vel = data.qvel[0]
    # this will have constant error when cmd_torque = weight of the pendulum = kp * (target_anle - current_angle)
    # torque = kp * (target_angle - current_angle) - kv * current_vel

    pole_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pole")
    hinge_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    m = model.body_mass[pole_link_id]
    l_c = model.body_ipos[pole_link_id][2] - model.jnt_pos[hinge_joint_id][2]
    torque = kp * (target_angle - current_angle) - kv * current_vel - m * 9.81 * l_c * np.sin(current_angle)

    return torque

with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(n_steps):
        step_start = time.time()

        torque = pd_controller(model, data, target_angle)
        data.ctrl[0] = torque

        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)