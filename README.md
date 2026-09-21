# RAD: Robotics After Dark

Repo for the eclectic pursuits of the RAD team.

## Getting Started

Clone the repo and check out this branch:

```bash
git clone https://github.com/schromya/robotics_after_dark.git
git checkout sim_mujoco
```

We use [UV](https://docs.astral.sh/uv/getting-started/installation/) to make our life easy. Install the dependencies with:

```bash
uv sync
```

To make sure the installation went well:

```bash
source .venv/bin/activate
python scripts/basic_pendulum_sim.py
```

Do not forget to swing the pendulum!!!

Go to Control and set the torque value in hinge_motor in the MuJoCo viewer.

Wanna do even cooler stuff? Let's try a PD with gravity compensation controller to control the pendulum:

```bash
python scripts/pd_with_grav_comp_controller_pendulum.py
```

**Note:** You can play with the PD gains and other settings in `scripts/pd_with_grav_comp_controller_pendulum.py`.

## Troubleshooting

### VS Code IntelliSense does not work for `mujoco`

**Problem:** No autocomplete or hover info for `mujoco` (for example `MjModel`, `MjData`). The `mujoco` package from PyPI is compiled code and does not include `.pyi` type stub files, so VS Code (Pylance) cannot see what is inside it.

**Fix:**

1. Generate the stubs into a `typings` folder:

   ```bash
   uv pip install pybind11-stubgen
   .venv/bin/pybind11-stubgen mujoco -o typings --ignore-all-errors
   ```

2. Tell Pylance where the stubs and the virtual environment are. Add this to `.vscode/settings.json`:

   ```json
   {
       "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
       "python.analysis.stubPath": "typings"
   }
   ```

3. Restart VS code if needed.