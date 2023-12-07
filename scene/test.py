import mujoco
import mujoco.viewer
from os.path import abspath, dirname, join
import faulthandler
faulthandler.enable()
path = abspath(dirname(__file__))

m = mujoco.MjModel.from_xml_path(join(path, 'deformable.xml'))
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()
