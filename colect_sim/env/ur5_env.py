import numpy as np
import time

from colect_sim.controller.operational_space_controller import ImpedanceController, ComplianceController, OperationalSpaceController, TargetType
from colect_sim.controller.joint_effort_controller import JointEffortController
from colect_sim.controller.joint_velocity_controller import JointVelocityController
from colect_sim.controller.joint_position_controller import JointPositionController
from colect_sim.controller.force_torque_sensor_controller import ForceTorqueSensorController
from colect_sim.env.mujoco_env import MujocoEnv
from colect_sim.utils.mujoco_utils import MujocoModelNames
from mujoco import viewer
from os import path

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 0.0,
    "elevation": -20.0,
    "lookat": np.array([0, 0, 1]),
}

class BaseRobot(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 42,
    }

    def __init__(
        self,
        model_path="../../scene/scene.xml",
        frame_skip=12,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

        super().__init__(
            xml_file_path,
            frame_skip,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.init_qvel = self.data.qvel.copy()
        self.init_ctrl = self.data.ctrl.copy()


        self.model_names = MujocoModelNames(self.model) 


        
        

        # self.controller = ImpedanceController(
        #     model=self.model, 
        #     data=self.data, 
        #     model_names=self.model_names,
        #     eef_name='robot0:eef_site', 
        #     joint_names=[
        #         'robot0:ur5e:shoulder_pan_joint',
        #         'robot0:ur5e:shoulder_lift_joint',
        #         'robot0:ur5e:elbow_joint',
        #         'robot0:ur5e:wrist_1_joint',
        #         'robot0:ur5e:wrist_2_joint',
        #         'robot0:ur5e:wrist_3_joint',
        #     ],
        #     actuator_names=[
        #         'robot0:ur5e:shoulder_pan',
        #         'robot0:ur5e:shoulder_lift',
        #         'robot0:ur5e:elbow',
        #         'robot0:ur5e:wrist_1',
        #         'robot0:ur5e:wrist_2',
        #         'robot0:ur5e:wrist_3',
        #     ],
        #     min_effort=[-150, -150, -150, -150, -150, -150],
        #     max_effort=[150, 150, 150, 150, 150, 150],
        #     null_damp_kv=10,
        # )

        self.controller = OperationalSpaceController(
            model=self.model, 
            data=self.data, 
            model_names=self.model_names,
            eef_name='eef_site', 
            joint_names=[
                'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint',
            ],
            actuator_names=[
                'shoulder_pan',
                'shoulder_lift',
                'elbow',
                'wrist_1',
                'wrist_2',
                'wrist_3',
            ],
            min_effort=[-150, -150, -150, -150, -150, -150],
            max_effort=[150, 150, 150, 150, 150, 150],
            target_type=TargetType.POSE,
            kp=200.0,
            ko=200.0,
            kv=50.0,
            vmax_xyz=0.2,
            vmax_abg=1,
            null_damp_kv=10,
        )


        # self.controller = JointEffortController(
        #     model=self.model, 
        #     data=self.data, 
        #     model_names=self.model_names,
        #     eef_name='robot0:eef_site', 
        #     joint_names=[
        #         'robot0:ur5e:shoulder_pan_joint',
        #         'robot0:ur5e:shoulder_lift_joint',
        #         'robot0:ur5e:elbow_joint',
        #         'robot0:ur5e:wrist_1_joint',
        #         'robot0:ur5e:wrist_2_joint',
        #         'robot0:ur5e:wrist_3_joint',
        #     ],
        #     actuator_names=[
        #         'robot0:ur5e:shoulder_pan',
        #         'robot0:ur5e:shoulder_lift',
        #         'robot0:ur5e:elbow',
        #         'robot0:ur5e:wrist_1',
        #         'robot0:ur5e:wrist_2',
        #         'robot0:ur5e:wrist_3',
        #     ],
        #     min_effort=[-150, -150, -150, -150, -150, -150],
        #     max_effort=[150, 150, 150, 150, 150, 150],
        # )

        # self.controller = JointVelocityController(
        #     model=self.model, 
        #     data=self.data, 
        #     model_names=self.model_names,
        #     eef_name='robot0:eef_site', 
        #     joint_names=[
        #         'robot0:ur5e:shoulder_pan_joint',
        #         'robot0:ur5e:shoulder_lift_joint',
        #         'robot0:ur5e:elbow_joint',
        #         'robot0:ur5e:wrist_1_joint',
        #         'robot0:ur5e:wrist_2_joint',
        #         'robot0:ur5e:wrist_3_joint',
        #     ],
        #     actuator_names=[
        #         'robot0:ur5e:shoulder_pan',
        #         'robot0:ur5e:shoulder_lift',
        #         'robot0:ur5e:elbow',
        #         'robot0:ur5e:wrist_1',
        #         'robot0:ur5e:wrist_2',
        #         'robot0:ur5e:wrist_3',
        #     ],
        #     min_effort=[-150, -150, -150, -150, -150, -150],
        #     max_effort=[150, 150, 150, 150, 150, 150],
        #     min_velocity=[-1, -1, -1, -1, -1, -1],
        #     max_velocity=[1, 1, 1, 1, 1, 1],
        #     kp=[100, 100, 100, 100, 100, 100],
        #     ki=0,
        #     kd=0,
        # )

        # self.controller = JointPositionController(
        #     model=self.model, 
        #     data=self.data, 
        #     model_names=self.model_names,
        #     eef_name='robot0:eef_site', 
        #     joint_names=[
        #         'robot0:ur5e:shoulder_pan_joint',
        #         'robot0:ur5e:shoulder_lift_joint',
        #         'robot0:ur5e:elbow_joint',
        #         'robot0:ur5e:wrist_1_joint',
        #         'robot0:ur5e:wrist_2_joint',
        #         'robot0:ur5e:wrist_3_joint',
        #     ],
        #     actuator_names=[
        #         'robot0:ur5e:shoulder_pan',
        #         'robot0:ur5e:shoulder_lift',
        #         'robot0:ur5e:elbow',
        #         'robot0:ur5e:wrist_1',
        #         'robot0:ur5e:wrist_2',
        #         'robot0:ur5e:wrist_3',
        #     ],
        #     min_effort=[-150, -150, -150, -150, -150, -150],
        #     max_effort=[150, 150, 150, 150, 150, 150],
        #     min_position=[-1, -1, -1, -1, -1, -1],
        #     max_position=[1, 1, 1, 1, 1, 1],
        #     kp=[100, 100, 100, 100, 100, 100],
        #     kd=[20, 20, 20, 20, 20, 20],
        # )


        self.init_qpos_config = {
            "shoulder_pan_joint": 0,
            "shoulder_lift_joint": -np.pi / 2.0,
            "elbow_joint": -np.pi / 2.0,
            "wrist_1_joint": -np.pi / 2.0,
            "wrist_2_joint": np.pi / 2.0,
            "wrist_3_joint": 0,
        }
        for joint_name, joint_pos in self.init_qpos_config.items():
            joint_id = self.model_names.joint_name2id[joint_name]
            qpos_id = self.model.jnt_qposadr[joint_id]
            self.init_qpos[qpos_id] = joint_pos



        # self.controller = ComplianceController(
        #     model=self.model, 
        #     data=self.data, 
        #     model_names=self.model_names,
        #     eef_name='robot0:eef_site', 
        #     joint_names=[
        #         'robot0:ur5e:shoulder_pan_joint',
        #         'robot0:ur5e:shoulder_lift_joint',
        #         'robot0:ur5e:elbow_joint',
        #         'robot0:ur5e:wrist_1_joint',
        #         'robot0:ur5e:wrist_2_joint',
        #         'robot0:ur5e:wrist_3_joint',
        #     ],
        #     actuator_names=[
        #         'robot0:ur5e:shoulder_pan',
        #         'robot0:ur5e:shoulder_lift',
        #         'robot0:ur5e:elbow',
        #         'robot0:ur5e:wrist_1',
        #         'robot0:ur5e:wrist_2',
        #         'robot0:ur5e:wrist_3',
        #     ],
        #     min_effort=[-150, -150, -150, -150, -150, -150],
        #     max_effort=[150, 150, 150, 150, 150, 150],
        #     min_velocity=[-1, -1, -1, -1, -1, -1],
        #     max_velocity=[1, 1, 1, 1, 1, 1],
        #     kp_jnt_vel=[100, 100, 100, 100, 100, 100],
        #     ki_jnt_vel=0,
        #     kd_jnt_vel=0,
        #     kp=[10, 10, 10, 10, 10, 10],
        #     ki=[0, 0, 0, 0, 0, 0],
        #     kd=[3, 3, 3, 8, 8, 8],
        #     control_period=self.model.opt.timestep,
        #     ft_sensor_site='robot0:eef_site',
        #     force_sensor_name='robot0:eef_force',
        #     torque_sensor_name='robot0:eef_torque',
        #     subtree_body_name='robot0:ur5e:wrist_3_link',
        #     ft_smoothing_factor=0,
        # )

        self.viewer = viewer.launch_passive(self.model, self.data)
        self.wait_for_viewer()

    def wait_for_viewer(self):
        timeout = 5.0
        start_time = time.time()
        while not self.viewer.is_running():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise RuntimeError("Timeout while waiting for viewer to start.")

    def step(self, action):
        target_pose = np.array([0.25, 0.25, 0.1,0,0,0,-1])
        """target_twist = np.array([0,0,0,0,0,0])
        target_wrench = np.array([0,0,0,0,0,0])"""

        for i in range(self.frame_skip):
            ctrl = self.data.ctrl.copy()

            self.controller.run(
                action, 
                ctrl
            )


            self.do_simulation(ctrl, n_frames=1)
        # Update the visualization
        self.viewer.sync()
        # reward, terminated, truncated, info
        return 0.0, not self.viewer.is_running(), False, {}

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()
