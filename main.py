import matplotlib.pyplot as plt
import numpy as np
import time

from colect_sim.env.ur5_env import UR5Env
from colect_sim.input_devices.keyboard_input import KeyboardInput
from colect_sim.utils.traj_generation import linear_traj_w_gauss_noise
from datetime import datetime

keyboard_input = KeyboardInput()

env = UR5Env()#plot=True)

try:
    timestamp = time.time()
    timestamp = datetime.fromtimestamp(timestamp).strftime('%d%m%y_%H%M')
    started = keyboard_input.wait_for_start(env)
    quat = np.array([0,1,0,1])
    quat = quat / np.linalg.norm(quat)
    # Linear scanning near one edge
    traj_start = np.array([0.375, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])
    traj_stop = np.array([0.625, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])
    traj = linear_traj_w_gauss_noise(traj_start, traj_stop, 100, 0., 0.0005)
    np.save('./recorded_trajectories/reference_' + timestamp + '.npy', traj)
    #traj = np.load('./trained_models/sinusoidal_traj.npy')
    if started:
        i = 0
        terminated = False
        while not terminated:
            next = traj[i]
            op_target_reached = False
            while not op_target_reached:
                op_target_reached, terminated = env.step(next)
            env.enable_recording = True # inelegant, but works for aligning the recording to the target
            i += 1
            if i > len(traj) - 1 : terminated = True

    # Plot the executed trajectory against the target
    poses = np.array(env.recorded_data)
    np.save('./recorded_trajectories/recorded_' + timestamp + '.npy', poses)
    fig, ax = plt.subplots(4,4,figsize=(12,6))
    labels = ['Position x [m]',
                'Position y [m]',
                'Position z [m]',
                'Quaternion w [m]',
                'Quaternion x [m]',
                'Quaternion y [m]',
                'Quaternion z [m]',
            'Force x [N]',
            'Force y [N]',
            'Force z [N]',
            'Torque x [Nm]',
            'Torque y [Nm]',
            'Torque z [Nm]']
    t1 = np.linspace(0,traj.shape[0],traj.shape[0])
    t2 = np.linspace(0,traj.shape[0],poses.shape[0])
    plot_index = 0
    for j in range(4):
        for i in range(4):
            if i != 3 or j == 1:
                if plot_index < traj.shape[1]: # generated trajectories don't have wrench info
                    ax[j,i].plot(t1,traj[:,plot_index])
                ax[j,i].plot(t2,poses[:,plot_index])
                ax[j,i].grid()
                if j == 1:
                    ax[j,i].set_xlabel('Time [s]')
                ax[j,i].set_ylabel(labels[plot_index])
                plot_index += 1
            else:
                ax[j,i].axis('off')
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    print(' Stopped')
finally:
    env.close()
