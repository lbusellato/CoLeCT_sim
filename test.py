import matplotlib.pyplot as plt
import numpy as np

traj = np.load('./recorded_trajectories/reference_020124_1852.npy')
actual = np.load('./recorded_trajectories/recorded_020124_1852.npy')#[:,:-1]
quat = np.array([0,1,0,1])
quat = quat / np.linalg.norm(quat)
traj_start = np.array([0.375, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])
traj_stop = np.array([0.625, 0.45, 0.205,quat[0],quat[1],quat[2],quat[3]])
fig, ax = plt.subplots(4,4,figsize=(12,6))
labels = ['Position x [m]',
            'Position y [m]',
            'Position z [m]',
            'Quaternion w',
            'Quaternion x',
            'Quaternion y',
            'Quaternion z',
            'Force x [N]',
            'Force y [N]',
            'Force z [N]',
            'Torque x [Nm]',
            'Torque y [Nm]',
            'Torque z [Nm]']
t1 = np.linspace(0,traj.shape[0],traj.shape[0])
t2 = np.linspace(0,traj.shape[0],actual.shape[0])
plot_index = 0
for j in range(4):
    for i in range(4):
        if i != 3 or j == 1:
            if plot_index < traj.shape[1]: # generated trajectories don't have wrench info
                ax[j,i].plot(t1,traj[:,plot_index])
            ax[j,i].plot(t2,actual[:,plot_index])
            ax[j,i].grid()
            if j == 1:
                ax[j,i].set_xlabel('Time [s]')
            ax[j,i].set_ylabel(labels[plot_index])
            plot_index += 1
        else:
            ax[j,i].axis('off')
plt.tight_layout()
plt.show()