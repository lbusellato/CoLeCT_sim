import numpy as np
import faulthandler
from colect_sim.env.ur5_env import BaseRobot
from colect_sim.input_devices.keyboard_input import KeyboardInput

faulthandler.enable()

keyboard_input = KeyboardInput()

env = BaseRobot()

res = keyboard_input.wait_for_start()

traj = np.load('./trained_models/sinusoidal_traj.npy')
i = 0
terminated = False
while not terminated:
    if i >= len(traj): 
        reward, terminated, truncated, info = env.step(np.array([0.25, 0.25, 0.1,0,0,0,-1]))
    else:
        reward, terminated, truncated, info = env.step(np.array([0.25, 0.25, 0.1 + traj[i],0,0,0,-1]))
        i += 1
env.close()
