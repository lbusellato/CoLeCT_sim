import numpy as np
import time

from colect_sim.env.ur5_env import BaseRobot
from colect_sim.input_devices.keyboard_input import KeyboardInput

keyboard_input = KeyboardInput()

env = BaseRobot()

started = keyboard_input.wait_for_start(env)

if started:
    traj = np.load('./trained_models/sinusoidal_traj.npy')
    i = 0
    terminated = False
    start_time = time.time()
    while not terminated:
        op_target_reached, terminated = env.step(np.array([0.25, 0.25, 0.1 + traj[i],0,0,0,1]))
        while not op_target_reached:
            op_target_reached, terminated = env.step(np.array([0.25, 0.25, 0.1 + traj[i],0,0,0,1]))
        i += 1
        if i > len(traj) - 1 : i = len(traj) - 1
        time.sleep(0.001)
    env.close()
