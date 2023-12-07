import numpy as np
import time

from colect_sim.env.ur5_env import BaseRobot
from colect_sim.input_devices.keyboard_input import KeyboardInput

keyboard_input = KeyboardInput()

env = BaseRobot(plot=True)

try:
    started = keyboard_input.wait_for_start(env)

    if started:
        traj = np.load('./trained_models/sinusoidal_traj.npy')
        i = 0
        terminated = False
        freq = 20 #Hz
        while not terminated:
            start_time = time.time()
            op_target_reached, terminated = env.step(np.array([0.25, 0.25, 0.1 + traj[i],0,0,0,1]))
            while not op_target_reached:
                op_target_reached, terminated = env.step(np.array([0.25, 0.25, 0.1 + traj[i],0,0,0,1]))
            time_passed = time.time() - start_time
            if time_passed < 1 / freq: time.sleep(1 / freq - time_passed)
            i += 1
            if i > len(traj) - 1 : i = len(traj) - 1
except KeyboardInterrupt:
    print(' Stopped')
finally:
    env.close()
