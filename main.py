from colect_sim.env.ur5_env import BaseRobot
from colect_sim.input_devices.keyboard_input import KeyboardInput

keyboard_input = KeyboardInput()

env = BaseRobot()

terminated = False
while not terminated:
    action = keyboard_input.get_action()
    reward, terminated, truncated, info = env.step(action)

env.close()
