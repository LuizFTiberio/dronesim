import time
import xml.etree.ElementTree as etxml
import os
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from dronesim.envs.BaseAviary import DroneModel, Physics
from dronesim.envs.CtrlAviary import CtrlAviary
from dronesim.control.INDIControl import INDIControl
from dronesim.utils.Logger import Logger
from dronesim.utils.utils import sync, str2bool


def simulate_falcon(ctrl_gains):
    duration_sec = 4
    simulation_frequency = 240
    ctrl_frequency = 96
    drone_model =['Falcon']
    num_drones = 1
    # Initialize the simulation
    AGGR_PHY_STEPS = int(simulation_frequency / ctrl_frequency)
    # To forward X #
    INIT_RPYS = np.array([[0, np.radians(270), 0]])
    INIT_VELS = np.array([[0, 0., 6]])
    INIT_XYZS = np.array([[0., 0., 20.]])

    PERIOD = 15
    NUM_WP = ctrl_frequency * PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3)) + np.array([0, 0, 20])
    for i in range(1, NUM_WP):
        TARGET_POS[i, :] = TARGET_POS[i - 1, :] + (25 / ctrl_frequency, 0, 0)
    wp_counters = np.array([int((i * NUM_WP / 6) % NUM_WP) for i in range(num_drones)])

    env = CtrlAviary(drone_model=drone_model,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_vels=INIT_VELS,
                     initial_rpys=INIT_RPYS,
                     physics= Physics.PYB,
                     neighbourhood_radius=10,
                     freq=240,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=False,
                     record=False,
                     obstacles=False,
                     user_debug_gui=False,
                     ctrl_gains=ctrl_gains
                     )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    states_logger = []
    ctrl = [INDIControl(drone_model=drone) for drone in drone_model]

    #Run the simulation
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / ctrl_frequency))
    action = {str(i): np.array([1., 1., 1., 1.]) for i in range(num_drones)}
    START = time.time()
    for i in range(0, int(duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):

        #Step the simulation
        obs, reward, done, info = env.step(action)

        # Compute control at the desired frequency
        if i % CTRL_EVERY_N_STEPS == 0:

            # Compute control for the current way point
            for j in range(num_drones):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                    state=obs[str(j)]["state"],
                    target_pos=np.array([250, 0, 50]),
                    target_vel=np.array([15, 0, 0]))

            # Go to the next way point and loop
            for j in range(num_drones):
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0

        # Camera View follows the vehicle
        if i % (CTRL_EVERY_N_STEPS * 1) == 0:
            x, y, z, = obs[str(0)]["state"][0:3]
            p.resetDebugVisualizerCamera(cameraDistance=6,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[x, y, z],
                                         physicsClientId=PYB_CLIENT
                                         )

        state = obs[str(0)]["state"]
        states_logger.append(state)

        # Printout
        if i % env.SIM_FREQ == 0:
            env.render()
        # Sync the simulation
        sync(i, START, env.TIMESTEP)

    # Close the environment
    env.close()

    states_logger = np.array(states_logger)
    states_x = states_logger[:, 0]
    states_y = states_logger[:, 1]
    states_z = states_logger[:, 2]

    #plt.plot(states_x)
    #plt.show()
    #plt.plot(states_y)
    #plt.show()
    #plt.plot(states_z)
    #plt.show()

    return states_x[-1],states_z[-1],states_y[-1]

if __name__ == "__main__":
    states_x,states_z,error_y = simulate_falcon()
    print(states_x[-1],states_z[-1],error_y)