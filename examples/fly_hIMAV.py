"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories
in the X-Y plane, around point (0, -.3).

"""
import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from dronesim.envs.BaseAviary import DroneModel, Physics
from dronesim.envs.CtrlAviary import CtrlAviary
from dronesim.control.INDIControl import INDIControl
from dronesim.utils.Logger import Logger
from dronesim.utils.utils import sync, str2bool
from dronesim.utils.wind_simulation import WindSimulation

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=['Falcon_opt'],  type=list,    help='Drone model (default: CF2X)', metavar='', choices=[DroneModel])
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=False,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=False,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=96,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=110.,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = 0.50
    H_STEP = .05
    R = .6
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    INIT_XYZS = np.array([[0., 0, 40]])

    ## To forward X ###
    INIT_RPYS = np.array([[0, 0, 0]])
    INIT_VELS = np.array([[16, 0., 0]])
    target_vel = np.array([0, 0, 0])



    #### Initialize a circular trajectory ######################
    PERIOD = 15
    NUM_WP = ARGS.control_freq_hz*PERIOD
    trajectory_setpoints = np.array([

                                     [500,0,50],
                                    ])
    ARRIVED_AT_WAYPOINT = 10

    # Two options of trajectory
    TARGET_POS = np.zeros((NUM_WP,3))+ np.array([0,0,20])
    for i in range(1,NUM_WP):
        TARGET_POS[i, :] = TARGET_POS[i-1, :] + (25/ARGS.control_freq_hz,0,0) # to ensure 15m in 1 sec
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])

    #### Create the environment with or without video capture ##
    if ARGS.vision:
        env = VisionAviary(drone_model=ARGS.drone,
                           num_drones=ARGS.num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=ARGS.physics,
                           neighbourhood_radius=10,
                           freq=ARGS.simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=ARGS.gui,
                           record=ARGS.record_video,
                           obstacles=ARGS.obstacles
                           )
    else:
        env = CtrlAviary(drone_model=ARGS.drone,
                         num_drones=ARGS.num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_vels=INIT_VELS,
                         initial_rpys=INIT_RPYS,
                         physics=ARGS.physics,
                         neighbourhood_radius=10,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=ARGS.gui,
                         record=ARGS.record_video,
                         obstacles=ARGS.obstacles,
                         user_debug_gui=ARGS.user_debug_gui,
                         geometry_coeffs={},
                         aero_coeffs ={}
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    states_logger = []
    controls_logger = []
    ctrl = [INDIControl(drone_model=drone,control_gains={}) for drone in ARGS.drone]
    wind = WindSimulation(1 / ARGS.simulation_freq_hz)
    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([.3,.3,.3,.3]) for i in range(ARGS.num_drones)}

    START = time.time()

    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
        current_wind = wind.update()
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action,current_wind)

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            x, y, z, = obs[str(0)]["state"][0:3]
            if z < 10 or z>65:
                break
            target_pos = trajectory_setpoints[0]
            diff = target_pos - np.array([x,y,z])
            diff_norm = np.linalg.norm(diff)
            if diff_norm < ARRIVED_AT_WAYPOINT:
                try:
                    trajectory_setpoints = np.delete(trajectory_setpoints,0,0)
                    target_pos = trajectory_setpoints[0]
                    print("*******SETPOINT VISITED. UPDATING TO :", trajectory_setpoints[0] )
                except:
                    print("*******LAST SETPOINT VISITED ********")
                    break

            #### Compute control for the current way point #############
            for j in range(ARGS.num_drones):
                state = obs[str(j)]["state"]
                states_logger.append(state)
                action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(j)]["state"],
                                                                       target_pos= target_pos,##TARGET_POS[wp_counters[j]],
                                                                       target_vel=target_vel,
                                                                       current_wind = current_wind.reshape((6)),
                                                                       nav_type = 'GVF')
                # Over-write the action
                #action[str(j)] = np.array([.5,1.,.5,1.])
                controls_logger.append(action[str(j)])


        #### Camera View follows the vehicle #######################
        if i%(CTRL_EVERY_N_STEPS*1) == 0:
            x,y,z, = obs[str(0)]["state"][0:3]
            p.resetDebugVisualizerCamera(cameraDistance=6,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[x, y, z],
                                         physicsClientId=PYB_CLIENT
                                         )

        #### Log the simulation ####################################
        for j in range(ARGS.num_drones):
            logger.log(drone=j,
                       timestamp=i/env.SIM_FREQ,
                       state= obs[str(j)]["state"],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)]))

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()
            #### Print matrices with the images captured by each drone #
            if ARGS.vision:
                for j in range(ARGS.num_drones):
                    print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
                          obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
                          obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
                          )

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

        ### Break the simulation if we are close to ground
        # if z < 0.3:
        #     break
    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    #logger.save()

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()

    angle = np.arange(0, 2 * np.pi, 0.01)
    nominal_traj_x = 250 * np.cos(angle)
    nominal_traj_y = 250 * np.sin(angle)

    states_logger = np.array(states_logger)
    np.save('falcon_baseline_Wind',states_logger)

    controls_logger = np.array(controls_logger)
    np.save('Controls_falcon_baseline_Wind', controls_logger)


    logging_freq_hz = int(ARGS.simulation_freq_hz / AGGR_PHY_STEPS)
    states_x = states_logger[:, 0]
    states_y = states_logger[:, 1]
    states_z = states_logger[:, 2]
    t = np.arange(0, len(states_z) / logging_freq_hz, 1 / logging_freq_hz)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(states_y, states_x, label='real traj')
    axs[0].plot(nominal_traj_y, nominal_traj_x, '.',color = 'blue', label = 'desired traj')
    axs[0].set_xlabel('Y [m]')
    axs[0].set_ylabel('X [m]')
    axs[0].legend(loc='upper right')
    ##axs[1].plot(t, states_z)
    #axs[1].set_xlabel('time [s]')
    #axs[1].set_ylabel('Z [m]')
    #axs[0].set_aspect(1)
    ## axs[1].set_aspect(1)
    plt.show()
    #print(t[-1])
