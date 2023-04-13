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
import xml.etree.ElementTree as ET

from dronesim.envs.BaseAviary import DroneModel, Physics
from dronesim.envs.CtrlAviary import CtrlAviary
from dronesim.control.INDIControl import INDIControl
from dronesim.utils.Logger import Logger
from dronesim.utils.utils import sync, str2bool
from dronesim.utils.wind_simulation import WindSimulation

if __name__ == "__main__":
    ## changeable parameters ##
    ctrl_gains = {
        'G': np.array(
            [[150., -150., -150., 150], [650., -650., 650, -650.], [1350., 1350., -1350., -1350.],
             [900, 900, 900, 900]]),
        'kp': 0.8,
        'kd': 0.65,
        'att_p': 20,
        'att_q': 30,
        'att_r': 25,
        'rate_p': 8,
        'rate_q': 8,
        'rate_r': 8,
    }

    aero_coeffs = {
        'CL0': 0.48115,
        'CL_alpha': 4.28,
        'CL_q': 6.689983,
        'CL_del_e': 0,
        'CD0': 0.06130,
        'CD_q': 0.0,
        'CD_del_e': 0.0,
        'CY0': 0.0,
        'CY_beta': -0.049996,
        'CY_p': 0.088261,
        'CY_r': -0.009328,
        'CY_del_r': 0.0,
        'CY_del_a': 0.0,
        'Cl_beta': -0.047674,
        'Cl_p': -0.443968,
        'Cl_r': 0.119986,
        'Cl_del_r': 0.0,
        'Cl_del_a': 0.0,
        'Cm0': -0.11072,
        'Cm_alpha': -1.042060,
        'Cm_q': -2.406242,
        'Cm_del_e': 0,
        'Cn_beta': 0.005068,
        'Cn_p': -0.034716,
        'Cn_r': -0.004010,
        'Cn_del_r': 0.0,
        'Cn_del_a': 0.0,
        'oswald': 0.9,
    }

    geometry_coeffs = {
        'mass':0.728,
        'AR': 6.5,
        'Bref': 0.7,
        'Sref': 0.075,
        'Cref': 0.105,
        'Ixx': 0.0061,
        'Iyy': 0.0010,
        'Izz': 0.0058,
        'CG': 0.032,
        'prop_angle': 0.35,
        'prop_x': 0.02,
        'prop_y': 0.0906,
        'prop_z': 0.075,
    }


    # let's change the parameters directly into the urdf file #
    tree = ET.parse('../dronesim/assets/Falcon_opt.urdf')
    root = tree.getroot()

    # general data
    mass_tree = root.find("link/inertial/mass")
    mass_tree.attrib['value'] = str(geometry_coeffs['mass'])
    CG_tree = root.find("link/inertial/origin")
    CG_tree.attrib['xyz'] = str(geometry_coeffs['CG'])+' '+str(0)+' '+str(0)

    # link specific data
    CG_tree = root.findall("link/inertial/origin")
    CoM = CG_tree[1]
    CoM.attrib['xyz'] = str(geometry_coeffs['CG']) + ' ' + str(0) + ' ' + str(0)

    UR = CG_tree[2]
    UR.attrib['rpy']= str(0) + ' ' + str(geometry_coeffs['prop_angle']) + ' ' + str(0)
    UR.attrib['xyz'] = str(geometry_coeffs['prop_x']) + ' ' + str(-geometry_coeffs['prop_y']) + ' ' + str(geometry_coeffs['prop_z'])

    LR = CG_tree[3]
    LR.attrib['rpy'] = str(0) + ' ' + str(-geometry_coeffs['prop_angle']) + ' ' + str(0)
    LR.attrib['xyz'] = str(geometry_coeffs['prop_x']) + ' ' + str(-geometry_coeffs['prop_y']) + ' ' + str(-
        geometry_coeffs['prop_z'])

    UL = CG_tree[4]
    UL.attrib['rpy'] = str(0) + ' ' + str(geometry_coeffs['prop_angle']) + ' ' + str(0)
    UL.attrib['xyz'] = str(geometry_coeffs['prop_x']) + ' ' + str(geometry_coeffs['prop_y']) + ' ' + str(
        geometry_coeffs['prop_z'])

    LL = CG_tree[5]
    LL.attrib['rpy'] = str(0) + ' ' + str(-geometry_coeffs['prop_angle']) + ' ' + str(0)
    LL.attrib['xyz'] = str(geometry_coeffs['prop_x']) + ' ' + str(geometry_coeffs['prop_y']) + ' ' + str(-
        geometry_coeffs['prop_z'])

    # control data
    control_gains = root.find("control")
    G = ctrl_gains['G']
    control_gains[1].attrib['roll'] = str(G[0][0])+' '+str(G[0][1])+' '+str(G[0][2])+' '+str(G[0][3])
    control_gains[2].attrib['pitch'] = str(G[1][0]) + ' ' + str(G[1][1]) + ' ' + str(G[1][2]) + ' ' + str(G[1][3])
    control_gains[3].attrib['yaw'] = str(G[2][0]) + ' ' + str(G[2][1]) + ' ' + str(G[2][2]) + ' ' + str(G[2][3])
    control_gains[4].attrib['thrust'] = str(G[3][0]) + ' ' + str(G[3][1]) + ' ' + str(G[3][2]) + ' ' + str(G[3][3])

    guidance_gains = root.find("control/indi_guidance_gains/pos")
    guidance_gains.attrib['kp'] = str(ctrl_gains['kp'])
    guidance_gains.attrib['kd'] = str(ctrl_gains['kd'])

    att_att_gains = root.find("control/indi_att_gains/att")
    att_att_gains.attrib['p'] = str(ctrl_gains['att_p'])
    att_att_gains.attrib['q'] = str(ctrl_gains['att_q'])
    att_att_gains.attrib['r'] = str(ctrl_gains['att_r'])

    att_rate_gains = root.find("control/indi_att_gains/rate")
    att_rate_gains.attrib['p'] = str(ctrl_gains['rate_p'])
    att_rate_gains.attrib['q'] = str(ctrl_gains['rate_q'])
    att_rate_gains.attrib['r'] = str(ctrl_gains['rate_r'])

    inertia = root.find("link/inertial/inertia")
    inertia.attrib['ixx'] = str(geometry_coeffs['Ixx'])
    inertia.attrib['iyy'] = str(geometry_coeffs['Iyy'])
    inertia.attrib['izz'] = str(geometry_coeffs['Izz'])

    # aero data
    ref = root.find("aero_coeffs/ref")
    ref.attrib['AR'] =   str(geometry_coeffs['AR'])
    ref.attrib['Bref'] = str(geometry_coeffs['Bref'])
    ref.attrib['Sref'] = str(geometry_coeffs['Sref'])
    ref.attrib['Cref'] = str(geometry_coeffs['Cref'])
    ref.attrib['oswald'] = str(aero_coeffs['oswald'])

    CL = root.find("aero_coeffs/CL")
    CL.attrib['CL0'] = str(aero_coeffs['CL0'])
    CL.attrib['CL_alpha'] = str(aero_coeffs['CL_alpha'])
    CL.attrib['CL_q'] = str(aero_coeffs['CL_q'])
    CL.attrib['CL_del_e'] = str(aero_coeffs['CL_del_e'])

    cd = root.find("aero_coeffs/CD")
    cd.attrib['CD0'] = str(aero_coeffs['CD0'])
    cd.attrib['CD_q'] = str(aero_coeffs['CD_q'])
    cd.attrib['CD_del_e'] = str(aero_coeffs['CD_del_e'])

    cy = root.find("aero_coeffs/CY")
    cy.attrib['CY0'] = str(aero_coeffs['CY0'])
    cy.attrib['CY_beta'] = str(aero_coeffs['CY_beta'])
    cy.attrib['CY_p'] = str(aero_coeffs['CY_p'])
    cy.attrib['CY_r'] = str(aero_coeffs['CY_r'])
    cy.attrib['CY_del_r'] = str(aero_coeffs['CY_del_r'])
    cy.attrib['CY_del_a'] = str(aero_coeffs['CY_del_a'])

    cl = root.find("aero_coeffs/Cl")
    cl.attrib['Cl_beta'] = str(aero_coeffs['Cl_beta'])
    cl.attrib['Cl_p'] = str(aero_coeffs['Cl_p'])
    cl.attrib['Cl_r'] = str(aero_coeffs['Cl_r'])
    cl.attrib['Cl_del_r'] = str(aero_coeffs['Cl_del_r'])
    cl.attrib['Cl_del_a'] = str(aero_coeffs['Cl_del_a'])

    cm = root.find("aero_coeffs/Cm")
    cm.attrib['Cm0'] = str(aero_coeffs['Cm0'])
    cm.attrib['Cm_alpha'] = str(aero_coeffs['Cm_alpha'])
    cm.attrib['Cm_q'] = str(aero_coeffs['Cm_q'])
    cm.attrib['Cm_del_e'] = str(aero_coeffs['Cm_del_e'])

    cn = root.find("aero_coeffs/Cn")
    cn.attrib['Cn_beta'] = str(aero_coeffs['Cn_beta'])
    cn.attrib['Cn_p'] = str(aero_coeffs['Cn_p'])
    cn.attrib['Cn_r'] = str(aero_coeffs['Cn_r'])
    cn.attrib['Cn_del_r'] = str(aero_coeffs['Cn_del_r'])
    cn.attrib['Cn_del_a'] = str(aero_coeffs['Cn_del_a'])

    tree.write('../dronesim/assets/Falcon_opt.urdf')

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone', default=['Falcon_opt'], type=list, help='Drone model (default: CF2X)', metavar='',
                        choices=[DroneModel])
    parser.add_argument('--num_drones', default=1, type=int, help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics', default="pyb", type=Physics, help='Physics updates (default: PYB)', metavar='',
                        choices=Physics)
    parser.add_argument('--vision', default=False, type=str2bool, help='Whether to use VisionAviary (default: False)',
                        metavar='')
    parser.add_argument('--gui', default=True, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--record_video', default=False, type=str2bool, help='Whether to record a video (default: False)',
                        metavar='')
    parser.add_argument('--plot', default=True, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=False, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate', default=True, type=str2bool,
                        help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles', default=False, type=str2bool,
                        help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240, type=int, help='Simulation frequency in Hz (default: 240)',
                        metavar='')
    parser.add_argument('--control_freq_hz', default=96, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=115, type=int, help='Duration of the simulation in seconds (default: 5)',
                        metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = 0.50
    H_STEP = .05
    R = .6
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz / ARGS.control_freq_hz) if ARGS.aggregate else 1

    INIT_XYZS = np.array([[-100., -100., 40]])

    ## To forward X ###
    INIT_RPYS = np.array([[0, 0, 0]])
    INIT_VELS = np.array([[18, 0., 0]])
    target_vel = np.array([0, 0, 0])

    # INIT_RPYS = np.array([[0, 0, np.radians(180)]])
    # INIT_VELS = np.array([[-18, 0., 0]])
    # target_vel=np.array([0,0,0])

    #### Initialize a circular trajectory ######################
    PERIOD = 15
    NUM_WP = ARGS.control_freq_hz * PERIOD
    trajectory_setpoints = np.array([
        # [-50, 30, 40],
        # [500, 10, 40],
        # [1500, 10, 40],
        [0, -100, 40],
        [0, 300, 40],
        [0, -100, 40],
        # []
    ])
    ARRIVED_AT_WAYPOINT = 10

    # Two options of trajectory
    TARGET_POS = np.zeros((NUM_WP, 3)) + np.array([0, 0, 20])
    for i in range(1, NUM_WP):
        TARGET_POS[i, :] = TARGET_POS[i - 1, :] + (25 / ARGS.control_freq_hz, 0, 0)  # to ensure 15m in 1 sec
    wp_counters = np.array([int((i * NUM_WP / 6) % NUM_WP) for i in range(ARGS.num_drones)])

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
                         aero_coeffs={}
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz / AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    states_logger = []
    ctrl = [INDIControl(drone_model=drone, control_gains={}) for drone in ARGS.drone]
    wind = WindSimulation(1 / ARGS.simulation_freq_hz)
    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / ARGS.control_freq_hz))
    action = {str(i): np.array([.95, .95, .95, .95]) for i in range(ARGS.num_drones)}

    START = time.time()

    for i in range(0, int(ARGS.duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):
        current_wind = wind.update()
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action, current_wind)

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:
            x, y, z, = obs[str(0)]["state"][0:3]
        target_pos = trajectory_setpoints[0]
        diff = target_pos - np.array([x, y, z])
        diff_norm = np.linalg.norm(diff)
        if diff_norm < ARRIVED_AT_WAYPOINT:
            try:
                trajectory_setpoints = np.delete(trajectory_setpoints, 0, 0)
                target_pos = trajectory_setpoints[0]
                print("*******SETPOINT VISITED. UPDATING TO :", trajectory_setpoints[0])
            except:
                print("*******LAST SETPOINT VISITED ********")
                break
                # print(x,y,target_pos)

                #### Compute control for the current way point #############
        for j in range(ARGS.num_drones):
            action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                                                   state=obs[str(j)]["state"],
                                                                   target_pos=target_pos,  ##TARGET_POS[wp_counters[j]],
                                                                   target_vel=target_vel,
                                                                   current_wind=current_wind)
            # Over-write the action
            # action[str(j)] = np.array([.95,.95,.95,.95])

            #### Camera View follows the vehicle #######################
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

        #### Log the simulation ####################################
        for j in range(ARGS.num_drones):
            logger.log(drone=j,
                       timestamp=i / env.SIM_FREQ,
                       state=obs[str(j)]["state"],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)]))

        #### Printout ##############################################
        if i % env.SIM_FREQ == 0:
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
    # logger.save()

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()

    states_logger = np.array(states_logger)
    states_x = states_logger[:, 0]
    states_y = states_logger[:, 1]
    states_z = states_logger[:, 2]