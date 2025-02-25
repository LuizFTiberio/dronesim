import os
from sys import platform
import time
import collections
from datetime import datetime
from enum import Enum
import xml.etree.ElementTree as etxml
from PIL import Image
import numpy as np
import pybullet as pyb
import pybullet_data
import gym
import pickle
import sys


# For advanced propeller model based on database
from dronesim.utils.utils import calculate_propeller_forces_moments
from dronesim.database.propeller_database import *
from dronesim.utils.wind_simulation import WindSimulation

filename = "/home/luiztiberio/Documents/dronesim/dronesim/utils/kpls_thrust.pkl"
with open(filename, "rb") as thrust:
   thrust_model = pickle.load(thrust)

torque_model = None
filename = "/home/luiztiberio/Documents/dronesim/dronesim/utils/kplsk_torque.pkl"
with open(filename, "rb") as torque:
   torque_model = pickle.load(torque)

# for fixed-wing vehicle's physics
from dronesim.utils.utils import R_aero_to_body,Quaternion2Rotation

class DroneModel(Enum):
    """Drone models enumeration class."""

    CF2X = "cf2x"   # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"   # Bitcraze Craziflie 2.0 in the + configuration
    HB = "hb"       # Generic quadrotor (with AscTec Hummingbird inertial properties)
    TELLO = "tello" # Tello quadrotor
    TAILSITTER = "darkO"

################################################################################

class Physics(Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"                         # Base PyBullet physics update
    DYN = "dyn"                         # Update with an explicit model of the dynamics
    PYB_GND = "pyb_gnd"                 # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"               # PyBullet physics update with drag
    PYB_DW = "pyb_dw"                   # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw" # PyBullet physics update with ground effect, drag, and downwash

################################################################################

class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0     # Red, green, blue (and alpha)
    DEP = 1     # Depth
    SEG = 2     # Segmentation by object id
    BW = 3      # Black and white

################################################################################

from dataclasses import dataclass

@dataclass
class Drone:
    TYPE: str
    M: float # int - str - tuple
    L: float
    THRUST2WEIGHT_RATIO: float
    J: float
    J_INV: float
    KF: float
    KM: float
    COLLISION_H: float
    COLLISION_R: float
    COLLISION_Z_OFFSET: float
    MAX_SPEED_KMH: float
    GND_EFF_COEFF: float
    PROP_RADIUS: float
    DRAG_COEFF: float
    DW_COEFF_1: float
    DW_COEFF_2: float
    DW_COEFF_3: float
    PWM2RPM_SCALE: float #list[floats] #FIXME ?? These are list of floats right now...
    PWM2RPM_CONST: float #list[floats] #FIXME ??
    INDI_ACTUATOR_NR: int
    INDI_OUTPUT_NR: int
    G1: float #list[floats] #FIXME
    MIN_PWM: float
    MAX_PWM: float

  # def __post_init__(self):
  #   self.M = float(self.M) # int - str - tuple
  #   self.L = float(self.L)
  #   self.THRUST2WEIGHT_RATIO = float(self.THRUST2WEIGHT_RATIO)
  #   self.J = float(self.J)
  #   self.J_INV = float(self.J_INV)
  #   self.KF = float(self.KF)
  #   self.KM = float(self.KM)
  #   self.COLLISION_H = float(self.COLLISION_H)
  #   self.COLLISION_R = float(self.)
  #   self.COLLISION_Z_OFFSET = float(self.)
  #   self.MAX_SPEED_KMH = float(self.)
  #   self.GND_EFF_COEFF = float(self.)
  #   self.PROP_RADIUS = float(self.)
  #   self.DRAG_COEFF = float(self.)
  #   self.DW_COEFF_1 = float(self.)
  #   self.DW_COEFF_2 = float(self.)
  #   self.DW_COEFF_3 = float(self.)
  #   # self.name = str(self.name, 'utf-8')
  #   # self.linkName = str(self.linkName, 'utf-8')

################################################################################

class BaseAviary(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    metadata = {'render.modes': ['human']}
    
    ################################################################################

    def __init__(self,
                 drone_model: list=['tello'], #DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_vels=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=True,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 dynamics_attributes=False,
                 geometry_coeffs={},
                 aero_coeffs={}
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.
        dynamics_attributes : bool, optional
            Whether to allocate the attributes needed by subclasses accepting thrust and torques inputs.

        """
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.SIM_FREQ = freq
        self.TIMESTEP = 1./self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        self.GEOMETRY_COEFFS = geometry_coeffs
        self.AERO_COEFFS = aero_coeffs
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = [drone + ".urdf" for drone in drone_model]
        self.drones = [Drone(*self._parseURDFParameters(drone)) for drone in self.URDF]

        for drone, urdf in zip(self.drones, self.URDF):
            if 'fixed_wing' in drone.TYPE:
                self._parseURDFFixedwingParameters(drone, urdf)
            if 'winged_vtol_physics' in drone.TYPE:
                self._parseURDFVTOLParameters(drone, urdf)
            if 'winged_physics' in drone.TYPE:
                self._parseURDFVTOLParameters(drone, urdf)

        #### Compute constants #####################################
#        self.GRAVITY = self.G*self.M
        # self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
#        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        # self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        # if self.DRONE_MODEL == DroneModel.CF2X:
        #     self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        # elif self.DRONE_MODEL in [DroneModel.CF2P, DroneModel.HB]:
        #     self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        # self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        # self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create attributes for vision tasks ####################
        self.VISION_ATTR = vision_attributes
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.SIM_FREQ/self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ%self.AGGR_PHY_STEPS != 0:
                print("[ERROR] in BaseAviary.__init__(), aggregate_phy_steps incompatible with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC))
                exit()
            if self.RECORD:
                self.ONBOARD_IMG_PATH = os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/onboard-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+"/"
                os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        #### Create attributes for dynamics control inputs #########
        self.DYNAMICS_ATTR = dynamics_attributes
        if self.DYNAMICS_ATTR:
            if self.DRONE_MODEL == DroneModel.CF2X:
                self.A = np.array([ [1, 1, 1, 1], [1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2), -1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2)], [-1, 1, -1, 1] ])
            elif self.DRONE_MODEL in [DroneModel.CF2P, DroneModel.HB]:
                self.A = np.array([ [1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1] ])
            self.INV_A = np.linalg.inv(self.A)
            self.B_COEFF = np.array([1/self.KF, 1/(self.KF*self.L), 1/(self.KF*self.L), 1/self.KM])
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = pyb.connect(pyb.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [pyb.COV_ENABLE_RGB_BUFFER_PREVIEW, pyb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, pyb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                pyb.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            pyb.resetDebugVisualizerCamera(cameraDistance=6,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.CLIENT
                                         )
            ret = pyb.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1*np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = pyb.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = pyb.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = pyb.connect(pyb.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH=int(640)
                self.VID_HEIGHT=int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.SIM_FREQ/self.FRAME_PER_SEC)
                self.CAM_VIEW = pyb.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                    yaw=-30,
                                                                    pitch=-30,
                                                                    roll=0,
                                                                    cameraTargetPosition=[0, 0, 0],
                                                                    upAxisIndex=2,
                                                                    physicsClientId=self.CLIENT
                                                                    )
                self.CAM_PRO = pyb.computeProjectionMatrixFOV(fov=60.0,
                                                            aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0
                                                            )
        #### Set initial poses #####################################
        if initial_xyzs is None:
            self.INIT_XYZS = np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]), \
                                        np.array([y*4*self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES, 3)
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES,3):
            self.INIT_XYZS = initial_xyzs
        else:
            print("[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")
        

        self.INIT_VELS = initial_vels
        #print('INIT',self.INIT_VELS)

        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
    
    ################################################################################

    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.

        """
        pyb.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        return self._computeObs()
    
    ################################################################################

    def step(self,
             action,
             current_wind
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current epoisode is over, check the specific implementation of `_computeDone()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = pyb.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=pyb.ER_TINY_RENDERER,
                                                     flags=pyb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = pyb.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = pyb.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.SIM_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [pyb.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            # self._saveLastAction(action) # FIXME
            # clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
            clipped_action = self._preprocessAction(action)

        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.AGGR_PHY_STEPS):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.AGGR_PHY_STEPS > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[str(i)], i,current_wind)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                pyb.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.AGGR_PHY_STEPS)
        return obs, reward, done, info
    
    ################################################################################
    
    def render(self,
               mode='human',
               close=False
               ):
        """Prints a textual output of the environment.

        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        # self.CLIENT = pyb.connect(pyb.GUI)
        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
            if self.GUI:
                print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
                      "——— wall-clock time {:.1f}s,".format(time.time()-self.RESET_TIME),
                      "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter*self.TIMESTEP, self.SIM_FREQ, (self.step_counter*self.TIMESTEP)/(time.time()-self.RESET_TIME)))
        for i in range (self.NUM_DRONES):
            if self.GUI:
                print("[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                      "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                      "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                      "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0]*self.RAD2DEG, self.rpy[i, 1]*self.RAD2DEG, self.rpy[i, 2]*self.RAD2DEG),
                      "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i, 2]))
    
    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        if self.RECORD and self.GUI:
            pyb.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        pyb.disconnect(physicsClientId=self.CLIENT)
    
    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT
    
    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS
    
    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_action = -1*np.ones((self.NUM_DRONES, 4))

        self.last_clipped_action = {str(i): np.zeros(self.drones[i].INDI_ACTUATOR_NR) for i in range(self.NUM_DRONES)}# Action is a dictionary in order to keep heterogeneous control action of multi-vehicle scenarios. np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        pyb.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        pyb.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        pyb.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = pyb.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        self.DRONE_IDS = np.array([pyb.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF[i],
                                              self.INIT_XYZS[i,:],
                                              pyb.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
                                              flags = pyb.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            #### If the vehicle initialized with flight velocity : e.g. Fixed-wing configuration
            if self.INIT_VELS is not None:
                if self.INIT_VELS[i] is not None: # FIXME Ugly :( use something else ? instance
                    pyb.resetBaseVelocity(self.DRONE_IDS[i], linearVelocity= self.INIT_VELS[i], physicsClientId=self.CLIENT)
            #### Show the frame of reference of the drone, note that ###
            #### It severly slows down the GUI #########################
            if self.GUI and self.USER_DEBUG:
                self._showDroneLocalAxes(i)
            #### Disable collisions between drones' and the ground plane
            #### E.g., to start a drone at [0,0,0] #####################
            # pyb.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles()
    
    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range (self.NUM_DRONES):
            self.pos[i], self.quat[i] = pyb.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rpy[i] = pyb.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = pyb.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
    
    ################################################################################

    def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.

        """
        if self.RECORD and self.GUI:
            self.VIDEO_ID = pyb.startStateLogging(loggingType=pyb.STATE_LOGGING_VIDEO_MP4,
                                                #fileName=os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".mp4",
                                                  fileName="/home/luiztiberio/Videos/video-" + datetime.now().strftime(
                                                      "%m.%d.%Y_%H.%M.%S") + ".mp4",
                                                physicsClientId=self.CLIENT
                                                )
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+"/"
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
    
    ################################################################################

    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[str(nth_drone)]])
        return state # state.reshape(20,) FIXME

    ################################################################################

    def _getDroneImages(self,
                        nth_drone,
                        segmentation: bool=True
                        ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation : bool, optional
            Whehter to compute the compute the segmentation mask.
            It affects performance.

        Returns
        -------
        ndarray 
            (h, w, 4)-shaped array of uint8's containing the RBG(A) image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the depth image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the segmentation image captured from the n-th drone's POV.

        """
        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()
        rot_mat = np.array(pyb.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat,np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])
        DRONE_CAM_VIEW = pyb.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :]+np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[0, 0, 1],
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO =  pyb.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=self.L,
                                                      farVal=1000.0
                                                      )
        SEG_FLAG = pyb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else pyb.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = pyb.getCameraImage(width=self.IMG_RES[0],
                                                 height=self.IMG_RES[1],
                                                 shadow=1,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.CLIENT
                                                 )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    ################################################################################

    def _exportImage(self,
                     img_type: ImageType,
                     img_input,
                     path: str,
                     frame_num: int=0
                     ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        img_type : ImageType
            The image type: RGB(A), depth, segmentation, or B&W (from RGB).
        img_input : ndarray
            (h, w, 4)-shaped array of uint8's for RBG(A) or B&W images.
            (h, w)-shaped array of uint8's for depth or segmentation images.
        path : str
            Path where to save the output as PNG.
        fram_num: int, optional
            Frame number to append to the PNG's filename.

        """
        if img_type == ImageType.RGB:
            (Image.fromarray(img_input.astype('uint8'), 'RGBA')).save(path+"frame_"+str(frame_num)+".png")
        elif img_type == ImageType.DEP:
            temp = ((img_input-np.min(img_input)) * 255 / (np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type == ImageType.SEG:
            temp = ((img_input-np.min(img_input)) * 255 / (np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type == ImageType.BW:
            temp = (np.sum(img_input[:, :, 0:2], axis=2) / 3).astype('uint8')
        else:
            print("[ERROR] in BaseAviary._exportImage(), unknown ImageType")
            exit()
        if img_type != ImageType.RGB:
            (Image.fromarray(temp)).save(path+"frame_"+str(frame_num)+".png")

    ################################################################################

    def _getAdjacencyMatrix(self):
        """Computes the adjacency matrix of a multi-drone system.

        Attribute NEIGHBOURHOOD_RADIUS is used to determine neighboring relationships.

        Returns
        -------
        ndarray
            (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the adjacency matrix 
            of the system: adj_mat[i,j] == 1 if (i, j) are neighbors; == 0 otherwise.

        """
        adjacency_mat = np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES-1):
            for j in range(self.NUM_DRONES-i-1):
                if np.linalg.norm(self.pos[i, :]-self.pos[j+i+1, :]) < self.NEIGHBOURHOOD_RADIUS:
                    adjacency_mat[i, j+i+1] = adjacency_mat[j+i+1, i] = 1
        return adjacency_mat
    
    ################################################################################
    
    def _physics(self,
                 cmd, # was rpm
                 nth_drone,
                 current_wind
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        cmd : ndarray
            (4)-shaped array of ints containing the command values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        """
        pyb.changeDynamics(0, -1, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 0, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 1, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 2, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 3, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 4, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 5, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 6, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 7, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 8, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 9, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(0, 10, linearDamping=0, angularDamping=0)


        pyb.changeDynamics(1, -1, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 0, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 1, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 2, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 3, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 4, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 5, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 6, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 7, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 8, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 9, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(1, 10, linearDamping=0, angularDamping=0)

        pyb.changeDynamics(2, -1, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 0, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 1, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 2, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 3, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 4, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 5, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 6, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 7, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 8, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 9, linearDamping=0, angularDamping=0)
        pyb.changeDynamics(2, 10, linearDamping=0, angularDamping=0)

        if 'quad' in self.drones[nth_drone].TYPE:
            self._quad_copter_physics(cmd,nth_drone)
        elif 'morphing_hexa' in self.drones[nth_drone].TYPE:
            self._morphing_hexa_physics(cmd,nth_drone)
        elif 'fixed_wing' in self.drones[nth_drone].TYPE:
            self._fixed_wing_physics(cmd,nth_drone)
        elif 'tail_sitter' in self.drones[nth_drone].TYPE:
            self._tail_sitter_physics(cmd,nth_drone)
        elif 'coaxial_birotor' in self.drones[nth_drone].TYPE:
            self._coaxial_birotor_physics(cmd,nth_drone)
        elif 'winged_vtol_physics' in self.drones[nth_drone].TYPE:
            self._winged_vtol_physics(cmd,nth_drone,current_wind)
        elif '_winged_physics' in self.drones[nth_drone].TYPE:
            self._winged_physics(cmd,nth_drone,current_wind)
        else:
            rpm = self.drones[nth_drone].PWM2RPM_SCALE * cmd + self.drones[nth_drone].PWM2RPM_CONST
            # rpm = 20000.*cmd
            forces = np.array(rpm**2)*self.drones[nth_drone].KF
            torques = np.array(rpm**2)*self.drones[nth_drone].KM
            z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
            for i in range(4):
                pyb.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0, 0, forces[i]],
                                     posObj=[0, 0, 0],
                                     flags=pyb.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )
            pyb.applyExternalTorque(self.DRONE_IDS[nth_drone],
                                  4, # FIX ME This should be -1 to define the body, here it only works as there is an additional link called center of mass in the URDF file! will not work for tohers !
                                  torqueObj=[0, 0, z_torque],
                                  flags=pyb.LINK_FRAME,
                                  physicsClientId=self.CLIENT
                                  )

    ################################################################################


    def _winged_vtol_physics(self,
                            cmd,
                            nth_drone,
                            current_wind
                            ):
        '''
        VTOL flight dynamics based on  the models by Randy Beard and Tim McLain
        https://github.com/randybeard/uavbook
        We begin by taking states, and then calculate the forces and moments
        '''
        # calculate atm data
        quaternion = self.quat[nth_drone, :]
        cur_rpy = np.array(pyb.getEulerFromQuaternion(quaternion))
        u, v, w = self.vel[nth_drone, :]
        R_vb = np.array(pyb.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        # We now correct R_vb from Pybullet frame to wind frame
        R_vb = R_vb @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        steady_state = current_wind[0:3]
        gust = current_wind[3:6]
        # convert wind vector from world to body frame and add gust
        wind_body_frame = R_vb @ steady_state + gust
        v_air_i = np.array([u, v, w])
        v_air_b = R_vb.T.dot(v_air_i)
        ur = v_air_b[0] - wind_body_frame[0]
        vr = v_air_b[1] - wind_body_frame[1]
        wr = v_air_b[2] - wind_body_frame[2]
        # compute airspeed
        Va = np.sqrt(ur ** 2 + vr ** 2 + wr ** 2)[0]
        # compute angle of attack
        if ur == 0:
            alpha = np.sign(wr) * np.pi / 2
        else:
            alpha = np.arctan(wr / ur)
        # compute sideslip angle
        if Va == 0:
            beta = np.sign(vr) * np.pi / 2
        else:
            beta = np.arcsin(vr / np.sqrt(ur ** 2 + vr ** 2 + wr ** 2))
        alpha = alpha[0]
        beta = beta[0]
        p, q, r = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ self.ang_v[nth_drone, :]
        drone = self.drones[nth_drone]

        cmd_m1 = cmd[0] *1570+730
        cmd_m2 = cmd[1] *1570+730
        cmd_m3 = cmd[2] *1570+730
        cmd_m4 = cmd[3] *1570+730

        # small hack to avoid smt to print everything all the time
        sys.stdout = open(os.devnull, 'w')
        alpha_M = alpha + drone.prop_angle
        T1 = thrust_model.predict_values(np.array([Va, cmd_m1, alpha_M]).reshape((-1,3)))[0][0]
        Q1 = torque_model.predict_values(np.array([Va, cmd_m1, alpha_M]).reshape((-1,3)))[0][0]
        T2 = thrust_model.predict_values(np.array([Va, cmd_m2, alpha_M]).reshape((-1,3)))[0][0]
        Q2 = torque_model.predict_values(np.array([Va, cmd_m2, alpha_M]).reshape((-1,3)))[0][0]
        T3 = thrust_model.predict_values(np.array([Va, cmd_m3, alpha_M]).reshape((-1,3)))[0][0]
        Q3 = torque_model.predict_values(np.array([Va, cmd_m3, alpha_M]).reshape((-1,3)))[0][0]
        T4 = thrust_model.predict_values(np.array([Va, cmd_m4, alpha_M]).reshape((-1,3)))[0][0]
        Q4 = torque_model.predict_values(np.array([Va, cmd_m4, alpha_M]).reshape((-1,3)))[0][0]

        # reverting the small hack
        sys.stdout = sys.__stdout__

        # retrieve the commands
        cmd_elevator = 0
        cmd_aileron = 0
        cmd_rudder = 0

        # calculate forces and moments - aero
        n_sigma = np.exp(-drone.M * (alpha - drone.alpha0))
        p_sigma = np.exp(drone.M * (alpha + drone.alpha0))
        sigma = (1 + p_sigma + n_sigma) / ((1 + n_sigma) * (1 + p_sigma))

        CL_a = (1 - sigma) * (drone.CL0 + drone.CL_alpha * alpha) + \
               sigma * (2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha))
        CD_a = drone.CD0 + (drone.CL0 + drone.CL_alpha * alpha) ** 2 / (np.pi * drone.oswald * drone.AR)

        CL = (-CD_a * np.sin(alpha) - CL_a * np.cos(alpha)) + \
             ((-drone.CD_q * np.sin(alpha) - drone.CL_q * np.cos(alpha)) * drone.Cref * q / (2 * Va)) + \
             ((-drone.CD_del_e * np.sin(alpha) - drone.CL_del_e * np.cos(alpha)) * cmd_elevator)
        CD = (-CD_a * np.cos(alpha) + CL_a * np.sin(alpha)) + \
             ((-drone.CD_q * np.cos(alpha) + drone.CL_q * np.sin(alpha)) * drone.Cref * q / (2 * Va)) + \
             ((-drone.CD_del_e * np.cos(alpha) + drone.CL_del_e * np.sin(alpha)) * cmd_elevator)

        # compute Lift and Drag Forces
        F_lift = .5 * drone.rho * drone.Sref * Va ** 2 * CL
        F_drag = .5 * drone.rho * drone.Sref * Va ** 2 * CD

        Fy = .5 * drone.rho * drone.Sref * Va ** 2 * (
                    drone.CY_beta * beta + drone.CY_p * p * drone.Bref / (2 * Va) +
                            drone.CY_r * r * drone.Bref / (2 * Va) +
                    drone.CY_del_a * cmd_aileron + drone.CY_del_r * cmd_rudder)

        My = .5 * drone.rho * drone.Sref * Va ** 2 * drone.Cref * \
                    (drone.Cm0 + drone.Cm_alpha * alpha +(drone.Cm_q * q *drone.Cref / (2 * Va))
                     + drone.Cm_del_e * cmd_elevator)

        Mx = .5 * drone.rho * drone.Sref * Va ** 2 * drone.Bref * (
                drone.Cl_beta * beta + drone.Cl_p * p * drone.Bref / (2 * Va) +
                drone.Cl_r * r * drone.Bref / (2 * Va) +
                drone.Cl_del_a * cmd_aileron + drone.Cl_del_r * cmd_rudder)


        Mz = 0.5 * drone.rho * drone.Sref * Va ** 2 * drone.Bref * (
                                 drone.Cn_beta * beta + drone.Cn_p * p * drone.Bref / (2 * Va) +
                                 drone.Cn_r * r * drone.Bref / (2 * Va) +
                                 drone.Cn_del_a * cmd_aileron + drone.Cn_del_r * cmd_rudder)

        ### AERO FORCES
        pyb.applyExternalForce(self.DRONE_IDS[nth_drone],
                               1,
                               forceObj=[F_drag, -Fy, -F_lift],
                               posObj=[0, 0, 0],
                               flags=pyb.LINK_FRAME,
                               physicsClientId=self.CLIENT
                               )
        ### AERO MOMENTS
        pyb.applyExternalTorque(self.DRONE_IDS[nth_drone],
                                1,
                                torqueObj=[Mx, -My, -Mz],
                                flags=pyb.LINK_FRAME,
                                physicsClientId=self.CLIENT
                                )
        ### Prop1 FORCES
        pyb.applyExternalForce(self.DRONE_IDS[nth_drone],
                               2,
                               forceObj=[T1, 0, 0],
                               posObj=[0, 0, 0],
                               flags=pyb.LINK_FRAME,
                               physicsClientId=self.CLIENT
                               )
        ### Prop1 MOMENTS
        pyb.applyExternalTorque(self.DRONE_IDS[nth_drone],
                                2,
                                torqueObj=[Q1, 0, 0],
                                flags=pyb.LINK_FRAME,
                                physicsClientId=self.CLIENT
                                )
        ### Prop2 FORCES
        pyb.applyExternalForce(self.DRONE_IDS[nth_drone],
                               3,
                               forceObj=[T2, 0., 0],
                               posObj=[0, 0, 0],
                               flags=pyb.LINK_FRAME,
                               physicsClientId=self.CLIENT
                               )
        ### Prop2 MOMENTS
        pyb.applyExternalTorque(self.DRONE_IDS[nth_drone],
                                3,
                                torqueObj=[-Q2, 0, 0],
                                flags=pyb.LINK_FRAME,
                                physicsClientId=self.CLIENT
                                )
        ### Prop3 FORCES
        pyb.applyExternalForce(self.DRONE_IDS[nth_drone],
                               4,
                               forceObj=[T3, 0., 0],
                               posObj=[0, 0, 0],
                               flags=pyb.LINK_FRAME,
                               physicsClientId=self.CLIENT
                               )
        ### Prop3 MOMENTS
        pyb.applyExternalTorque(self.DRONE_IDS[nth_drone],
                                4,
                                torqueObj=[-Q3, 0, 0],
                                flags=pyb.LINK_FRAME,
                                physicsClientId=self.CLIENT
                                )
        ### Prop4 FORCES
        pyb.applyExternalForce(self.DRONE_IDS[nth_drone],
                               5,
                               forceObj=[T4, 0., 0],
                               posObj=[0, 0, 0],
                               flags=pyb.LINK_FRAME,
                               physicsClientId=self.CLIENT
                               )
        ### Prop4 MOMENTS
        pyb.applyExternalTorque(self.DRONE_IDS[nth_drone],
                                5,
                                torqueObj=[Q4, 0, 0],
                                flags=pyb.LINK_FRAME,
                                physicsClientId=self.CLIENT
                                )

    def _winged_physics(self,
                                 cmd,
                                 nth_drone,
                                 current_wind
                                 ):
        '''
        VTOL flight dynamics based on  the models by Randy Beard and Tim McLain
        https://github.com/randybeard/uavbook
        We begin by taking states, and then calculate the forces and moments
        '''
        # calculate atm data
        quaternion = self.quat[nth_drone, :]
        cur_rpy = np.array(pyb.getEulerFromQuaternion(quaternion))
        u, v, w = self.vel[nth_drone, :]
        R_vb = np.array(pyb.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        # We now correct R_vb from Pybullet frame to wind frame
        R_vb = R_vb @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        steady_state = current_wind[0:3]
        gust = current_wind[3:6]
        # convert wind vector from world to body frame and add gust
        wind_body_frame = R_vb @ steady_state + gust
        v_air_i = np.array([u, v, w])
        v_air_b = R_vb.T.dot(v_air_i)
        ur = v_air_b[0] - wind_body_frame[0]
        vr = v_air_b[1] - wind_body_frame[1]
        wr = v_air_b[2] - wind_body_frame[2]
        # compute airspeed
        Va = np.sqrt(ur ** 2 + vr ** 2 + wr ** 2)[0]
        # compute angle of attack
        if ur == 0:
            alpha = np.sign(wr) * np.pi / 2
        else:
            alpha = np.arctan(wr / ur)
        # compute sideslip angle
        if Va == 0:
            beta = np.sign(vr) * np.pi / 2
        else:
            beta = np.arcsin(vr / np.sqrt(ur ** 2 + vr ** 2 + wr ** 2))

        drone = self.drones[nth_drone]
        p, q, r = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ self.ang_v[nth_drone, :]
        cmd_prop = cmd[3] * 1570 + 730
        cmd_elevator = cmd[1]
        cmd_aileron = cmd[0]
        cmd_rudder = cmd[2]
        # small hack to avoid smt to print everything all the time
        sys.stdout = open(os.devnull, 'w')
        T1 = thrust_model.predict_values(np.array([Va, cmd_prop, 0]).reshape((-1, 3)))[0][0]
        Q1 = torque_model.predict_values(np.array([Va, cmd_prop, 0]).reshape((-1, 3)))[0][0]
        # reverting the small hack
        sys.stdout = sys.__stdout__
        # calculate forces and moments - aero
        n_sigma = np.exp(-drone.M * (alpha - drone.alpha0))
        p_sigma = np.exp(drone.M * (alpha + drone.alpha0))
        sigma = (1 + p_sigma + n_sigma) / ((1 + n_sigma) * (1 + p_sigma))
        CL_a = (1 - sigma) * (drone.CL0 + drone.CL_alpha * alpha) + \
               sigma * (2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha))
        CD_a = drone.CD0 + (drone.CL0 + drone.CL_alpha * alpha) ** 2 / (np.pi * drone.oswald * drone.AR)
        CL = (-CD_a * np.sin(alpha) - CL_a * np.cos(alpha)) + \
             ((-drone.CD_q * np.sin(alpha) - drone.CL_q * np.cos(alpha)) * drone.Cref * q / (2 * Va)) + \
             ((-drone.CD_del_e * np.sin(alpha) - drone.CL_del_e * np.cos(alpha)) * cmd_elevator)
        CD = (-CD_a * np.cos(alpha) + CL_a * np.sin(alpha)) + \
             ((-drone.CD_q * np.cos(alpha) + drone.CL_q * np.sin(alpha)) * drone.Cref * q / (2 * Va)) + \
             ((-drone.CD_del_e * np.cos(alpha) + drone.CL_del_e * np.sin(alpha)) * cmd_elevator)
        # compute Lift and Drag Forces
        F_lift = .5 * drone.rho * drone.Sref * Va ** 2 * CL
        F_drag = .5 * drone.rho * drone.Sref * Va ** 2 * CD
        Fy = .5 * drone.rho * drone.Sref * Va ** 2 * (
                drone.CY_beta * beta + drone.CY_p * p * drone.Bref / (2 * Va) +
                drone.CY_r * r * drone.Bref / (2 * Va) +
                drone.CY_del_a * cmd_aileron + drone.CY_del_r * cmd_rudder)
        My = .5 * drone.rho * drone.Sref * Va ** 2 * drone.Cref * \
             (drone.Cm0 + drone.Cm_alpha * alpha + (drone.Cm_q * q * drone.Cref / (2 * Va))
              + drone.Cm_del_e * cmd_elevator)
        Mx = .5 * drone.rho * drone.Sref * Va ** 2 * drone.Bref * (
                drone.Cl_beta * beta + drone.Cl_p * p * drone.Bref / (2 * Va) +
                drone.Cl_r * r * drone.Bref / (2 * Va) +
                drone.Cl_del_a * cmd_aileron + drone.Cl_del_r * cmd_rudder)
        Mz = 0.5 * drone.rho * drone.Sref * Va ** 2 * drone.Bref * (
                drone.Cn_beta * beta + drone.Cn_p * p * drone.Bref / (2 * Va) +
                drone.Cn_r * r * drone.Bref / (2 * Va) +
                drone.Cn_del_a * cmd_aileron + drone.Cn_del_r * cmd_rudder)

        ## AERO FORCES
        pyb.applyExternalForce(self.DRONE_IDS[nth_drone],
                               1,
                               forceObj=[F_drag, -Fy, -F_lift],
                               posObj=[0, 0, 0],
                               flags=pyb.LINK_FRAME,
                               physicsClientId=self.CLIENT
                               )
        ### AERO MOMENTS
        pyb.applyExternalTorque(self.DRONE_IDS[nth_drone],
                                1,
                                torqueObj=[Mx, -My, -Mz],
                                flags=pyb.LINK_FRAME,
                                physicsClientId=self.CLIENT
                                )
        ### Prop1 FORCE
        pyb.applyExternalForce(self.DRONE_IDS[nth_drone],
                               2,
                               forceObj=[T1, 0, 0],
                               posObj=[0, 0, 0],
                               flags=pyb.LINK_FRAME,
                               physicsClientId=self.CLIENT
                               )
        ### Prop1 MOMENT
        pyb.applyExternalTorque(self.DRONE_IDS[nth_drone],
                                2,
                                torqueObj=[-Q1, 0, 0],
                                flags=pyb.LINK_FRAME,
                                physicsClientId=self.CLIENT
                                )
        ################################################################################
        def _dynamics(self,
                      rpm,
                      nth_drone
                      ):
            """Explicit dynamics implementation.
            Based on code written at the Dynamic Systems Lab by James Xu.
            Parameters
            ----------
            rpm : ndarray
                (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone : int
                The ordinal number/position of the desired drone in list self.DRONE_IDS.
            """
            #### Current state #########################################
            pos = self.pos[nth_drone, :]
            quat = self.quat[nth_drone, :]
            rpy = self.rpy[nth_drone, :]
            vel = self.vel[nth_drone, :]
            rpy_rates = self.rpy_rates[nth_drone, :]
            rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            #### Compute forces and torques ############################
            forces = np.array(rpm ** 2) * self.KF
            thrust = np.array([0, 0, np.sum(forces)])
            thrust_world_frame = np.dot(rotation, thrust)
            force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
            z_torques = np.array(rpm ** 2) * self.KM
            z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
            if self.DRONE_MODEL == DroneModel.CF2X:
                x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
                y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
            elif self.DRONE_MODEL == DroneModel.CF2P or self.DRONE_MODEL == DroneModel.HB:
                x_torque = (forces[1] - forces[3]) * self.L
                y_torque = (-forces[0] + forces[2]) * self.L
            torques = np.array([x_torque, y_torque, z_torque])
            torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
            rpy_rates_deriv = np.dot(self.J_INV, torques)
            no_pybullet_dyn_accs = force_world_frame / self.M
            #### Update state ##########################################
            vel = vel + self.TIMESTEP * no_pybullet_dyn_accs
            rpy_rates = rpy_rates + self.TIMESTEP * rpy_rates_deriv
            pos = pos + self.TIMESTEP * vel
            rpy = rpy + self.TIMESTEP * rpy_rates
            #### Set PyBullet's state ##################################
            pyb.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                                pos,
                                                pyb.getQuaternionFromEuler(rpy),
                                                physicsClientId=self.CLIENT
                                                )
            #### Note: the base's velocity only stored and not used ####
            pyb.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                                  vel,
                                  [-1, -1, -1],  # ang_vel not computed by DYN
                                  physicsClientId=self.CLIENT
                                  )
            #### Store the roll, pitch, yaw rates for the next step ####
            self.rpy_rates[nth_drone, :] = rpy_rates

        ################################################################################

        def _normalizedActionToRPM(self,
                                   action
                                   ):
            """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.
            Parameters
            ----------
            action : ndarray
                (4)-shaped array of ints containing an input in the [-1, 1] range.
            Returns
            -------
            ndarray
                (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.
            """
            if np.any(np.abs(action)) > 1:
                print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
            return np.where(action <= 0, (action + 1) * self.HOVER_RPM,
                            action * self.MAX_RPM)  # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM

        ################################################################################

        def _saveLastAction(self,
                            action
                            ):
            """Stores the most recent action into attribute `self.last_action`.
            The last action can be used to compute aerodynamic effects.
            The method disambiguates between array and dict inputs
            (for single or multi-agent aviaries, respectively).
            Parameters
            ----------
            action : ndarray | dict
                (4)-shaped array of ints (or dictionary of arrays) containing the current RPMs input.
            """
            if isinstance(action, collections.Mapping):
                for k, v in action.items():
                    res_v = np.resize(v, (
                    1, 4))  # Resize, possibly with repetition, to cope with different action spaces in RL subclasses
                    self.last_action[int(k), :] = res_v
            else:
                res_action = np.resize(action, (
                1, 4))  # Resize, possibly with repetition, to cope with different action spaces in RL subclasses
                self.last_action = np.reshape(res_action, (self.NUM_DRONES, 4))

        ################################################################################

        def _showDroneLocalAxes(self,
                                nth_drone
                                ):
            """Draws the local frame of the n-th drone in PyBullet's GUI.
            Parameters
            ----------
            nth_drone : int
                The ordinal number/position of the desired drone in list self.DRONE_IDS.
            """
            if self.GUI:
                AXIS_LENGTH = 2 * self.L
                self.X_AX[nth_drone] = pyb.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                            lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                            lineColorRGB=[1, 0, 0],
                                                            parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                            parentLinkIndex=-1,
                                                            replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                            physicsClientId=self.CLIENT
                                                            )

                self.Y_AX[nth_drone] = pyb.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                            lineToXYZ=[0, AXIS_LENGTH, 0],
                                                            lineColorRGB=[0, 1, 0],
                                                            parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                            parentLinkIndex=-1,
                                                            replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                            physicsClientId=self.CLIENT
                                                            )
                self.Z_AX[nth_drone] = pyb.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                            lineToXYZ=[0, 0, AXIS_LENGTH],
                                                            lineColorRGB=[0, 0, 1],
                                                            parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                            parentLinkIndex=-1,
                                                            replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                            physicsClientId=self.CLIENT
                                                            )
                import pdb
                print('client :', self.CLIENT)
                print('parentObjectUniqueId= ', self.DRONE_IDS[nth_drone])
                print('int(self.Z_AX[nth_drone] :', int(self.Z_AX[nth_drone]))
                pdb.set_trace()

    ################################################################################


    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        # p.loadURDF("samurai.urdf",
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadSDF("stadium.sdf",
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/voliere.urdf",
        #            physicsClientId=self.CLIENT
        #            )
        pyb.loadURDF("duck_vhacd.urdf",
                   [-.5, -.5, .05],
                   pyb.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        pyb.loadURDF("cube_no_rotation.urdf",
                   [-.5, -2.5, .5],
                   pyb.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        pyb.loadURDF("sphere2.urdf",
                   [-2, 5, .5],
                   pyb.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=self.CLIENT
                   )

    ################################################################################
    def _parseURDFFixedwingParameters(self,drone,URDF):
        ''' Loads Fixed-wing related coefficients from URDF file '''
        URDF_TREE = etxml.parse(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+URDF).getroot()

        ref = URDF_TREE.find("fixed_wing_aero_coeffs/ref")
        drone.alpha0 = float(ref.attrib['alpha0'])
        drone.Bref = float(ref.attrib['Bref'])
        drone.Sref = float(ref.attrib['Sref'])
        drone.Cref = float(ref.attrib['Cref'])
        drone.Vref = float(ref.attrib['Vref'])

        cl = URDF_TREE.find("fixed_wing_aero_coeffs/CL")
        drone.CL0 = float(cl.attrib['CL0'])
        drone.CL_alpha = float(cl.attrib['CL_alpha'])
        drone.CL_beta  = float(cl.attrib['CL_beta'])
        vals = str(cl.attrib['CL_omega'])
        drone.CL_omega = [float(s) for s in vals.split(' ') if s != '']
        vals = str(cl.attrib['CL_ctrl'])
        drone.CL_ctrl = [float(s) for s in vals.split(' ') if s != '']

        cd = URDF_TREE.find("fixed_wing_aero_coeffs/CD")
        drone.CD0 = float(cd.attrib['CD0'])
        drone.CD_k1 = float(cd.attrib['CD_k1'])
        drone.CD_k2 = float(cd.attrib['CD_k1'])
        vals = str(cd.attrib['CD_ctrl'])
        drone.CD_ctrl = [float(s) for s in vals.split(' ') if s != '']

        cy = URDF_TREE.find("fixed_wing_aero_coeffs/CY")
        drone.CY_alpha = float(cy.attrib['CY_alpha'])
        drone.CY_beta = float(cy.attrib['CY_beta'])
        vals = str(cy.attrib['CY_omega'])
        drone.CY_omega  = [float(s) for s in vals.split(' ') if s != '']
        vals = str(cy.attrib['CY_ctrl'])
        drone.CY_ctrl = [float(s) for s in vals.split(' ') if s != '']

        cl = URDF_TREE.find("fixed_wing_aero_coeffs/Cl")
        drone.Cl_alpha = float(cl.attrib['Cl_alpha'])
        drone.Cl_beta = float(cl.attrib['Cl_beta'])
        vals = str(cl.attrib['Cl_omega'])
        drone.Cl_omega  = [float(s) for s in vals.split(' ') if s != '']
        vals = str(cl.attrib['Cl_ctrl'])
        drone.Cl_ctrl = [float(s) for s in vals.split(' ') if s != '']

        cm = URDF_TREE.find("fixed_wing_aero_coeffs/Cm")
        drone.Cm0 = float(cm.attrib['Cm0'])
        drone.Cm_alpha = float(cm.attrib['Cm_alpha'])
        drone.Cm_beta = float(cm.attrib['Cm_beta'])
        vals = str(cm.attrib['Cm_omega'])
        drone.Cm_omega  = [float(s) for s in vals.split(' ') if s != '']
        vals = str(cm.attrib['Cm_ctrl'])
        drone.Cm_ctrl = [float(s) for s in vals.split(' ') if s != '']

        cn = URDF_TREE.find("fixed_wing_aero_coeffs/Cn")
        drone.Cn_alpha = float(cn.attrib['Cn_alpha'])
        drone.Cn_beta = float(cn.attrib['Cn_beta'])
        vals = str(cn.attrib['Cn_omega'])
        drone.Cn_omega = [float(s) for s in vals.split(' ') if s != '']
        vals = str(cn.attrib['Cn_ctrl'])
        drone.Cn_ctrl = [float(s) for s in vals.split(' ') if s != '']

    ################################################################################

    def _parseURDFVTOLParameters(self, drone, URDF):
        ''' Loads Fixed-wing related coefficients from URDF file '''
        URDF_TREE = etxml.parse(os.path.dirname(os.path.abspath(__file__)) + "/../assets/" + URDF).getroot()

        ref = URDF_TREE.find("aero_coeffs/ref")
        drone.M = float(ref.attrib['M'])
        drone.alpha0 = float(ref.attrib['alpha0'])
        drone.oswald = float(ref.attrib['oswald'])
        drone.rho = float(ref.attrib['rho'])
        if self.GEOMETRY_COEFFS == {}:
            drone.AR = float(ref.attrib['AR'])
            drone.Bref = float(ref.attrib['Bref'])
            drone.Sref = float(ref.attrib['Sref'])
            drone.Cref = float(ref.attrib['Cref'])
        else:
            drone.AR =  self.GEOMETRY_COEFFS['AR']
            drone.Bref = self.GEOMETRY_COEFFS['Bref']
            drone.Sref = self.GEOMETRY_COEFFS['Sref']
            drone.Cref = self.GEOMETRY_COEFFS['Cref']

        if self.AERO_COEFFS == {}:
            CL = URDF_TREE.find("aero_coeffs/CL")
            drone.CL0 = float(CL.attrib['CL0'])
            drone.CL_alpha = float(CL.attrib['CL_alpha'])
            drone.CL_q = float(CL.attrib['CL_q'])
            drone.CL_del_e = float(CL.attrib['CL_del_e'])

            cd = URDF_TREE.find("aero_coeffs/CD")
            drone.CD0 = float(cd.attrib['CD0'])
            drone.CD_q = float(cd.attrib['CD_q'])
            drone.CD_del_e = float(cd.attrib['CD_del_e'])

            cy = URDF_TREE.find("aero_coeffs/CY")
            drone.CY0 = float(cy.attrib['CY0'])
            drone.CY_beta = float(cy.attrib['CY_beta'])
            drone.CY_p = float(cy.attrib['CY_p'])
            drone.CY_r = float(cy.attrib['CY_r'])
            drone.CY_del_r = float(cy.attrib['CY_del_r'])
            drone.CY_del_a = float(cy.attrib['CY_del_a'])

            cl = URDF_TREE.find("aero_coeffs/Cl")
            drone.Cl_beta = float(cl.attrib['Cl_beta'])
            drone.Cl_p = float(cl.attrib['Cl_p'])
            drone.Cl_r = float(cl.attrib['Cl_r'])
            drone.Cl_del_r = float(cl.attrib['Cl_del_r'])
            drone.Cl_del_a = float(cl.attrib['Cl_del_a'])

            cm = URDF_TREE.find("aero_coeffs/Cm")
            drone.Cm0 = float(cm.attrib['Cm0'])
            drone.Cm_alpha = float(cm.attrib['Cm_alpha'])
            drone.Cm_q = float(cm.attrib['Cm_q'])
            drone.Cm_del_e = float(cm.attrib['Cm_del_e'])

            cn = URDF_TREE.find("aero_coeffs/Cn")
            drone.Cn_beta = float(cn.attrib['Cn_beta'])
            drone.Cn_p = float(cn.attrib['Cn_p'])
            drone.Cn_r = float(cn.attrib['Cn_r'])
            drone.Cn_del_r = float(cn.attrib['Cn_del_r'])
            drone.Cn_del_a = float(cn.attrib['Cn_del_a'])
        else:
            drone.CL0 = self.AERO_COEFFS['CL0']
            drone.CL_alpha = self.AERO_COEFFS['CL_alpha']
            drone.CL_q = self.AERO_COEFFS['CL_q']
            drone.CL_del_e = self.AERO_COEFFS['CL_del_e']
            drone.CD0 = self.AERO_COEFFS['CD0']
            drone.CD_q = self.AERO_COEFFS['CD_q']
            drone.CD_del_e = self.AERO_COEFFS['CD_del_e']
            drone.CY0 = self.AERO_COEFFS['CY0']
            drone.CY_beta = self.AERO_COEFFS['CY_beta']
            drone.CY_p = self.AERO_COEFFS['CY_p']
            drone.CY_r = self.AERO_COEFFS['CY_r']
            drone.CY_del_r = self.AERO_COEFFS['CY_del_r']
            drone.CY_del_a = self.AERO_COEFFS['CY_del_a']
            drone.Cl_beta = self.AERO_COEFFS['Cl_beta']
            drone.Cl_p = self.AERO_COEFFS['Cl_p']
            drone.Cl_r = self.AERO_COEFFS['Cl_r']
            drone.Cl_del_r = self.AERO_COEFFS['Cl_del_r']
            drone.Cl_del_a = self.AERO_COEFFS['Cl_del_a']
            drone.Cm0 = self.AERO_COEFFS['Cm0']
            drone.Cm_alpha = self.AERO_COEFFS['Cm_alpha']
            drone.Cm_q = self.AERO_COEFFS['Cm_q']
            drone.Cm_del_e = self.AERO_COEFFS['Cm_del_e']
            drone.Cn_beta = self.AERO_COEFFS['Cn_beta']
            drone.Cn_p = self.AERO_COEFFS['Cn_p']
            drone.Cn_r = self.AERO_COEFFS['Cn_r']
            drone.Cn_del_r = self.AERO_COEFFS['Cn_del_r']
            drone.Cn_del_a = self.AERO_COEFFS['Cn_del_a']
            drone.oswald = self.AERO_COEFFS['oswald']


        Thrust = URDF_TREE.find("motor_coeffs/ref")
        drone.prop_angle = float(Thrust.attrib['prop_angle'])
        #drone.x_pos = float(Thrust.attrib['x_pos'])
        #drone.y_pos = float(Thrust.attrib['y_pos'])
        #drone.z_pos = float(Thrust.attrib['z_pos'])


    ###############################################################################
    def _parseURDFParameters(self,URDF):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+URDF).getroot()

        conf = URDF_TREE.find("configuration")
        TYPE = str(conf.attrib['type'])

        mass = URDF_TREE.find("link/inertial/mass")
        M = float(mass.attrib['value'])


        prop = URDF_TREE.find("properties")
        L = float(prop.attrib['arm'])
        THRUST2WEIGHT_RATIO = float(prop.attrib['thrust2weight'])
        KF = float(prop.attrib['kf'])
        KM = float(prop.attrib['km'])

        if self.GEOMETRY_COEFFS == {}:
            inertia = URDF_TREE.find("link/inertial/inertia")
            IXX = float(inertia.attrib['ixx'])
            IYY = float(inertia.attrib['iyy'])
            IZZ = float(inertia.attrib['izz'])
            J = np.diag([IXX, IYY, IZZ])
            J_INV = np.linalg.inv(J)
        else:
            IXX =  self.GEOMETRY_COEFFS['Ixx']
            IYY =  self.GEOMETRY_COEFFS['Iyy']
            IZZ =  self.GEOMETRY_COEFFS['Izz']
            J = np.diag([IXX, IYY, IZZ])
            J_INV = np.linalg.inv(J)

        coll = URDF_TREE.find("link/collision/geometry/cylinder")
        COLLISION_H = float(coll.attrib['length'])
        COLLISION_R = float(coll.attrib['radius'])

        coll_offset = URDF_TREE.find("link/collision/origin")
        COLLISION_SHAPE_OFFSETS = [float(s) for s in coll_offset.attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]

        MAX_SPEED_KMH = float(prop.attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(prop.attrib['gnd_eff_coeff'])
        PROP_RADIUS   = float(prop.attrib['prop_radius'])
        DRAG_COEFF_XY = float(prop.attrib['drag_coeff_xy'])
        DRAG_COEFF_Z  = float(prop.attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])

        DW_COEFF_1 = float(prop.attrib['dw_coeff_1'])
        DW_COEFF_2 = float(prop.attrib['dw_coeff_2'])
        DW_COEFF_3 = float(prop.attrib['dw_coeff_3'])

        indi = URDF_TREE.find("control/indi")
        indi_actuator_nr = int(indi.attrib['actuator_nr'])
        indi_output_nr = int(indi.attrib['output_nr'])
        G1 = np.zeros((indi_output_nr, indi_actuator_nr))

        pwm2rpm = URDF_TREE.find("control/pwm/pwm2rpm")
        # PWM2RPM_SCALE = float(pwm2rpm.attrib['scale'])
        # PWM2RPM_CONST = float(pwm2rpm.attrib['const'])
        vals = [str(k) for k in pwm2rpm.attrib.values()]
        PWM2RPM_SCALE = [float(s) for s in vals[0].split(' ') if s != '']
        PWM2RPM_CONST = [float(s) for s in vals[1].split(' ') if s != '']

        pwmlimit = URDF_TREE.find("control/pwm/limit")
        vals = [str(k) for k in pwmlimit.attrib.values()]
        MIN_PWM = [float(s) for s in vals[0].split(' ') if s != '']
        MAX_PWM = [float(s) for s in vals[1].split(' ') if s != '']


        return TYPE, M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3, PWM2RPM_SCALE, PWM2RPM_CONST, \
               indi_actuator_nr, indi_output_nr, G1, MIN_PWM, MAX_PWM
    
    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs.

        """
        raise NotImplementedError

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeDone(self):
        """Computes the current done value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError
