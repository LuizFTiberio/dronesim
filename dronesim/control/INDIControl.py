import os
import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as etxml

from dronesim.control.BaseControl import BaseControl
from dronesim.envs.BaseAviary import DroneModel, BaseAviary

import pdb


# Active set library from : https://github.com/JimVaranelli/ActiveSet
import sys
# from dronesim.control.ActiveSet import ActiveSet, ConstrainedLS
from dronesim.control.wls_alloc import wls_alloc

from dataclasses import dataclass

# @dataclass
# class PlayingCard:
#     rank: str
#     suit: str

@dataclass
class Rate:
    p: float=0.
    q: float=0.
    r: float=0.

@dataclass
class Gains:
    att = Rate()
    rate = Rate()

def quat_comp(a2b,b2c):
    # qi,qx,qy,qz = 0,1,2,3
    qi,qx,qy,qz = 3,0,1,2
    a2c = np.zeros(4)
    a2c[qi] = a2b[qi] * b2c[qi] - a2b[qx] * b2c[qx] - a2b[qy] * b2c[qy] - a2b[qz] * b2c[qz]
    a2c[qx] = a2b[qi] * b2c[qx] + a2b[qx] * b2c[qi] + a2b[qy] * b2c[qz] - a2b[qz] * b2c[qy]
    a2c[qy] = a2b[qi] * b2c[qy] - a2b[qx] * b2c[qz] + a2b[qy] * b2c[qi] + a2b[qz] * b2c[qx]
    a2c[qz] = a2b[qi] * b2c[qz] + a2b[qx] * b2c[qy] - a2b[qy] * b2c[qx] + a2b[qz] * b2c[qi]
    return a2c

def quat_inv_comp(q1,q2):
    # i,x,y,z = 0,1,2,3
    i,x,y,z = 3,0,1,2
    qerr = np.zeros(4)
    qerr[i] = q1[i] * q2[i] + q1[x] * q2[x] + q1[y] * q2[y] + q1[z] * q2[z]
    qerr[x] = q1[i] * q2[x] - q1[x] * q2[i] - q1[y] * q2[z] + q1[z] * q2[y]
    qerr[y] = q1[i] * q2[y] + q1[x] * q2[z] - q1[y] * q2[i] - q1[z] * q2[x]
    qerr[z] = q1[i] * q2[z] - q1[x] * q2[y] + q1[y] * q2[x] - q1[z] * q2[i]
    return qerr

def quat_norm(q):
    return np.sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3])

def quat_normalize(q):
    n = quat_norm(q)
    if (n > 0.):
        for i in range(4):
            q[i]= q[i]/n
    return q

def quat_wrap_shortest(q):
    w=3 # 0 or 3 according to quaternion definition.
    if (q[w] < 0) : 
        for i in range(4): # QUAT_EXPLEMENTARY(q)
            q[i]=-q[i]
    return q


def thrust_from_rpm(rpm):
    ''' input is the array of actuator rpms'''
    thrust = 0.
    for _rpm in rpm:
        thrust += _rpm**2.*3.16e-10
    return thrust

def skew(w):
    return np.array([[   0., -w[2],  w[1]],
                     [ w[2],    0., -w[0]],
                     [-w[1],  w[1],    0.] ])

def jac_vec_quat(vec,q):
    w = q[3]
    v = q[:3]
    I = np.eye(3)
    p1 = w*vec + np.cross(v,vec)
    p2 = np.dot(np.dot(v.T,vec),I) + v.dot(vec.T) - vec.dot(v.T) - w*skew(vec)
    return np.hstack([p1.reshape(3,1),p2])*2 # p1, p2

def normalize_angle(angle):
    if angle > np.pi:
        angle -=  np.pi
    if angle < -np.pi:
        angle +=  np.pi
    return angle


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]

class INDIControl(BaseControl):
    """INDI control class for Crazyflies.

    by Murat Bronz based on work conducted at TUDelft by Ewoud Smeur.

    """

    ################################################################################

    def __init__(self,
                 drone_model: list=['tello'],
                 g: float=9.8,
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        self.DRONE_MODEL = drone_model
        self.URDF = self.DRONE_MODEL + ".urdf"

        # this is being called from the init of BaseControl !
        # self._parseURDFControlParameters() 
        #========
        # self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        # self.I_COEFF_FOR = np.array([.05, .05, .05])
        # self.D_COEFF_FOR = np.array([.2, .2, .5])
        # self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        # self.I_COEFF_TOR = np.array([.0, .0, 500.])
        # self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        # self.PWM2RPM_SCALE = 0.2685
        # self.PWM2RPM_CONST = 4070.3
        # self.MIN_PWM = 20000
        # self.MAX_PWM = 65535
        # if self.DRONE_MODEL == DroneModel.CF2X:
        #     self.MIXER_MATRIX = np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ])
        # elif self.DRONE_MODEL == DroneModel.CF2P:
        #     self.MIXER_MATRIX = np.array([ [0, -1,  -1], [+1, 0, 1], [0,  1,  -1], [-1, 0, 1] ])
        #========
        self.reset()

    ################################################################################
    def _parseURDFControlParameters(self):
        """Loads Control parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF).getroot()

        mass = URDF_TREE.find("link/inertial/mass")
        self.m = float(mass.attrib['value'])

        indi = URDF_TREE.find("control/indi")
        self.indi_actuator_nr = int(indi.attrib['actuator_nr'])
        self.indi_output_nr = int(indi.attrib['output_nr'])
        self.G1 = np.zeros((self.indi_output_nr, self.indi_actuator_nr))
        print('*************')

        indi = URDF_TREE.find("control")
        for i in range(self.indi_output_nr):
            vals = [str(k) for k in indi[i+1].attrib.values()]
            self.G1[i] = [float(s) for s in vals[0].split(' ') if s != '']

        self.indi_gains = Gains()
        guidance_gains = URDF_TREE.find("control/indi_guidance_gains/pos")
        self.guidance_indi_pos_gain   = float(guidance_gains.attrib['kp'])
        self.guidance_indi_speed_gain = float(guidance_gains.attrib['kd'])

        att_att_gains = URDF_TREE.find("control/indi_att_gains/att")
        att_rate_gains = URDF_TREE.find("control/indi_att_gains/rate")

        self.indi_gains.att.p = float(att_att_gains.attrib['p'])
        self.indi_gains.att.q = float(att_att_gains.attrib['q'])
        self.indi_gains.att.r = float(att_att_gains.attrib['r'])
        self.indi_gains.rate.p = float(att_rate_gains.attrib['p'])
        self.indi_gains.rate.q = float(att_rate_gains.attrib['q'])
        self.indi_gains.rate.r = float(att_rate_gains.attrib['r'])

        pwm2rpm = URDF_TREE.find("control/pwm/pwm2rpm")
        # self.PWM2RPM_SCALE = float(pwm2rpm.attrib['scale'])
        # self.PWM2RPM_CONST = float(pwm2rpm.attrib['const'])
        vals = [str(k) for k in pwm2rpm.attrib.values()]
        self.PWM2RPM_SCALE = [float(s) for s in vals[0].split(' ') if s != '']
        self.PWM2RPM_CONST = [float(s) for s in vals[1].split(' ') if s != '']

        pwmlimit = URDF_TREE.find("control/pwm/limit")
        # self.MIN_PWM = float(pwmlimit.attrib['min'])
        # self.MAX_PWM = float(pwmlimit.attrib['max'])
        vals = [str(k) for k in pwmlimit.attrib.values()]
        self.MIN_PWM = [float(s) for s in vals[0].split(' ') if s != '']
        self.MAX_PWM = [float(s) for s in vals[1].split(' ') if s != '']

    ################################################################################        
    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        self.rpy = np.zeros(3)

        self.last_rates = np.zeros(3)  # p,q,r 
        # self.last_pwm = np.ones(self.indi_actuator_nr)*1. # initial pwm
        self.last_thrust = 0.0
        # self.indi_increment = np.zeros(4)
        #self.cmd = np.ones(self.indi_actuator_nr)*0.5
        self.cmd = np.array([.8,.8,.8,.8]) # by doing that i make sure it starts smooth for fixed wing
        self.last_vel = np.zeros(3)
        self.last_torque = np.zeros(3) # For SU2 controller

        self.xax = -1
        self.yax = -1
        self.zax = -1
        self.xax1 = -2
        self.yax1 = -2
        self.zax1 = -2

        # for debugging logs...
        self.att_log  = np.zeros((30*100, 20)) 
        self.guid_log = np.zeros((30*100, 20))
        self.att_log_inc = 0
        self.guid_log_inc = 0

        self.rpm = np.zeros(self.indi_actuator_nr)



    def rpm_of_pwm(self, pwm):
        self.rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        return self.rpm

    ################################################################################
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e, quat_ = self._INDIPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )

        rpm = self._INDIAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          cur_ang_vel,
                                          computed_target_rpy,
                                          quat_,
                                          target_rpy_rates
                                          )

        cur_rpy = p.getEulerFromQuaternion(cur_quat)

        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]


    def computeControl_hybrid(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       current_wind = np.zeros(6),
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        #print(self.control_counter)

        if self.control_counter ==  1500:
            print('debug')

        nav_carrot = self._WayPointNavigation(control_timestep,
                                              cur_pos,
                                              target_pos)

        gi_speed_sp = self._compute_guidance_indi_run_pos(control_timestep,
                               cur_pos,
                               cur_vel,
                               nav_carrot,
                               target_vel)

        sp_accel = self._compute_accel_from_speed_sp(control_timestep, cur_quat, cur_vel, gi_speed_sp,
                                                     current_wind)

        thrust, computed_target_rpy = self._guidance_indi_hybrid_run(control_timestep,
                                                                     cur_pos,
                                                                     cur_quat,
                                                                     cur_vel,
                                                                     nav_carrot,
                                                                     target_rpy,
                                                                     target_vel,
                                                                     sp_accel,
                                                                     current_wind)


        print(self.control_counter,gi_speed_sp,sp_accel,np.degrees(computed_target_rpy))

        rpm = self._INDIAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          cur_ang_vel,
                                          computed_target_rpy)

        return rpm, 0. ,0.

    
    ################################################################################

    ################################################################################
    def _INDIPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel,
                               use_quaternion = False,
                               nonlinear_increment = False
                               ):

        """ENAC generic INDI position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        debug_log = False

        # Linear controller to find the acceleration setpoint from position and velocity
        pos_e = target_pos - cur_pos

        # Speed setpoint
        speed_sp = pos_e * self.guidance_indi_pos_gain

        #TODO: I'm forcing target to current to avoid defining the navigation for now!
        target_vel = cur_vel
        vel_e = speed_sp + target_vel - cur_vel

        # Set acceleration setpoint :
        accel_sp = vel_e * self.guidance_indi_speed_gain


        # Calculate the acceleration via finite difference TODO : this is a rotated sensor output in real life, so add sensor to the sim !
        if self.control_counter == 1:
            # lets avoid assuming v(t-1) is zero because it causes explosion in the accel error
            self.last_vel = cur_vel
        cur_accel = (cur_vel - self.last_vel) / control_timestep
        self.last_vel = cur_vel
        accel_e = accel_sp - cur_accel
        accel_e = np.array([accel_e[0], 0, 0])

        # Bound the acceleration error so that the linearization still holds
        accel_e = np.clip(accel_e, -6.0, 6.0) # For Z : -9.0, 9.0 FIX ME !

        # EULER VERSION
        # # Calculate matrix of partial derivatives
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        phi, theta, psi = cur_rpy[0],cur_rpy[1],cur_rpy[2]
        theta = np.pi/2 - theta

        sphi,stheta,spsi = np.sin(phi),np.sin(theta),np.sin(psi)
        cphi,ctheta,cpsi = np.cos(phi),np.cos(theta),np.cos(psi)

        lift = - np.sin(-theta) * self.GRAVITY
        T = -np.cos(theta) * self.GRAVITY

        min_pitch = -80.0
        middle_pitch = -50.0
        max_pitch = -20.0
        GUIDANCE_INDI_LIFTD_ASQ = 0.20
        GUIDANCE_INDI_LIFTD_P80 = GUIDANCE_INDI_LIFTD_ASQ * 12**2
        GUIDANCE_INDI_LIFTD_P50 = GUIDANCE_INDI_LIFTD_P80/2
        airspeed = np.linalg.norm(cur_vel)
        pitch_interp = np.clip(np.degrees(theta),min_pitch, max_pitch)
        if airspeed < 12:
            if pitch_interp > middle_pitch:
                ratio = (pitch_interp - max_pitch) / (middle_pitch - max_pitch)
                liftd = -GUIDANCE_INDI_LIFTD_P50 * ratio
            else:
                ratio = (pitch_interp - middle_pitch) / (min_pitch - middle_pitch)
                liftd = -(GUIDANCE_INDI_LIFTD_P80-GUIDANCE_INDI_LIFTD_P50) * ratio - GUIDANCE_INDI_LIFTD_P50

        else:
            liftd = -GUIDANCE_INDI_LIFTD_ASQ * airspeed * airspeed

        # Matrix of partial derivatives for Lift force
        GUIDANCE_INDI_PITCH_EFF_SCALING = 1.0

        G_0_0 = cphi * ctheta * spsi * T + cphi * spsi * lift;
        G_1_0 = -cphi * ctheta * cpsi * T - cphi * cpsi * lift;
        G_2_0 = -sphi * ctheta * T - sphi * lift;
        G_0_1 = (ctheta * cpsi - sphi * stheta * spsi) * T * GUIDANCE_INDI_PITCH_EFF_SCALING + sphi * spsi * liftd;
        G_1_1 = (ctheta * spsi + sphi * stheta * cpsi) * T * GUIDANCE_INDI_PITCH_EFF_SCALING - sphi * cpsi * liftd;
        G_2_1 = -cphi * stheta * T * GUIDANCE_INDI_PITCH_EFF_SCALING + cphi * liftd;
        G_0_2 = stheta * cpsi + sphi * ctheta * spsi
        G_1_2 = stheta * spsi - sphi * ctheta * cpsi
        G_2_2 = cphi * ctheta

        G = np.array([[G_0_0, G_0_1, -G_0_2],
                      [G_1_0, G_1_1, -G_1_2],
                      [G_2_0, G_2_1, -G_2_2]])

        # Invert this matrix
        G_inv = np.linalg.pinv(G) #FIX ME

        # Calculate roll,pitch and thrust command
        control_increment = G_inv.dot(accel_e)

        target_quat = np.array([0.0, 0.0, 0.0, 1.0])
        yaw_increment = target_rpy[2] - psi #cur_rpy[2]
        target_euler = (phi,theta-np.pi/2,psi) + np.array([control_increment[0], control_increment[1], yaw_increment])
        target_euler = np.array([target_euler[0],0,0])

        thrust = self.last_thrust + control_increment[2]


        return thrust, target_euler, pos_e, target_quat #quat_increment
    
    ################################################################################

    def _INDIAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               cur_ang_vel,
                               target_euler
                               ):
        """INDI attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (actuator_nr,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        att_err = target_euler - cur_rpy
        if  att_err[2] >= 2*np.pi * 0.9:
            att_err[2] -= 2*np.pi
        elif att_err[2] <= -2*np.pi * 0.9:
             att_err[2] += 2*np.pi


        # local variable to compute rate setpoints based on attitude error
        rate_sp = Rate()

        rate_sp.p =  self.indi_gains.att.p * att_err[0] / self.indi_gains.rate.p
        rate_sp.q =  self.indi_gains.att.q * att_err[1] / self.indi_gains.rate.q
        rate_sp.r =  self.indi_gains.att.r * att_err[2] / self.indi_gains.rate.r

        # Rotate angular velocity to body frame
        R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_ang_vel = R.T.dot(cur_ang_vel)

        # Calculate the angular acceleration via finite difference
        if self.control_counter == 1:
            angular_accel = np.zeros(3)
        else:
            angular_accel = (cur_ang_vel - self.last_rates) / (1.*control_timestep)

        # Filter the "noisy" angular velocities : Doing nothing for the moment... placeholder.
        rates_filt = Rate()
        rates_filt.p = cur_ang_vel[0]
        rates_filt.q = cur_ang_vel[1]
        rates_filt.r = cur_ang_vel[2]

        # Remember the last rates for differentiation on the next step
        self.last_rates = cur_ang_vel

        # Calculate the virtual control (reference acceleration) based on a PD controller
        angular_accel_ref = Rate()
        angular_accel_ref.p = (rate_sp.p - rates_filt.p) * self.indi_gains.rate.p
        angular_accel_ref.q = (rate_sp.q - rates_filt.q) * self.indi_gains.rate.q
        angular_accel_ref.r = (rate_sp.r - rates_filt.r) * self.indi_gains.rate.r

        indi_v = np.zeros(4) # roll-pitch-yaw-thrust
        indi_v[0] = angular_accel_ref.p - angular_accel[0]
        indi_v[1] = angular_accel_ref.q - angular_accel[1] 
        indi_v[2] = angular_accel_ref.r - angular_accel[2]
        indi_v[3] = thrust - self.last_thrust #* 0.
        self.last_thrust = thrust


        indi_du = np.dot(np.linalg.pinv(self.G1),indi_v)
        self.cmd += indi_du
        self.cmd = np.clip(self.cmd, self.MIN_PWM, self.MAX_PWM) # command in PWM

        return self.cmd
    
    ################################################################################
    def _guidance_indi_hybrid_run(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel,
                               sp_accel,
                               current_wind
                               ):

        """ENAC generic INDI position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        K_beta = 4
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        rphi, rtheta, rpsi = cur_rpy[0], cur_rpy[1], cur_rpy[2]

        # For INDI, theta is 0 when hover and -90 in cruise
        theta = -np.radians(90) - rtheta
        phi = rphi
        psi = -rpsi

        crphi = np.cos(phi)
        srphi = np.sin(phi)
        crtheta = np.cos(theta)
        srtheta = np.sin(theta)
        crpsi = np.cos(psi)
        srpsi = np.sin(psi)

        # Calculate the transition percentage so that the ctrl_effecitveness scheduling works
        transition_percentage = theta/np.radians(-75) * 100
        transition_percentage = np.clip(transition_percentage,0,100)


        lift = -np.sin(-theta) * self.GRAVITY/9.8
        T = -np.cos(theta) * self.GRAVITY/9.8
        min_pitch = -80.0
        middle_pitch = -50.0
        max_pitch = -20.0
        GUIDANCE_INDI_LIFTD_ASQ = 0.20
        GUIDANCE_INDI_LIFTD_P80 = GUIDANCE_INDI_LIFTD_ASQ * 12 ** 2
        GUIDANCE_INDI_LIFTD_P50 = GUIDANCE_INDI_LIFTD_P80 / 2
        airspeed = np.linalg.norm(cur_vel)
        pitch_interp = np.clip(np.degrees(theta), min_pitch, max_pitch)
        if airspeed < 12:
            if pitch_interp > middle_pitch:
                ratio = (pitch_interp - max_pitch) / (middle_pitch - max_pitch)
                liftd = -GUIDANCE_INDI_LIFTD_P50 * ratio
            else:
                ratio = (pitch_interp - middle_pitch) / (min_pitch - middle_pitch)
                liftd = -(GUIDANCE_INDI_LIFTD_P80 - GUIDANCE_INDI_LIFTD_P50) * ratio - GUIDANCE_INDI_LIFTD_P50
        else:
            liftd = -GUIDANCE_INDI_LIFTD_ASQ * airspeed * airspeed
        # Matrix of partial derivatives for Lift force
        GUIDANCE_INDI_PITCH_EFF_SCALING = 1.0


        G_0_0 = crphi*crtheta*srpsi*T + crphi*srpsi*lift
        G_1_0 = -crphi*crtheta*crpsi*T - crphi*crpsi*lift
        G_2_0 = -srphi*crtheta*T -srphi*lift
        G_0_1 = (crtheta*crpsi - srphi*srtheta*srpsi)*T*GUIDANCE_INDI_PITCH_EFF_SCALING + srphi*srpsi*liftd
        G_1_1 = (crtheta*srpsi + srphi*srtheta*crpsi)*T*GUIDANCE_INDI_PITCH_EFF_SCALING - srphi*crpsi*liftd
        G_2_1 = -crphi*srtheta*T*GUIDANCE_INDI_PITCH_EFF_SCALING + crphi*liftd
        G_0_2 = srtheta*crpsi + srphi*crtheta*srpsi
        G_1_2 = srtheta*srpsi - srphi*crtheta*crpsi
        G_2_2 = crphi*crtheta

        G = np.array([[-G_0_0, G_0_1, -G_0_2],
                      [-G_1_0, G_1_1, -G_1_2],
                      [-G_2_0, G_2_1, -G_2_2]])

        # Invert this matrix
        G_inv = np.linalg.pinv(G)

        # Calculate the acceleration via finite difference
        if self.control_counter == 1:
            # lets avoid assuming v(t-1) is zero because it causes explosion in the accel error
            self.last_vel = cur_vel
        cur_accel = (cur_vel - self.last_vel) / control_timestep
        self.last_vel = cur_vel

        a_diff = np.zeros(3)
        a_diff[0] = sp_accel[0] - cur_accel[0]
        a_diff[1] = sp_accel[1] - cur_accel[1]
        a_diff[2] = sp_accel[2] - cur_accel[2]

        # Bound the acceleration error so that the linearization still holds
        a_diff[0] = np.clip(a_diff[0], -6.0, 6.0)
        a_diff[1] = np.clip(a_diff[1], -6.0, 6.0)
        a_diff[2] = np.clip(a_diff[2], -9.0, 9.0)

        euler_cmd = G_inv.dot(a_diff)
        thrust = euler_cmd[2]

        # Coordinated turn
        # Feedforward estimate angular rotation omega = g*tan(phi)/v
        max_phi = np.radians(30.)

        # We are dividing by the airspeed, so a lower bound is important
        airspeed_turn = np.clip(np.linalg.norm(cur_vel),10,30)

        guidance_euler_cmd = np.zeros(3)
        guidance_euler_cmd[0] = phi + euler_cmd[0]
        guidance_euler_cmd[1] = theta + euler_cmd[1]

        # Bound euler angles to prevent flipping
        guidance_euler_cmd[0] = np.clip(guidance_euler_cmd[0], -max_phi, max_phi)
        guidance_euler_cmd[1] = np.clip(guidance_euler_cmd[1], np.radians(-120), np.radians(25))


        # Use the current roll angle to determine the corresponding heading rate of change.
        coordinated_turn_roll = phi

        # Sideslip calculation - I'm using simulation knowledge for the actual value
        R_vb = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # We now correct R_vb from Pybullet frame to wind frame
        R_vb = R_vb @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        steady_state = current_wind[0:3]
        gust = current_wind[3:6]
        # convert wind vector from world to body frame and add gust
        wind_body_frame = R_vb @ steady_state + gust
        v_air_b = R_vb.T.dot(cur_vel)
        ur = v_air_b[0] - wind_body_frame[0]
        vr = v_air_b[1] - wind_body_frame[1]
        wr = v_air_b[2] - wind_body_frame[2]
        # compute airspeed
        Va = np.sqrt(ur ** 2 + vr ** 2 + wr ** 2)
        if Va == 0:
            beta = np.sign(cur_vel[1]) * np.pi / 2
        else:
            beta = np.arcsin(cur_vel[1] / Va)

        phi_cond1 = 1 if guidance_euler_cmd[0]> 0 else -1
        phi_cond2 = 1 if guidance_euler_cmd[0]< 0 else -1
        if ((guidance_euler_cmd[1] > 0.0) and (np.abs(guidance_euler_cmd[0]) < guidance_euler_cmd[1])):
            coordinated_turn_roll = np.sign(phi) * guidance_euler_cmd[1]

        if (np.abs(coordinated_turn_roll) < max_phi):
            omega = 9.81 * np.tan(coordinated_turn_roll) / airspeed_turn
        else :
            # max 60 degrees roll
            omega = 9.81 * np.tan(max_phi) * np.sign(phi) / airspeed_turn

        guidance_indi_hybrid_heading_sp = psi + (omega - K_beta * beta)/96
        guidance_indi_hybrid_heading_sp = normalize_angle(guidance_indi_hybrid_heading_sp)
        guidance_euler_cmd[2] = -guidance_indi_hybrid_heading_sp

        # now we correct the angles again
        guidance_euler_cmd[1] = guidance_euler_cmd[1] + np.radians(90)

        return thrust, guidance_euler_cmd


    def _compute_guidance_indi_run_pos(self,
                               control_timestep,
                               cur_pos,
                               cur_vel,
                               target_pos,
                               target_vel):

        """ENAC generic INDI position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        gi_speed_sp = np.zeros(3)
        pos_err = target_pos - cur_pos
        gi_speed_sp[0] = pos_err[0] * self.guidance_indi_pos_gain + target_vel[0]
        gi_speed_sp[1] = pos_err[1] * self.guidance_indi_pos_gain + target_vel[1]
        gi_speed_sp[2] = pos_err[2] * self.guidance_indi_pos_gain + target_vel[2]
        airspeed = np.linalg.norm(cur_vel)
        if airspeed>13:
            gi_speed_sp[2] = np.clip(gi_speed_sp[2],-4,4)
        return gi_speed_sp


    def _compute_accel_from_speed_sp(self,
                               control_timestep,
                               cur_quat,
                               cur_vel,
                               gi_speed_sp,
                               current_wind,
                               ):

        """ENAC generic INDI position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        guidance_indi_max_airspeed = 25
        heading_bank_gain = 6
        speed_gain =self.guidance_indi_speed_gain
        speed_gainz = self.guidance_indi_speed_gain*0.8

        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        rphi, rtheta, rpsi = cur_rpy[0], cur_rpy[1], cur_rpy[2]
        # For INDI, theta is 0 when hover and -90 in cruise
        theta = -np.radians(90) - rtheta
        psi = -rpsi
        phi = rphi
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        accel_sp = np.zeros(3)

        speed_sp_b_x =  cpsi * gi_speed_sp[0] + spsi * gi_speed_sp[1]
        speed_sp_b_y = -spsi * gi_speed_sp[0] + cpsi * gi_speed_sp[1]
        airspeed =np.linalg.norm(cur_vel)

        R_vb = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # We now correct R_vb from Pybullet frame to wind frame
        R_vb = R_vb @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        steady_state = current_wind[0:3]
        gust = current_wind[3:6]
        # convert wind vector from world to body frame and add gust
        windspeed = R_vb @ steady_state + gust
        desired_airspeed = gi_speed_sp - windspeed
        norm_des_as = np.linalg.norm(desired_airspeed)

        if airspeed>10 and norm_des_as>12 :
            #turn
            if norm_des_as > guidance_indi_max_airspeed:
                groundspeed_factor = 0.0
                if np.linalg.norm(windspeed) < guidance_indi_max_airspeed:
                    av = gi_speed_sp[0] * gi_speed_sp[0] + gi_speed_sp[1] * gi_speed_sp[1]
                    bv = -2. *(windspeed[0] * gi_speed_sp[0] + windspeed[1] * gi_speed_sp[1])
                    cv = windspeed[0] * windspeed[0] + windspeed[1] * windspeed[1] - guidance_indi_max_airspeed * guidance_indi_max_airspeed
                    dv = np.abs(bv * bv - 4.0*av * cv)
                    groundspeed_factor = (-bv + np.sqrt(dv)) / (2. * av)

                desired_airspeed[0] = groundspeed_factor * gi_speed_sp[0] - windspeed[0]
                desired_airspeed[1] = groundspeed_factor * gi_speed_sp[1] - windspeed[1]
                speed_sp_b_x = guidance_indi_max_airspeed

            # desired airspeed can not be larger than max airspeed
            speed_sp_b_x = np.minimum(norm_des_as, guidance_indi_max_airspeed)
            # calculate accel sp in body axes, because we need to regulate airspeed
            sp_accel_b = np.zeros(3)
            sp_accel_b[1] = np.arctan2(desired_airspeed[1], desired_airspeed[0]) - psi
            sp_accel_b[1] = normalize_angle(sp_accel_b[1]) * heading_bank_gain
            sp_accel_b[0] = (speed_sp_b_x - airspeed) * speed_gain

            accel_sp[0] = cpsi * sp_accel_b[0] - spsi * sp_accel_b[1]
            accel_sp[1] = spsi * sp_accel_b[0] + cpsi * sp_accel_b[1]
            accel_sp[2] = (gi_speed_sp[2] - cur_vel[2]) * speed_gainz
        else:
            # Go somewhere in the shortest way
            if airspeed > 10:
                groundspeed_x = cpsi * cur_vel[0] + spsi * cur_vel[1]
                speed_increment = speed_sp_b_x - groundspeed_x

                if ((speed_increment + airspeed) > guidance_indi_max_airspeed):
                    speed_sp_b_x = guidance_indi_max_airspeed + groundspeed_x - airspeed

            gi_speed_sp[0] = cpsi * speed_sp_b_x - spsi * speed_sp_b_y;
            gi_speed_sp[1] = spsi * speed_sp_b_x + cpsi * speed_sp_b_y;

            accel_sp[0] = (gi_speed_sp[0]- cur_vel[0]) * speed_gain
            accel_sp[1] = (gi_speed_sp[1]- cur_vel[1]) * speed_gain
            accel_sp[2] = (gi_speed_sp[2]- cur_vel[2]) * speed_gainz

            accelbound = 3.0 + airspeed / guidance_indi_max_airspeed * 5.0
            accel_sp[0] = np.clip(accel_sp[0], -accelbound, accelbound)
            accel_sp[1] = np.clip(accel_sp[1], -accelbound, accelbound)
            accel_sp[2] = np.clip(accel_sp[2], -3.0, 3.0)

        return accel_sp



    def _WayPointNavigation(self,control_timestep,cur_pos,target_pos):

        """ENAC generic wp navigation

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        CLOSE_TO_WAYPOINT = 5
        NAV_CARROT_DIST = 12
        path_to_waypoint = target_pos - cur_pos
        path_to_waypoint = np.clip(path_to_waypoint,-150,150)
        dist_to_waypoint = np.linalg.norm(path_to_waypoint)
        if (dist_to_waypoint < CLOSE_TO_WAYPOINT):
            nav_carrot = target_pos
        else:
            path_to_carrot= path_to_waypoint * NAV_CARROT_DIST/dist_to_waypoint
            nav_carrot = target_pos + path_to_carrot
        return nav_carrot

#EOF