import pybullet as p
from time import sleep
import pybullet_data
import pdb
import numpy as np
from dronesim.utils.wind_simulation import WindSimulation
# First let's define a class for the JointInfo.
from dataclasses import dataclass
from dronesim.control.INDIControl import *
from dronesim.utils.utils import *
import matplotlib.pyplot as plt


@dataclass
class Joint:
    index: int
    name: str
    type: int
    gIndex: int
    uIndex: int
    flags: int
    damping: float
    friction: float
    lowerLimit: float
    upperLimit: float
    maxForce: float
    maxVelocity: float
    linkName: str
    axis: tuple
    parentFramePosition: tuple
    parentFrameOrientation: tuple
    parentIndex: int

    def __post_init__(self):
        self.name = str(self.name, 'utf-8')
        self.linkName = str(self.linkName, 'utf-8')


#################################################################################

def compute_accel_from_speed_sp(
                               cur_quat,
                               cur_vel,
                               gi_speed_sp
                               ):

        guidance_indi_max_airspeed = 25
        heading_bank_gain = 6
        speed_gain =2
        speed_gainz = speed_gain*0.8

        cur_quat = np.array([cur_quat[3], cur_quat[0], cur_quat[1], cur_quat[2]])
        cur_rpy = get_euler_from_quaternion_ZXY(cur_quat)
        rphi, rtheta, rpsi = cur_rpy[0], cur_rpy[1], cur_rpy[2]
        # For INDI, theta is 0 when hover and -90 in cruise
        theta = -np.radians(90) - rtheta
        psi = rpsi
        phi = rphi
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        accel_sp = np.zeros(3)

        speed_sp_b_x = cpsi * gi_speed_sp[0] + spsi * gi_speed_sp[1]
        speed_sp_b_y = -spsi * gi_speed_sp[0] + cpsi * gi_speed_sp[1]
        airspeed = np.linalg.norm(cur_vel)

        R_vb = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # We now correct R_vb from Pybullet frame to wind frame
        R_vb = R_vb @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        steady_state = np.zeros(3)
        gust = np.zeros(3)
        # convert wind vector from world to body frame and add gust
        windspeed = R_vb @ steady_state + gust
        desired_airspeed = gi_speed_sp - windspeed
        norm_des_as = np.linalg.norm(desired_airspeed)

        if airspeed > 10 and norm_des_as > 12:
            # turn
            if norm_des_as > guidance_indi_max_airspeed:
                groundspeed_factor = 0.0
                if np.linalg.norm(windspeed) < guidance_indi_max_airspeed:
                    av = gi_speed_sp[0] * gi_speed_sp[0] + gi_speed_sp[1] * gi_speed_sp[1]
                    bv = -2. * (windspeed[0] * gi_speed_sp[0] + windspeed[1] * gi_speed_sp[1])
                    cv = windspeed[0] * windspeed[0] + windspeed[1] * windspeed[
                        1] - guidance_indi_max_airspeed * guidance_indi_max_airspeed
                    dv = np.abs(bv * bv - 4.0 * av * cv)
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

            gi_speed_sp[0] = cpsi * speed_sp_b_x - spsi * speed_sp_b_y
            gi_speed_sp[1] = spsi * speed_sp_b_x + cpsi * speed_sp_b_y

            accel_sp[0] = (gi_speed_sp[0] - cur_vel[0]) * speed_gain
            accel_sp[1] = (gi_speed_sp[1] - cur_vel[1]) * speed_gain
            accel_sp[2] = (gi_speed_sp[2] - cur_vel[2]) * speed_gainz

            accelbound = 3.0 + airspeed / guidance_indi_max_airspeed * 5.0
            accel_sp[0] = np.clip(accel_sp[0], -accelbound, accelbound)
            accel_sp[1] = np.clip(accel_sp[1], -accelbound, accelbound)
            accel_sp[2] = np.clip(accel_sp[2], -3.0, 3.0)

        return accel_sp





physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

#p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
textureId = p.loadTexture("checker_grid.jpg")

vehicleStartPos = [0, 0, 1]
#vehicleStartOrientation = p.getQuatern ionFromEuler([0,np.radians(-90),0])
vehicleStartOrientation = p.getQuaternionFromEuler([0,0, 0])
#vehicleStartOrientation = p.getQuaternionFromEuler(np.radians([0,0, 72.11131777]))

servo1Id = p.addUserDebugParameter("serv1", -1.5, 1.5, 0)
servo2Id = p.addUserDebugParameter("serv2", -0.3, 0.3, 0)
vehicle = p.loadURDF("../dronesim/assets/Falcon.urdf", vehicleStartPos, vehicleStartOrientation)
v_Pos, v_cubeOrn = p.getBasePositionAndOrientation(vehicle)

useRealTimeSimulation = 0
# p.configureDebugVisualizer(enable=1)

# p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)

numJoints = p.getNumJoints(vehicle)
print(f'numJoints : {numJoints}')

linkstates = p.getLinkState(vehicle, 2)
print(f'numStates : {linkstates}')

# p.getAABB(                        p.getConstraintInfo(              p.getJointStatesMultiDof(         p.getPhysicsEngineParameters(
# p.getAPIVersion(                  p.getConstraintState(             p.getKeyboardEvents(              p.getQuaternionFromAxisAngle(
# p.getAxisAngleFromQuaternion(     p.getConstraintUniqueId(          p.getLinkState(                   p.getQuaternionFromEuler(
# p.getAxisDifferenceQuaternion(    p.getContactPoints(               p.getLinkStates(                  p.getQuaternionSlerp(
# p.getBasePositionAndOrientation(  p.getDebugVisualizerCamera(       p.getMatrixFromQuaternion(        p.getUserData(
# p.getBaseVelocity(                p.getDifferenceQuaternion(        p.getMeshData(                    p.getUserDataId(
# p.getBodyInfo(                    p.getDynamicsInfo(                p.getMouseEvents(                 p.getUserDataInfo(
# p.getBodyUniqueId(                p.getEulerFromQuaternion(         p.getNumBodies(                   p.getVREvents(
# p.getCameraImage(                 p.getJointInfo(                   p.getNumConstraints(              p.getVisualShapeData(
# p.getClosestPoints(               p.getJointState(                  p.getNumJoints(
# p.getCollisionShapeData(          p.getJointStateMultiDof(          p.getNumUserData(
# p.getConnectionInfo(              p.getJointStates(                 p.getOverlappingObjects(

for i in range(numJoints):
    print(p.getJointInfo(vehicle, i))

# pdb.set_trace()

# Let's analyze the Vehicle !
print(f"Vehicle unique ID: {vehicle}")
for i in range(p.getNumJoints(vehicle)):
    joint = Joint(*p.getJointInfo(vehicle, i))
    print(joint)

# p.getJointInfo(vehicle, 1)

if (useRealTimeSimulation):
    p.setRealTimeSimulation(1)  # We need 0 to be able to use the external force and moments
else:
    #p.setGravity(0, 0, -9.81, physicsClientId=physicsClient)
    p.setRealTimeSimulation(0, physicsClientId=physicsClient)
    p.setTimeStep(0.03, physicsClientId=physicsClient)

# p.resetDebugVisualizerCamera( cameraDistance=10, cameraYaw=10, cameraPitch=-20, cameraTargetPosition=[0.0, 0.0, 0.25])
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-80, cameraPitch=0, cameraTargetPosition=[0.0, 0.0, 0.0])
wind = WindSimulation(1 / 240)
current_wind = wind.update()
u = 15
v = 0
w = 0
p.changeDynamics(1, 0, linearDamping=0, angularDamping=0, lateralFriction=0,spinningFriction=0,rollingFriction=0,)
u_array = []
t = []
tt = 0
while True:
    servo1 = p.readUserDebugParameter(servo1Id)
    servo2 = p.readUserDebugParameter(servo2Id)

    p.applyExternalTorque(vehicle,
                          1,  # link number
                          torqueObj=[0, 0, servo2/1000],
                          flags=p.LINK_FRAME,
                          physicsClientId=physicsClient
                          )


    #p.applyExternalTorque(vehicle,
    #                     2,  # link number
    #                      torqueObj=[0.00, 0, 0],
    #                      flags=p.LINK_FRAME,
    #                      physicsClientId=physicsClient
    #                     )
#
    #p.applyExternalTorque(vehicle,
    #                      3,  # link number
    #                      torqueObj=[-0.001/np.cos(.35), 0, 0],
    #                      flags=p.LINK_FRAME,
    #                      physicsClientId=physicsClient
    #                      )
#
    #p.applyExternalTorque(vehicle,
    #                      4,  # link number
    #                      torqueObj=[-0.001/np.cos(.35), 0, 0],
    #                      flags=p.LINK_FRAME,
    #                      physicsClientId=physicsClient
    #                      )
#
    #p.applyExternalTorque(vehicle,
    #                      5,  # link number
    #                      torqueObj=[0.00, 0, 0],
    #                      flags=p.LINK_FRAME,
    #                      physicsClientId=physicsClient
    #                      )
#
    #p.applyExternalForce(vehicle,
    #                     3,  # link number
    #                     forceObj=[1, 0, 0],
    #                     posObj=[0, 0, 0],
    #                     flags=p.LINK_FRAME,
    #                     physicsClientId=physicsClient
    #                     )
##
    #p.applyExternalForce(vehicle,
    #                     4,  # link number
    #                     forceObj=[1, 0, 0],
    #                     posObj=[0, 0, 0],
    #                     flags=p.LINK_FRAME,
    #                     physicsClientId=physicsClient
    #                     )
    #p.applyExternalForce(vehicle,
    #                     5,  # link number
    #                     forceObj=[1, 0, 0],
    #                     posObj=[0, 0, 0],
    #                     flags=p.LINK_FRAME,
    #                     physicsClientId=physicsClient
    #                     )
#
    #p.applyExternalTorque(vehicle,
    #                       1,
    #                       torqueObj=[0,0,0],
    #                       flags=p.LINK_FRAME,
    #                       physicsClientId=physicsClient
    #                       )
    #v_Pos, v_cubeOrn = p.getBasePositionAndOrientation(vehicle)
    #R_vb = np.array(p.getMatrixFromQuaternion(v_cubeOrn)).reshape(3, 3)
##
    #v_cubeOrn = np.array([v_cubeOrn[3],v_cubeOrn[0],v_cubeOrn[1],v_cubeOrn[2]])
    #b_cur_rpy = np.array(get_euler_from_quaternion_ZXY(v_cubeOrn))
    #print(np.degrees(b_cur_rpy))
    u,v,w = p.getBaseVelocity(vehicle)[0]
    roll,q,r = p.getBaseVelocity(vehicle)[1]
    print(roll,q,r)
    #R_vb = Quaternion2Rotation(v_cubeOrn).T
    #steady_state = current_wind[0:3]
    #gust = current_wind[3:6]
    #v_air_i = np.array( p.getBaseVelocity(vehicle)[0])
    #v_air_b = R_vb.T.dot(v_air_i)
    #wind_body_frame = R_vb @ steady_state + gust
    #ur = v_air_b[0] - wind_body_frame[0]
    #vr = v_air_b[1] - wind_body_frame[1]
    #wr = v_air_b[2] - wind_body_frame[2]
    ## compute airspeed
    #Va = np.sqrt(ur ** 2 + vr ** 2 + wr ** 2)[0]
    ## compute angle of attack
    #if ur == 0:
    #    alpha = np.sign(wr) * np.pi / 2
    #else:
    #    alpha = np.arctan(wr / ur)
    ## compute sideslip angle
    #if Va == 0:
    #    beta = np.sign(vr) * np.pi / 2
    #else:
    #    beta = np.arcsin(vr / np.sqrt(ur ** 2 + vr ** 2 + wr ** 2))
#
    #alpha = alpha[0]
    #beta = beta[0]
    ##print(np.degrees(b_cur_rpy),np.degrees(alpha),np.degrees(beta),u,v,w,v_air_b)
    #print(u,v,w)
    #u_array.append(u)
    #t.append(tt)
    #tt += 0.01


    if not useRealTimeSimulation:
        p.stepSimulation(physicsClientId=physicsClient)
    sleep(0.01)  # Time in seconds.

#plt.plot(t,u_array)
#plt.show()
#
#us = np.array(u_array)
#ts = np.array(t)
#accel = 4*np.cos(.35)/0.728
#accel_s = us / ts
#print(accel,accel_s)

