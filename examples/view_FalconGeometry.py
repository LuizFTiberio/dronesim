import pybullet as p
from time import sleep
import pybullet_data
import pdb
import numpy as np

# First let's define a class for the JointInfo.
from dataclasses import dataclass


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

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

#p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
textureId = p.loadTexture("checker_grid.jpg")

vehicleStartPos = [0, 0, 1]
vehicleStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

servo1Id = p.addUserDebugParameter("serv1", -1.5, 1.5, 0)
servo2Id = p.addUserDebugParameter("serv2", -0.3, 0.3, 0)

# boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
# boxId = p.loadURDF("cartpole.urdf", cubeStartPos, cubeStartOrientation)
# boxId = p.loadURDF("pole.urdf", cubeStartPos, cubeStartOrientation)
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
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-80, cameraPitch=-30, cameraTargetPosition=[0.0, 0.0, 0.0])

#while 1:
#    if not useRealTimeSimulation:
#        p.stepSimulation(physicsClientId=physicsClient)
#        sleep(0.01)  # Time in seconds.
#    else:
#        p.stepSimulation()


while 1:
    servo1 = p.readUserDebugParameter(servo1Id)
    servo2 = p.readUserDebugParameter(servo2Id)

    p.applyExternalForce(vehicle,
                         2,  # link number
                         forceObj=[0, 0, 0],
                         posObj=[0, 0, 0],
                         flags=p.LINK_FRAME,
                         physicsClientId=physicsClient
                         )

    # AERO MOMENTS
    p.applyExternalTorque(vehicle,
                            1,
                            torqueObj=[0, 4*servo2, 0],
                            flags=p.LINK_FRAME,
                            physicsClientId=physicsClient
                            )


    if not useRealTimeSimulation:
        p.stepSimulation(physicsClientId=physicsClient)
    sleep(0.01)  # Time in seconds.

'''
Results are:
Positive MX leads to inverse rolling moment, so MX needs to be negative in the simulation
Positive MY leads to pitch up, so it's correct
Positive MZ leads do inverse yaw moment, MZ needs to be negative as well.

M1 (joint 1) is upper left (seeing from the front) or UR in the paper convention
M2 (joint 2) is lower left (seeing from the front) or LR in the paper convention
M3 (joint 3) is upper right (seeing from the front) or UL in the paper convention
M3 (joint 4) is lower right (seeing from the front) or LL in the paper convention
'''

