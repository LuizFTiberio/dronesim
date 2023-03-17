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
vehicleStartOrientation = p.getQuaternionFromEuler([0, np.pi/2, 0])

servo1Id = p.addUserDebugParameter("serv1", -1.5, 1.5, 0)
servo2Id = p.addUserDebugParameter("serv2", -15, 15, 0)

# boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
# boxId = p.loadURDF("cartpole.urdf", cubeStartPos, cubeStartOrientation)
# boxId = p.loadURDF("pole.urdf", cubeStartPos, cubeStartOrientation)
vehicle = p.loadURDF("hexarotor.urdf", vehicleStartPos, vehicleStartOrientation)
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

while 1:
    servo1 = p.readUserDebugParameter(servo1Id)
    servo2 = p.readUserDebugParameter(servo2Id)
    # p.setGravity(0, 0, -10)
    s = 1
    for i in range(0, 12, 2):  # range(numJoints):
        # s = -1 if i%2 ==0 else 1
        p.resetJointState(vehicle, i, s * servo1)
        s *= -1
    # p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=-80, cameraPitch=-30, cameraTargetPosition=[0.0, 0.0, 0.0])
    # p.resetBasePositionAndOrientation(drone_id,pos,p.getQuaternionFromEuler(rpy),physicsClientId=self.CLIENT)
    p.removeUserDebugItem(0)

    p.applyExternalForce(vehicle,
                         3,  # link number : I dont know which link is which !!??!! :(
                         forceObj=[0, 0, servo2],
                         posObj=[0, 0, 0],
                         flags=p.LINK_FRAME,
                         physicsClientId=physicsClient
                         )
    p.applyExternalForce(vehicle,
                         8,  # link number : I dont know which link is which !!??!! :(
                         forceObj=[0, 0, servo2],
                         posObj=[0, 0, 0],
                         flags=p.LINK_FRAME,
                         physicsClientId=physicsClient
                         )
    # for i in range(12):
    #     linkstates = p.getLinkState(vehicle, i)
    #     print(f'numStates-{i} :  {linkstates[0]}')
    if not useRealTimeSimulation:
        p.stepSimulation(physicsClientId=physicsClient)
    sleep(0.01)  # Time in seconds.
''' 
# Note to myself : Murat Bronz
Apparently the order of link definition in the frame does not count, it is the joints which is deciding the link id for the force and moment application point.
Here for the hexacopter example, 
-1 is the base
 0 is the first arm front right,
 1 is the first propeller
 ..
 ..
 8 is 5th arm
 9 is the 5th propeller

'''
# else:
#   p.stepSimulation()



