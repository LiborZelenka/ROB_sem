import numpy as np
from ctu_crs import CRS97
import kinematics as k

robot = CRS97()
robot.initialize()

q0 = robot.q_home
# Get current tool pose
current_pose = k.fk(robot.get_q(), robot)

# Move the target 10cm down in the World Z-axis
target_pose = current_pose.copy()
target_pose[2, 3] -= 0.1 

# Get valid IK solutions
ik_sols = k.ik(target_pose, robot)

if len(ik_sols) > 0:
    # Find the solution physically closest to the current joint state
    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
    robot.move_to_q(closest_solution)
    robot.wait_for_motion_stop()
else:
    print("No IK solutions found!")

robot.close()