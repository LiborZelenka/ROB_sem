import numpy as np
from ctu_crs import CRS97
import kinematics as k
import os

if os.path.exists('./homography_matrix.npy'):
    H = np.load('./homography_matrix.npy')
else:
    raise FileNotFoundError("Homography matrix file 'homography_matrix.npy' not found.")

def pixel_to_world(u, v, h_matrix):
    """
    Converts pixel (u, v) to world (x, y) using Homography matrix H.
    Formula: P_world = H * P_pixel, then divide by scale factor w.
    """
    # Create homogeneous point [u, v, 1]
    p_camera = np.array([u, v, 1.0])
    
    # Apply Homography
    p_world_homo = h_matrix @ p_camera
    
    # Normalize by the 3rd component (w) to get Cartesian coordinates
    scale = p_world_homo[2]
    x = p_world_homo[0] / scale
    y = p_world_homo[1] / scale
    
    return x, y

def main():
    # --- 2. CONFIGURATION ---
    target_pixel = (1920/2, 1200/2)
    safe_z_height = 0.05        # Safe height in meters (world Z) to move to
    
    # --- 3. CALCULATE TARGET X, Y ---
    target_x, target_y = pixel_to_world(target_pixel[0], target_pixel[1], H)
    print(f"Moving to Pixel {target_pixel} -> World X: {target_x:.4f}, Y: {target_y:.4f}")

    # --- 4. ROBOT CONTROL ---
    robot = CRS97()
    robot.initialize(False)
    robot.soft_home()
    
    # Get current configuration to find the closest solution later
    q0 = robot.get_q()
    current_pose = k.fk(q0, robot)

    # Create target pose based on current rotation (keep tool orientation same)
    target_pose = current_pose.copy()

    target_pose[:3, :3] = k.get_rotation_from_z(np.array([0, 0, -1]))
    
    # Update position (X, Y from camera, Z from safety setting)
    target_pose[0, 3] = target_x
    target_pose[1, 3] = target_y
    target_pose[2, 3] = safe_z_height 

    # Solve IK
    print("Solving Inverse Kinematics...")
    ik_sols = k.ik(target_pose, robot)

    if len(ik_sols) > 0:
        # Pick the solution closest to current configuration (safest)
        best_sol = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
        
        print(f"Solution found. Moving...")
        robot.move_to_q(best_sol)
        robot.wait_for_motion_stop()
        print("Target reached.")
    else:
        print("ERROR: No valid IK solution found for these coordinates.")

    robot.close()

if __name__ == "__main__":
    main()