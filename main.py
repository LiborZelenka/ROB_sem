import ctu_crs
import kinematics as k
import numpy as np
import os
import cv2
import calibration
import arucoSearch
import mazes
from config import config

def main():

    # --- LOAD HOMOGRAPHY ---
    if os.path.exists('./homography_matrix.npy'):
        H = np.load('./homography_matrix.npy')
    else:
        raise FileNotFoundError("Homography matrix file 'homography_matrix.npy' not found.")

    # --- CONFIGURATION ---

    if config["robot"] == "CRS97":
        robot = ctu_crs.CRS97()
    else: 
        robot = ctu_crs.CRS93()

    robot.initialize(False) # remove False on first run to do full initialization
    robot.soft_home()
    rotate = False

    if config["task"] == "A":
        maze = mazes.maze_A
        initial_z = np.array([0, 0, -1])
        rotate = False
    elif config["task"] == "B":
        maze = mazes.maze_B
        initial_z = np.array([0, 0, -1])
        rotate = False
    elif config["task"] == "C":
        maze = mazes.maze_C
        initial_z = np.array([0, 0, -1])
        rotate = False
    elif config["task"] == "D":
        maze = mazes.maze_D
        initial_z = np.array([0, -1, 0])
        rotate = True
    elif config["task"] == "E": # not finished
        maze = mazes.maze_E
        initial_z = np.array([0, 0, -1]) # need to be changed
        rotate = True
    else:
        raise ValueError("Invalid task specified in config.")
    
    # --- EXECUTE TRAJECTORY ---
    world_points = maze.get_world_points(robot, H)
    world_points = world_points[::-1]  # reverse the order of points
    
    filtered_points = []

    for p in world_points: # filter points below safety height
        if p[2] < config["safety_height"]:
            continue
        filtered_points.append(p)

    print ("Filtered world points:")
    print (np.array(filtered_points))

    mask = []
    for p in filtered_points: # set rotation mask
        if rotate and p[2] > config["rotate_height_threshold"]:
            mask.append(True)
        else:
            mask.append(False)
    path = []

    path = k.calculate_trajectory(robot, filtered_points, initial_z, rotation_mask=mask, preferred_q=robot.q_home)
    
    if path is None:
        print("No valid trajectory found.")
        robot.soft_home()
        return
    else:
        for q in path:
            robot.move_to_q(q)

    path.reverse()

    for q in path:
        robot.move_to_q(q)

    robot.soft_home()

if __name__ == "__main__":
    main()
        






