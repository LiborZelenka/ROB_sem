import numpy as np
import arucoSearch
import copy
import math
from calibration import pixel_to_world
from config import config

class Maze:
    def __init__(self, start_x=0, start_y=0, start_z=0):
        self.points = []
        self.position = np.array([start_x, start_y, start_z], dtype=float)
        self.direction = np.array([0, 0, 1], dtype=float)
        self.points.append(self.position.copy())
        self.world_points = []

    def add_straight(self, goal, step=0.001):
        goal = np.array(goal, dtype=float)
        vec = goal - self.position
        self.direction = (vec)/np.linalg.norm(vec)
        distance = np.linalg.norm(vec)
        num_steps = int(distance / step)

        for i in range(1, num_steps + 1):
            new_pos = self.position + (vec) * (i/num_steps)
            self.points.append(new_pos.copy())

        self.position = goal

    def add_turn(self, axis, angle_degrees, radius, step_degrees=1): # works for D needs to be change for E
        step_rad = np.radians(abs(step_degrees))
        target_angle_rad = np.radians(abs(angle_degrees))
        sign = np.sign(angle_degrees) # +1 for Left, -1 for Right

        if axis == 'x':   rot_axis = np.array([1, 0, 0], dtype=float)
        elif axis == 'y': rot_axis = np.array([0, 1, 0], dtype=float)
        elif axis == 'z': rot_axis = np.array([0, 0, 1], dtype=float)
        else: raise ValueError("Axis must be 'x', 'y', or 'z'")

        # calculate CoR
        left_vec = np.cross(rot_axis, self.direction)
        left_vec /= np.linalg.norm(left_vec)
        
        center = self.position + (left_vec * radius * sign)

        u = self.position - center
        v = self.direction * radius

        # generate points
        num_steps = int(target_angle_rad / step_rad)
        
        for i in range(1, num_steps + 1):
            theta = step_rad * i
            effective_angle = theta * sign
            
            new_pos = center + (u * np.cos(effective_angle)) + (v * np.sin(effective_angle))
            
            self.points.append(new_pos)
            self.position = new_pos

        final_angle = target_angle_rad * sign
        self.position = center + (u * np.cos(final_angle)) + (v * np.sin(final_angle))
        self.points.append(self.position.copy())

        new_dir = sign * (-u * np.sin(final_angle) + v * np.cos(final_angle))
        self.direction = new_dir / np.linalg.norm(new_dir)

    def add_start(self, distance=0.05, step=0.01):
        last_point = copy.deepcopy(self.points[-1])
        new_point = last_point + distance * self.direction
        self.add_straight(new_point, step=step)
    
    
    def get_points(self):
        return np.array(self.points)

    def plot(self, ax):
        points = self.get_points()
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color='blue')

    def get_world_points(self, robot, H):
        self.world_points = []
        aruco = arucoSearch.ArucoSearch()
        img = robot.grab_image()
        midpoint, x_vec, y_vec = aruco.get_center(img=img, debug=False)

        anchor_x, anchor_y = pixel_to_world(int(midpoint[0]), int(midpoint[1]), H)

        scale_factor = 10.0 
        tip_u = midpoint[0] + (x_vec[0] * scale_factor)
        tip_v = midpoint[1] + (x_vec[1] * scale_factor)
        
        tip_x, tip_y = pixel_to_world(tip_u, tip_v, H)

        dx = tip_x - anchor_x
        dy = tip_y - anchor_y
        world_angle = math.atan2(dy, dx)

        c, s = np.cos(world_angle), np.sin(world_angle)
        
        transformation_matrix = np.array([
            [c, -s, 0, anchor_x],
            [s,  c, 0, anchor_y],
            [0,  0, 1, config["pedestal_height"]], # offset from table 
            [0,  0, 0, 1]
        ])
        
        for p in self.points:
            p_homo = np.array([p[0], p[1], p[2], 1.0])
            p_world = transformation_matrix @ p_homo
            self.world_points.append(p_world[:3].tolist())

        self.world_points = np.array(self.world_points)

        return self.world_points

    