import numpy as np
from config import config

# tool offset as SE3 matrix
T_es = np.eye(4)
T_es[:3, 3] = [-0.138, 0.0, 0.0075] 
T_es[:3, :3] = np.diag([1, 1, 1])
T_se = np.linalg.inv(T_es)

def fk(q: np.ndarray, robot):
    robot_fk = robot.fk(q) 
    return robot_fk @ T_es

def q_valid(q: np.ndarray, robot):
    limit_min = robot.q_min 
    limit_max = robot.q_max
    within_limits = np.all((q >= limit_min) & (q <= limit_max))
    
    pose = robot.fk(q)
    z_height = pose[2, 3]
    
    return within_limits and z_height > 0.03 # safety height

def ik(target_pose: np.ndarray, robot):
    ik_solutions = []
    num_steps = 32
    thetas = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)
    for theta in thetas:
        # create rotation matrix around Z
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])
        # apply rotation and tool offset
        T_rotated = target_pose @ R_z
        T_flange = T_rotated @ T_se
        
        sols = robot.ik(T_flange)
        
        for s in sols:
            if q_valid(s, robot):
                ik_solutions.append(s)

    if len(ik_solutions) == 0:
        print("IK ERROR: No valid solution found for pose.")
        return []

    # Return unique solutions
    return np.unique(np.array(ik_solutions), axis=0)

import numpy as np

def get_rotation_from_z(z_vector):
    """
    Creates a 3x3 rotation matrix where the Z-axis points in the
    direction of 'z_vector'. X and Y are generated automatically.
    """

    z = np.array(z_vector, dtype=float)
    norm = np.linalg.norm(z)
    if norm < 1e-6:
        raise ValueError("Z vector cannot be zero.")
    z /= norm

    aux = np.array([0, 1, 0], dtype=float)
    if np.abs(np.dot(aux, z)) > 0.99:
        aux = np.array([1, 0, 0], dtype=float)

    x = np.cross(aux, z)
    x /= np.linalg.norm(x)

    y = np.cross(z, x)

    R = np.eye(3)
    R[:, 0] = x
    R[:, 1] = y
    R[:, 2] = z
    
    return R

import numpy as np

def calculate_trajectory(robot, world_points, initial_z_vector, rotation_mask=None, preferred_q=None):

    
    if rotation_mask is None:
        rotation_mask = [False] * len(world_points)
        
    if preferred_q is None:
        preferred_q = robot.get_q()

    # bias for distance to preferred pose (most likely soft home)
    BIAS_WEIGHT = 0.05 

    layers = []
    
    print("Generating IK layers...")
    
    for i, point in enumerate(world_points):
        # check for rotation
        should_rotate = rotation_mask[i]
        if should_rotate and i > 0:
            z_vec = point - world_points[i-1]
            if np.linalg.norm(z_vec) < 1e-6:
                z_vec = initial_z_vector
        else:
            if i == 0:
                z_vec = initial_z_vector
            else:
                z_vec = np.array([0, 0, -1]) # z faces upwards

        target_pose = np.eye(4)
        try:
            target_pose[:3, :3] = get_rotation_from_z(z_vec)
        except ValueError:
            target_pose[:3, :3] = get_rotation_from_z(initial_z_vector)
            
        target_pose[:3, 3] = point

        sols = ik(target_pose, robot)
        
        if len(sols) == 0:
            print(f"PLANNING ERROR: Unreachable point at step {i}: {point}")
            return None 
            
        layers.append(sols)

    n_steps = len(layers)
    costs = [np.zeros(len(s)) for s in layers]
    parents = [np.zeros(len(s), dtype=int) for s in layers]

    current_q = robot.get_q()
    for j, sol in enumerate(layers[0]):
        dist_cost = np.linalg.norm(sol - current_q)
        bias_cost = np.linalg.norm(sol - preferred_q)
        costs[0][j] = dist_cost + (BIAS_WEIGHT * bias_cost)

    # forward pass
    for i in range(1, n_steps):
        prev_layer = layers[i-1]
        curr_layer = layers[i]
        
        for j_curr, sol_curr in enumerate(curr_layer):
            min_cost = float('inf')
            best_parent = -1
            bias_cost = np.linalg.norm(sol_curr - preferred_q) * BIAS_WEIGHT
            
            for j_prev, sol_prev in enumerate(prev_layer):
                dist = np.linalg.norm(sol_curr - sol_prev)
                total = costs[i-1][j_prev] + dist + bias_cost
                
                if total < min_cost:
                    min_cost = total
                    best_parent = j_prev
            
            costs[i][j_curr] = min_cost
            parents[i][j_curr] = best_parent

    # backtrack
    best_path = []
    curr_idx = np.argmin(costs[-1])
    
    for i in range(n_steps - 1, -1, -1):
        sol = layers[i][curr_idx]
        best_path.append(sol)
        curr_idx = parents[i][curr_idx]
        
    return best_path[::-1]