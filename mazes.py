import maze
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import config


maze_A = maze.Maze()
maze_A.add_straight([0, 0, 0.2], step=config["default_step_size"])
maze_A.add_start(distance=0.03)


maze_B = maze.Maze()
maze_B.add_straight([0, 0, 0.08], step=config["default_step_size"])
maze_B.add_straight([0.07, 0, 0.15], step=config["default_step_size"])
maze_B.add_straight([0.07, 0, 0.2], step=config["default_step_size"])
maze_B.add_start(distance=0.03)

maze_C = maze.Maze()
maze_C.add_straight([0, 0, 0.05], step=config["default_step_size"])
maze_C.add_straight([-0.05, 0, 0.1], step=config["default_step_size"])
maze_C.add_straight([-0.05, -0.05, 0.15], step=config["default_step_size"])
maze_C.add_straight([-0.05, -0.05, 0.2], step=config["default_step_size"])
maze_C.add_start(distance=0.03)

maze_D = maze.Maze()
maze_D.add_straight([0, 0, 0.03], step=config["default_step_size"])
maze_D.add_straight([-0.015, 0.0, 0.045], step=config["default_step_size"])
maze_D.add_straight([-0.015, 0.0, 0.1], step=config["default_step_size"])
maze_D.add_turn('y', 90, 0.04, step_degrees=config["turn_step_degrees"])
maze_D.add_start(distance=0.1)

maze_E = maze.Maze()
maze_E.add_straight([0, 0, 0.03], step=config["default_step_size"])
maze_E.add_straight([-0.017, 0.0, 0.06], step=config["default_step_size"])
maze_E.add_straight([-0.017, -0.05, 0.11], step=config["default_step_size"])
maze_E.add_turn("x", -90, 0.05, step_degrees=config["turn_step_degrees"])
# maze_E.add_straight([0.033, 0.05, 0.16], step=config["default_step_size"])
# maze_E.add_straight([0.033, 0.1, 0.16], step=config["default_step_size"])
maze_E.add_start(distance=0.05)



if __name__ == "__main__":
    print("Maze A Points:")
    print(maze_A.get_points())
    print("\nMaze B Points:")
    print(maze_B.get_points())

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(222, projection='3d')

    # ax3 = fig.add_subplot(223, projection='3d')
    # ax4 = fig.add_subplot(224, projection='3d')

    # mAwp = np.array(maze_A.get_world_points())
    # mBwp = np.array(maze_B.get_world_points())

    # Set equal axis limits for both subplots
    max_range = 0.3
    for ax in [ax1]:
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range])

    maze_E.plot(ax1)
    ax1.set_title('Maze E')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')

    # maze_B.plot(ax2)
    # ax2.set_title('Maze B')
    # ax2.set_xlabel('X axis')
    # ax2.set_ylabel('Y axis')
    # ax2.set_zlabel('Z axis')

    # ax3.plot(mAwp[:, 0], mAwp[:, 1], mAwp[:, 2], color='r')
    # ax3.set_title('Maze A World Points')
    # ax3.set_xlabel('X axis')
    # ax3.set_ylabel('Y axis')
    # ax3.set_zlabel('Z axis')

    # ax4.plot(mBwp[:, 0], mBwp[:, 1], mBwp[:, 2], color='r')
    # ax4.set_title('Maze B World Points')
    # ax4.set_xlabel('X axis')
    # ax4.set_ylabel('Y axis')
    # ax4.set_zlabel('Z axis')

    plt.show()




