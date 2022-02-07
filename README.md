# Human-Agent Co-Learning on a shared task

## Description
A human-agent collaborative game in which the team needs to co-operate in order to control the position of the end-effector(EE) of a robotic manipulator in a 2D plane, as shown in the following figure. Collaborative learning is achieved through Deep Reinforcement Learning (DRL).

![setup](https://github.com/ThanasisTs/human_robot_collaborative_learning/blob/main/pictures/setup.png)

The human controls the position of the EE in one axis (x-axis) while the DRL agent controls the positiion in a perpendicular axis (y-axis). Therefore, the EE is able to move in a 2D plane (xy-plane, plane of the table). The EE is confined to move inside a rectangle (xy-plane) with predifined boundaries and at a specified height (z-axis). 

![court](https://github.com/ThanasisTs/human_robot_collaborative_learning/blob/main/pictures/court.png)

In the beginning of a game, the EE is automatically placed in one of the four starting positions denoted with the symbol "S". The task is solved if the team manages to bring the EE inside the goal space with a maximum allowed velocity during a specific time duration. The goal position is denoted with the symbol "X" and the goal space is the denoted with the circle around it. Once the EE is placed at the initial position, a certain sound is produced indicating the start of the game. Furthermore, two different sounds are produced at the end of the game depending on the outcome. Lastly, the outcome of the game, the score of the team and the number of the game are visualized in a window.

<img src="https://github.com/ThanasisTs/human_robot_collaborative_learning/blob/main/pictures/visualization.png" height="500" width="400">

The entire game has been implemented in ROS using both roscpp and rospy APIs and has been tested on Ubuntu 18.04 and ROS Melodic distribution.

### Control
Both partners control the position of the EE (in their respective axis) by commanding the sign of the acceleration. Specifically, they can apply a positive acceleration, a negative acceleration and zero acceleration. The human achieves it using a keyboard.

### Reinforcement Learning
The Soft Actor-Critic (SAC) algorithm is used with modifications for discrete action space[1]. The MDP is summarized as follows:

* Observation space: The observation space is the position and velocity of the EE in the xy plane (4D vector).
* Action space: The action space is the positive, negative and zero acceleration (3D vector).
* Reward function: At each timestep, the agent receives a reward of -1 if the EE transitioned to a non-goal state and 10 if it reached the goal. 

### Transfer Learning
The game can be played using two different conditions.
* No transfer learning: The agent selects either a random action or an action derived by his policy.
* Transfer learning. Probabilistic Policy Reuse is implemented. The agent can select a random action, an action derived by his policy or an action derived by an expert, pre-trained agent.

### Baseline
A version of the game exists where the DRL agent is not used and the EE can move only to the human-controlled axis (x-axis). This way, the human can gain expertise on how to control the motion of the EE in his axis.

## Installation
* Run `sudo apt-get install ros-<ROS-DISTRO>-teleop-twist-keyboard`. This will install the ROS package for sending commands from the keyboard.
* Run `source install_dependencies/install.sh`. A python virtual environemnt will be created and the necessary libraries will be installed. Furthermore, the directory of the repo will be added to the `PYTHONPATH` environmental variable.


## Run
The game has been tested only on the real robot and not on a simulated environment. The following instructions assume that the [Universal Robot UR3 Cobot](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver) is used as a robotic manipulator along with a [Cartesian Velocity Controller Interface(CVCI)](https://github.com/Roboskel-Manipulation/manos/tree/updated_driver/manos_cartesian_control). Clone the [repo](https://github.com/Roboskel-Manipulation/manos), which contains important utilities for the robot, and install the necessary repos listed in its README.md.

* Run `roslaunch manos_bringup manos_bringup.launch robot_ip:=<robot_ip> kinematics_config:=<path_to_catkin_ws>/src/manos/manos_bringup/config/manos_calibration.yaml`. This will launch the UR3 drivers and the CVCI.
* Run `roslaunch human_robot_collaborative_learning game.launch`. This will launch the entire game. This includes the node of game loop, the node for the robot motion generation, the visualization node and the SAC implementation.
* Run `rosrun teleop_twist_keyboard teleop_twist_keyboard.py`. This launches the keyboard node.

<b>Note</b>: Be aware that the mouse cursor needs to be in the terminal window from which the `teleop_twist_keyboard` was launched so that keyboard commands can be sent to the robot.

If you want to run the baseline version, replace the `roslaunch human_robot_collaborative_learning game.launch` command with `roslaunch human_robot_collaborative_learning baseline.launch`.
## Folders
* `audio_files`: Contains the sound files.
* `games_info`: Files with data regarding the games.
* `install_dependencies`: Files for seting up the virtual environment.
* `rl_models`: The files of the trained RL models are stored here.

The rest are default folders of a ROS package.

[1] Christodoulou, Petros. "Soft actor-critic for discrete action settings." arXiv preprint arXiv:1910.07207 (2019).
