#include "human_robot_collaborative_learning/robot_control_sign.h"

int main(int argc, char** argv){
	ros::init(argc, argv, "robot_control");
	AccCommand loop;
	loop.run();
}
