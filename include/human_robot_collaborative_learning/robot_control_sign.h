#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Accel.h>
#include <cartesian_state_msgs/PoseTwist.h>
#include <human_robot_collaborative_learning/Reset.h>

#include <vector>
#include <math.h>
#include <memory>

class AccCommand{
public:
	ros::NodeHandle n;
	boost::shared_ptr<ros::AsyncSpinner> spinner;
	ros::Publisher pub;
	ros::ServiceServer service;
	ros::Subscriber human_sub, agent_sub, state_sub, reset_sub, train_sub;
	float tmp_time, min_vel, max_vel, min_x, max_x, min_y, max_y;
	double timestart;
	bool cmd_x, cmd_y, reset, train, stop_x, stop_y;
	int count_x, count_y;
	float last_time;
	std::vector<float> start_pos, start_position_0, start_position_1, start_position_2, start_position_3;
	std::vector<std::vector<float>> start_positions;
	boost::shared_ptr<cartesian_state_msgs::PoseTwist const> state;
	geometry_msgs::Twist::Ptr cmd_vel;	
	geometry_msgs::Twist::Ptr zero_vel;	
	geometry_msgs::Accel::Ptr cmd_acc;	

	AccCommand();

	bool reset_game(human_robot_collaborative_learning::Reset::Request& req, human_robot_collaborative_learning::Reset::Response& res);

	void ee_state_callback(const cartesian_state_msgs::PoseTwist::ConstPtr &ee_state);

	void agent_callback(const std_msgs::Float64::Ptr &msg);

	void human_callback(const geometry_msgs::Twist::ConstPtr &msg);

	void train_callback(const std_msgs::Bool::ConstPtr &msg);

	void run();

};

static float distance(const cartesian_state_msgs::PoseTwist::ConstPtr& state, const std::vector<float>& pos){
	return sqrt(pow(state->pose.position.x-pos[0], 2) + pow(state->pose.position.y-pos[1], 2) + pow(state->pose.position.z-pos[2], 2));
}