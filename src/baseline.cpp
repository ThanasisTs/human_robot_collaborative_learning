#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Accel.h>
#include <cartesian_state_msgs/PoseTwist.h>
#include <human_robot_collaborative_learning/Reset.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <memory>

class AccCommand{
public:
	ros::NodeHandle n;
	ros::Publisher pub;
	ros::Subscriber human_sub, state_sub;
	float tmp_time, min_vel, max_vel, min_y, max_y, velocity_tolerance, position_tolerance;
    double timestart, start_time, end_time, pause_duration, start_pause, end_pause, max_game_duration;
	bool cmd_y, reset, stop_y, pause, play, timeout;
	int count_y, reset_count, num_of_games;
	float last_time, travelled_distance;
	std::vector<float> start_pos;
	boost::shared_ptr<cartesian_state_msgs::PoseTwist const> state;
	cartesian_state_msgs::PoseTwist prev_state;
	geometry_msgs::Twist::Ptr cmd_vel;	
	geometry_msgs::Twist::Ptr zero_vel;	
	geometry_msgs::Accel::Ptr cmd_acc;
    std::string participant_file;
    std::ofstream duration_file, stats_file;
	AccCommand();
	void ee_state_callback(const cartesian_state_msgs::PoseTwist::ConstPtr &ee_state);
	void agent_callback(const std_msgs::Float64::Ptr &msg);
	void human_callback(const geometry_msgs::Twist::ConstPtr &msg);
	void run();
};

float distance(const cartesian_state_msgs::PoseTwist::ConstPtr& state, const std::vector<float>& pos){
	return sqrt(pow(state->pose.position.x-pos[0], 2) + pow(state->pose.position.y-pos[1], 2) + pow(state->pose.position.z-pos[2], 2));
}

AccCommand::AccCommand(){
	this->pub = this->n.advertise<geometry_msgs::Twist>("ur3_cartesian_velocity_controller/command_cart_vel", 100);
	this->tmp_time = 0;
	this->state = boost::make_shared<cartesian_state_msgs::PoseTwist>();
	this->cmd_vel = boost::make_shared<geometry_msgs::Twist>();
	this->zero_vel = boost::make_shared<geometry_msgs::Twist>();
	this->cmd_acc = boost::make_shared<geometry_msgs::Accel>();
	n.param("robot_movement_generation/min_vel", this->min_vel, 0.0f);
	n.param("robot_movement_generation/max_vel", this->max_vel, 0.0f);
	n.param("robot_movement_generation/min_y", this->min_y, 0.0f);
	n.param("robot_movement_generation/max_y", this->max_y, 0.0f);
    n.param("robot_movement_generation/participant_file", this->participant_file, std::string("/home/ttsitos/catkin_ws/src/human_robot_collaborative_learning/games_info/baseline/thanasis"));
	n.param("robot_movement_generation/start_position", this->start_pos, std::vector<float>(0));
	n.param("robot_movement_generation/position_tolerance", this->position_tolerance, 0.0f);
	n.param("robot_movement_generation/velocity_tolerance", this->velocity_tolerance, 0.0f);
	n.param("robot_movement_generation/num_of_games", this->num_of_games, 0);
	this->cmd_y = false, this->reset = true;
	this->stop_y = false;
	this->count_y = 0;
    this->duration_file.open(this->participant_file + ".txt");
	std::string headers = "time_duration,travelled_distance\n";
	this->duration_file << headers;
	this->stats_file.open(this->participant_file + "_stats.txt");
	headers = "human_action,ee_pos_y,ee_vel_y,cmd_acc_human,time\n";
	this->stats_file << headers;
	this->reset_count = 0;
	this->pause = false;
	this->play = true;
	this->pause_duration = 0;
	this->travelled_distance = 0;
	this->max_game_duration = 10.0;
}

void AccCommand::ee_state_callback(const cartesian_state_msgs::PoseTwist::ConstPtr &ee_state){
	this->state = ee_state;
	if (not this->pause){
		if (this->reset){
			if (ros::Time::now().toSec() - this->timestart > 3){
				this->cmd_vel->linear.x = 0.5*(this->start_pos[0] - this->state->pose.position.x);
				this->cmd_vel->linear.y = 0.5*(this->start_pos[1] - this->state->pose.position.y);
				this->cmd_vel->linear.z = 0.5*(this->start_pos[2] - this->state->pose.position.z);
				if (distance(this->state, this->start_pos) < 0.002){
					this->cmd_vel->linear.x = 0;
					this->cmd_vel->linear.y = 0;
					this->cmd_vel->linear.z = 0;
					this->reset = false;
					this->cmd_y = false;
					this->travelled_distance = 0;
					this->stats_file << "====================\n";
					ROS_INFO("State reset. Start...");
					this->reset_count ++;
					if (reset_count == this->num_of_games + 1)
						this->play = false;
					this->start_time = ros::Time::now().toSec();
					this->pause_duration = 0;
					this->prev_state.pose.position.y = 0.0f;
					this->timeout = false;
				}
			}
			else{
				this->cmd_vel->linear.x = 0;
				this->cmd_vel->linear.y = 0;
				this->cmd_vel->linear.z = 0;
			}
		}
		else{
			if (ros::Time::now().toSec() - this->start_time >= this->max_game_duration)
				this->timeout = true;
			if (this->cmd_y){
				if (this->prev_state.pose.position.y != 0.0f)
					this->travelled_distance += abs(this->state->pose.position.y - this->prev_state.pose.position.y);
				this->prev_state = *this->state;
				this->count_y ++;
				this->cmd_vel->linear.y += 0.008*this->cmd_acc->linear.y;
				if (this->cmd_vel->linear.y < this->min_vel)
					this->cmd_vel->linear.y = this->min_vel;
				else if (this->cmd_vel->linear.y > this->max_vel)
					this->cmd_vel->linear.y = this->max_vel;
				if (this->stop_y){
					if (this->state->pose.position.y < (this->min_y + this->max_y)/float(2))
						if (this->cmd_acc->linear.y <= 0)
							this->cmd_vel->linear.y = 0;
						else
							this->stop_y = false;
					else
						if (this->cmd_acc->linear.y >= 0)
							this->cmd_vel->linear.y = 0;
						else
							this->stop_y = false;
					this->count_y = 0;
				}
				else if (this->count_y > 50 and (this->state->pose.position.y + 0.01*this->cmd_vel->linear.y > this->max_y or this->state->pose.position.y + 0.01*this->cmd_vel->linear.y < this->min_y)){
					this->cmd_vel->linear.y = 0;
					this->stop_y = true;
				}
			}
			this->cmd_vel->linear.z = 0;
			if ((abs(this->state->pose.position.y - 0.246) < this->position_tolerance and abs(this->state->twist.linear.y) < this->velocity_tolerance) or this->timeout){
				ROS_WARN(this->timeout ? "YOU LOSE" : "YOU WIN");
				this->reset = true;
				this->timestart = ros::Time::now().toSec();
				this->end_time = ros::Time::now().toSec();
				std::ostringstream time_duration, trav_dis;
				time_duration << this->end_time - this->start_time - this->pause_duration;
				trav_dis << this->travelled_distance;
				this->duration_file << time_duration.str() << "," << trav_dis.str() << "\n";
			}
		}
		this->pub.publish(*this->cmd_vel);    
	}
	else{
		this->pub.publish(*this->zero_vel);
	}
}

void AccCommand::human_callback(const geometry_msgs::Twist::ConstPtr &msg){
	if (msg->linear.x == 0.5f and msg->angular.z == 1.0f){
		this->cmd_vel->linear.y = 0;
		this->pause = not this->pause;
		this->start_pause = ros::Time::now().toSec();
	}
	else{
		if (not this->reset){
			if (this->pause){
				this->end_pause = ros::Time::now().toSec();
				this->pause_duration += this->end_pause - this->start_pause;
			}
			this->pause = false;
			this->cmd_acc->linear.y = msg->linear.x / float(5);
			this->cmd_y = true;
			std::ostringstream human_action, ee_pos_y, ee_vel_y, cmd_acc_human, time;
			human_action << msg->linear.y;
			ee_pos_y << this->state->pose.position.y;
			ee_vel_y << this->state->twist.linear.y;
			cmd_acc_human << this->cmd_acc->linear.y;
			time << ros::Time::now().toSec() - this->start_time - this->pause_duration;
			this->stats_file << human_action.str() << "," << ee_pos_y.str() << "," << ee_vel_y.str() << "," << cmd_acc_human.str() << "," << time.str() << "\n";
		}
	}
}

void AccCommand::run(){
	this->human_sub = this->n.subscribe("cmd_vel", 100, &AccCommand::human_callback, this);
	this->state_sub = this->n.subscribe("ur3_cartesian_velocity_controller/ee_state", 100, &AccCommand::ee_state_callback, this);
	while (ros::ok() and this->play)
		ros::spinOnce();
	this->duration_file.close();
	this->stats_file.close();
}

int main(int argc, char** argv){
    ros::init(argc, argv, "robot_motion_generator");
    AccCommand robot_control; 
    robot_control.run();
}