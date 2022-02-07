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
	ros::Publisher pub;
	ros::Subscriber human_sub, state_sub;
	float tmp_time, min_vel, max_vel, min_x, max_x;
	double timestart;
	bool cmd_x, reset, stop_x;
	int count_x;
	float last_time;
	std::vector<float> start_pos;
	boost::shared_ptr<cartesian_state_msgs::PoseTwist const> state;
	geometry_msgs::Twist::Ptr cmd_vel;	
	geometry_msgs::Twist::Ptr zero_vel;	
	geometry_msgs::Accel::Ptr cmd_acc;

	AccCommand();

	void ee_state_callback(const cartesian_state_msgs::PoseTwist::ConstPtr &ee_state);

	void agent_callback(const std_msgs::Float64::Ptr &msg);

	void human_callback(const geometry_msgs::Twist::ConstPtr &msg);

	void run();

};

static float distance(const cartesian_state_msgs::PoseTwist::ConstPtr& state, const std::vector<float>& pos){
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
	n.param("robot_movement_generation/min_x", this->min_x, 0.0f);
	n.param("robot_movement_generation/max_x", this->max_x, 0.0f);
    this->start_pos = {-0.349, 0.250, 0.174};
	this->cmd_x = false, this->reset = false;
	this->stop_x = false;
	this->count_x = 0;
	this->last_time = 0.0f;
}

void AccCommand::ee_state_callback(const cartesian_state_msgs::PoseTwist::ConstPtr &ee_state){
	this->state = ee_state;

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
				this->cmd_x = false;
			}
		}
		else{
			this->cmd_vel->linear.x = 0;
			this->cmd_vel->linear.y = 0;
			this->cmd_vel->linear.z = 0;
		}
	}
	else{
		if (this->cmd_x){
			this->count_x ++;
			this->cmd_vel->linear.x += 0.008*this->cmd_acc->linear.x;
			if (this->cmd_vel->linear.x < this->min_vel)
				this->cmd_vel->linear.x = this->min_vel;
			else if (this->cmd_vel->linear.x > this->max_vel)
				this->cmd_vel->linear.x = this->max_vel;
			if (this->stop_x){
				if (this->state->pose.position.x < (this->min_x + this->max_x)/float(2))
					if (this->cmd_acc->linear.x <= 0)
						this->cmd_vel->linear.x = 0;
					else
						this->stop_x = false;
				else
					if (this->cmd_acc->linear.x >= 0)
						this->cmd_vel->linear.x = 0;
					else
						this->stop_x = false;
				this->count_x = 0;
			}
			else if (this->count_x > 50 and (this->state->pose.position.x + 0.01*this->cmd_vel->linear.x > this->max_x or this->state->pose.position.x + 0.01*this->cmd_vel->linear.x < this->min_x)){
				this->cmd_vel->linear.x = 0;
				this->stop_x = true;
			}
		}
		this->cmd_vel->linear.z = 0;
        if (abs(this->state->pose.position.x - (-0.271)) < 0.01 and abs(this->state->twist.linear.x) < 0.02){
            ROS_WARN("YOU WIN");
            this->reset = true;
            this->timestart = ros::Time::now().toSec();
        }
	}
    this->pub.publish(*(this->cmd_vel));    
}

void AccCommand::human_callback(const geometry_msgs::Twist::ConstPtr &msg){
    this->cmd_acc->linear.x = msg->linear.x / float(1);
    this->cmd_x = true;
}

void AccCommand::run(){
	this->human_sub = this->n.subscribe("cmd_vel", 100, &AccCommand::human_callback, this);
	this->state_sub = this->n.subscribe("ur3_cartesian_velocity_controller/ee_state", 100, &AccCommand::ee_state_callback, this);

	ros::spin();
}