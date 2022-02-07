#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, Accel
from cartesian_state_msgs.msg import PoseTwist

from scipy.spatial import distance

class AccCommand():
	def __init__(self):
		rospy.init_node('check')
		self.pub = rospy.Publisher('ur3_cartesian_velocity_controller/command_cart_vel', Twist, queue_size=10)
		self.state = PoseTwist()
		self.cmd_vel = Twist()
		self.start_pos = [-0.264, 0.242, 0.174]
		self.timestart = rospy.get_time()
		rospy.sleep(0.2)

	def run(self):
		self.state_sub = rospy.Subscriber('ur3_cartesian_velocity_controller/ee_state', PoseTwist, self.ee_state_callback)
		rospy.spin()

	def ee_state_callback(self, ee_state):
		self.state = ee_state
		if rospy.get_time() - self.timestart > 3:
			self.cmd_vel.linear.x = 0.5*(self.start_pos[0] - self.state.pose.position.x)
			self.cmd_vel.linear.y = 0.5*(self.start_pos[1] - self.state.pose.position.y)
			self.cmd_vel.linear.z = 0.5*(self.start_pos[2] - self.state.pose.position.z)
			if self.distance() < 0.002:
				self.cmd_vel.linear.x, self.cmd_vel.linear.y, self.cmd_vel.linear.z = 0, 0, 0
		else:
			self.cmd_vel.linear.x, self.cmd_vel.linear.y, self.cmd_vel.linear.z = 0, 0, 0

		self.pub.publish(self.cmd_vel)

	def distance(self):
		current_state = [self.state.pose.position.x, self.state.pose.position.y, self.state.pose.position.z]
		return distance.euclidean(self.start_pos, current_state)

if __name__ == "__main__":
	loop = AccCommand()
	loop.run()