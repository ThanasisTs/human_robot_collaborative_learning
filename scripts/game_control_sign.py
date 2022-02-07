#!/home/ttsitos/catkin_ws/src/transfer_learning_SAC/env/bin/python
import rospy
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist
from cartesian_state_msgs.msg import PoseTwist
from transfer_learning_SAC.srv import *
from transfer_learning_SAC.msg import Score
from utils import *

import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm
from pydub import AudioSegment
from pydub.playback import play
import threading

class RL_Control:
	def __init__(self):
		self.train_model = rospy.get_param('rl_control/Game/train_model', False)
		self.transfer_learning = rospy.get_param("rl_control/Game/load_model_transfer_learning", False)
		if self.train_model:
			self.load_model_for_training = rospy.get_param("rl_control/Game/load_model_training", False)
			if self.load_model_for_training:
				self.load_model_for_training_dir = rospy.get_param("rl_control/Game/load_model_training_dir", "dir")
				self.agent = get_SAC_agent(observation_space=[4], chkpt_dir=self.load_model_for_training_dir)
				self.agent.load_models()
				rospy.logwarn("Successfully loaded model at {} for training".format(self.load_model_for_training_dir))
			else:
				self.agent = get_SAC_agent(observation_space=[4])
				rospy.logwarn("User has not specified any model for training. Gonna initialize random agent")
				
			if self.transfer_learning:
				load_model_for_transfer_learning_dir = rospy.get_param("rl_control/Game/load_model_transfer_learning_dir", "dir")
				self.expert_agent = get_SAC_agent(observation_space=[4], chkpt_dir=load_model_for_transfer_learning_dir)
				self.expert_agent.load_models()
				self.ppr_threshold = rospy.get_param("rl_control/Game/ppr_threshold", 0.7)
				rospy.logwarn('Successfully loaded model at {} for transfer learning'.format(load_model_for_transfer_learning_dir))
			else:
				rospy.logwarn("User has not loaded any models for transfer learning")
		else:
			self.load_model_for_testing_dir = rospy.get_param("rl_control/Game/load_model_testing_dir", "dir")
			rospy.logwarn('User is testing the model {}'.format(self.load_model_for_testing_dir))
			self.agent = get_SAC_agent(observation_space=[4], chkpt_dir=self.load_model_for_testing_dir)
			self.agent.load_models()
			rospy.logwarn("Successfully loaded model at {} for testing".format(self.load_model_for_testing_dir))

		# Game parameters		
		self.goal = rospy.get_param('rl_control/Game/goal', [0, 0])
		self.goal_dis = rospy.get_param('rl_control/Game/goal_distance', 2)
		self.goal_vel = rospy.get_param('rl_control/Game/goal_velocity', 2)
		self.action_duration = rospy.get_param('rl_control/Experiment/action_duration', 0.1)
		audio_dir = os.path.join(rospy.get_param('rl_control/Game/full_path'), 'audio_files')
		start_audio_files = rospy.get_param('rl_control/Game/start_audio', ['', ''])
		self.start_audio = [AudioSegment.from_mp3(os.path.join(audio_dir, file)) for file in start_audio_files]
		win_audio_file = rospy.get_param('rl_control/Game/win_audio', '')
		lose_audio_file = rospy.get_param('rl_control/Game/lose_audio', '')
		self.win_audio = AudioSegment.from_mp3(os.path.join(audio_dir, win_audio_file))
		self.lose_audio = AudioSegment.from_mp3(os.path.join(audio_dir, lose_audio_file))
		self.test_count = 0
		
		# Experiment parameters for training
		self.max_episodes = rospy.get_param('rl_control/Experiment/max_episodes', 1000)
		self.max_timesteps = int(rospy.get_param('rl_control/Experiment/max_duration', 200)/self.action_duration)
		self.start_training_on_episode = rospy.get_param('rl_control/Experiment/start_training_on_episode', 10)
		self.total_update_cycles = rospy.get_param('rl_control/Experiment/total_update_cycles', 10)
		self.randomness_threshold = rospy.get_param('rl_control/Experiment/stop_random_agent', 10)
		self.scheduling = rospy.get_param('rl_control/Experiment/scheduling', 'uniform')
		self.update_cycles = self.total_update_cycles
		self.best_episode_reward = -100 - 1*self.max_timesteps
		self.win_reward = rospy.get_param('rl_control/Experiment/win_reward', 10)
		self.lose_reward = rospy.get_param('rl_control/Experiment/lose_reward', -1)	
		self.reward_history = []
		self.episode_duration = []
		self.travelled_distance = []
		self.number_of_timesteps = []
		self.column_names = ("human_action", "agent_action", "ee_pos_x_prev", "ee_pos_y_prev", "ee_vel_x_prev", "ee_vel_y_prev", "ee_pos_x_next", "ee_pos_y_next", "ee_vel_x_next", "ee_vel_y_next", "cmd_acc_human", "cmd_acc_agent")
		self.state_info = [self.column_names] 
		self.human_action = None
		self.rest_period = rospy.get_param("rl_control/Game/rest_period", 120)
		self.human_actions = []
		self.agent_actions = []
		self.ee_pos_x_prev = []
		self.ee_pos_y_prev = []
		self.ee_vel_x_prev = []
		self.ee_vel_y_prev = []
		self.ee_pos_x_next = []
		self.ee_pos_y_next = []
		self.ee_vel_x_next = []
		self.ee_vel_y_next = []
		self.cmd_acc_x = []
		self.cmd_acc_y = []

		# Experiment parameters for testing
		self.test_max_timesteps = int(rospy.get_param('rl_control/Experiment/test/max_duration', 200)/self.action_duration)
		self.test_max_episodes = rospy.get_param('rl_control/Experiment/test/max_episodes', 1000)
		self.test_interval = rospy.get_param('rl_control/Experiment/test_interval', 10)
		self.test_agent_flag = False
		self.test_best_reward = -100 -1*self.test_max_timesteps
		self.test_reward_history = []
		self.test_episode_duration = []
		self.test_travelled_distance = []
		self.test_number_of_timesteps = []
		self.test_state_info = [self.column_names] 

		self.ur3_state_sub = rospy.Subscriber('ur3_cartesian_velocity_controller/ee_state', PoseTwist, self.ee_state_callback)
		self.human_action_sub = rospy.Subscriber('cmd_vel', Twist, self.human_callback)
		self.agent_action_pub = rospy.Publisher('agent_action_topic', Float64, queue_size=10)
		self.train_pub = rospy.Publisher('train_topic', Bool, queue_size=10)
		self.score_pub = rospy.Publisher('score_topic', Score, queue_size=10)

		self.t_win = threading.Thread(target=play, args=(self.win_audio,))
		self.t_lose = threading.Thread(target=play, args=(self.lose_audio,))

		rospy.sleep(1)

	def reset(self):
		self.timestep = 0
		self.timeout = False
		self.episode_reward = 0
		self.human_actions = []
		self.agent_actions = []
		self.ee_pos_x_prev = []
		self.ee_pos_y_prev = []
		self.ee_vel_x_prev = []
		self.ee_vel_y_prev = []
		self.ee_pos_x_next = []
		self.ee_pos_y_next = []
		self.ee_vel_x_next = []
		self.ee_vel_y_next = []
		self.cmd_acc_x = []
		self.cmd_acc_y = []
		if self.transfer_learning and not self.test_agent_flag:
			self.ppr_threshold -= 0.01
		rospy.wait_for_service('reset')
		try:
			reset_game = rospy.ServiceProxy('reset', Reset)
			rospy.loginfo('Resetting the game')
			reset_game()
			rospy.loginfo('Game reset. Start episode')
		except rospy.ServiceException as e:
			rospy.logerr("Service call failed: %s"%e)


	def run(self, i_episode=0):
		rospy.loginfo('Episode: {}'.format(i_episode))
		start_time = rospy.Time.now().to_sec()
		count = 0
		while rospy.Time.now().to_sec() - start_time <= 4:
			play(self.start_audio[0]) if count < 3 else play(self.start_audio[1])
			count += 1
			rospy.sleep(0.5)
		tmp_time = 0
		total_travelled_distance = 0
		while not rospy.is_shutdown():
			self.timestep += 1
			
			if not self.test_agent_flag and self.timestep == self.max_timesteps:
				self.timeout = True
			
			if self.test_agent_flag and self.timestep == self.test_max_timesteps:
				self.timeout = True
			
			self.observation = self.get_state()
			if rospy.get_time() - tmp_time > self.action_duration:
				tmp_time = rospy.get_time()
				randomness_request = i_episode if not self.test_agent_flag else self.test_count
				self.compute_agent_action(randomness_request)
			
			rospy.sleep(self.action_duration)
			
			self.observation_ = self.get_state()
			self.reward = self.compute_reward()
			self.episode_reward += self.reward
			self.done = self.check_if_game_ended()

			if self.human_action is None:
				self.human_action = 0
			cmd_acc_human = self.human_action / 5.0
			if cmd_acc_human == 0.4:
				cmd_acc_human = -0.2
			cmd_acc_agent = self.agent_action / 5.0
			if cmd_acc_agent == 0.4:
				cmd_acc_agent = -0.2
			self.human_actions.append(self.human_action)
			self.agent_actions.append(self.agent_action)
			self.ee_pos_x_prev.append(self.observation[0])
			self.ee_pos_y_prev.append(self.observation[1])
			self.ee_vel_x_prev.append(self.observation[2])
			self.ee_vel_y_prev.append(self.observation[3])
			self.ee_pos_x_next.append(self.observation_[0])
			self.ee_pos_y_next.append(self.observation_[1])
			self.ee_vel_x_next.append(self.observation_[2])
			self.ee_vel_y_next.append(self.observation_[3])
			self.cmd_acc_x.append(cmd_acc_human)
			self.cmd_acc_y.append(cmd_acc_agent)
			if self.timestep == 1:
				self.start_time = rospy.get_time()
			
			if not self.test_agent_flag:
				self.save_experience([self.observation, self.agent_action, self.reward, self.observation_, self.done])
			
			total_travelled_distance += distance.euclidean(self.observation_[:2], self.observation[:2]) 
			
			if self.done:
				self.end_time = rospy.get_time()
				break
		
		new_state_info = list(zip(self.human_actions, self.agent_actions, self.ee_pos_x_prev, self.ee_pos_y_prev, self.ee_vel_x_prev, self.ee_vel_y_prev, self.ee_pos_x_next, self.ee_pos_y_next, self.ee_vel_x_next, self.ee_vel_y_next, self.cmd_acc_x, self.cmd_acc_y))
		
		if self.test_agent_flag:
			self.test_reward_history.append(self.episode_reward)
			self.test_episode_duration.append(self.end_time - self.start_time)
			self.test_travelled_distance.append(total_travelled_distance)
			self.test_number_of_timesteps.append(self.timestep)
			self.test_state_info.extend(new_state_info) 
			self.test_state_info.append((0,)*len(self.test_state_info[0]))
			if self.test_best_reward < self.episode_reward:
				self.test_best_reward = self.episode_reward
		else:
			self.reward_history.append(self.episode_reward)
			self.episode_duration.append(self.end_time - self.start_time)
			self.travelled_distance.append(total_travelled_distance)
			self.number_of_timesteps.append(self.timestep)
			self.state_info.extend(new_state_info)  
			self.state_info.append((0,)*len(self.state_info[0]))
			if self.best_episode_reward < self.episode_reward:
				self.best_episode_reward = self.episode_reward

	def e_greedy(self, randomness_request):
		if randomness_request < self.randomness_threshold:
			# Pure exploration
			self.agent_action = np.random.randint(self.agent.n_actions)
		else:
			# Explore with actions_prob
			self.agent_action = self.agent.actor.sample_act(self.observation)
		if self.test_agent_flag:
			self.save_models = False
		else:
			self.save_models = (randomness_request >= self.randomness_threshold)

	def compute_agent_action(self, randomness_request=None):
		assert randomness_request != None, 'randomness_request is None'
		if self.test_agent_flag:
			if self.train_model:
				rospy.loginfo("Testing with random agent") if randomness_request < self.randomness_threshold else rospy.loginfo("Testing with trained agent")
				self.e_greedy(randomness_request)
			else:
				rospy.loginfo("Testing with trained agent")
				self.agent_action = self.agent.actor.sample_act(self.observation)
				self.save_models = True
		else:
			if self.transfer_learning: 
				rospy.loginfo("Training with TL")
				self.ppr_request = np.random.randint(100)/100
				if self.ppr_request < self.ppr_threshold:
					print('Expert action')
					self.agent_action = self.expert_agent.actor.sample_act(self.observation)
				else:
					print('e greedy')
					self.e_greedy(randomness_request)
				self.save_models = True
			else:
				if not self.load_model_for_training:
					rospy.loginfo("Training from scratch")
					rospy.loginfo("Training with random agent") if randomness_request < self.randomness_threshold else rospy.loginfo('Training with trained agent')
					self.e_greedy(randomness_request)
				else:
					rospy.loginfo("Training existing model")
					self.agent_action = self.agent.actor.sample_act(self.observation)
					self.save_models = True
		agent_action_msg = Float64()
		agent_action_msg.data = self.agent_action
		self.agent_action_pub.publish(agent_action_msg)

	def compute_reward(self):
		if (distance.euclidean([self.ur3_state.pose.position.x, self.ur3_state.pose.position.y], self.goal) <= self.goal_dis and 
			distance.euclidean([self.ur3_state.twist.linear.x, self.ur3_state.twist.linear.y], [0, 0]) <= self.goal_vel):
			return self.win_reward
		return self.lose_reward

	def check_if_game_ended(self):
		if (distance.euclidean([self.ur3_state.pose.position.x, self.ur3_state.pose.position.y], self.goal) <= self.goal_dis and 
			distance.euclidean([self.ur3_state.twist.linear.x, self.ur3_state.twist.linear.y], [0, 0]) <= self.goal_vel) or self.timeout:
			score_msg = Score()
			score_msg.score.data = self.episode_reward
			if self.timeout:
				score_msg.outcome.data = False
				rospy.loginfo('Episode ended with timeout')
				t_lose = threading.Thread(target=play, args=(self.lose_audio,))
				t_lose.start()
			else:
				score_msg.outcome.data = True
				rospy.loginfo('Episode ended with goal reached')
				t_win = threading.Thread(target=play, args=(self.win_audio,))
				t_win.start()
			self.score_pub.publish(score_msg)
			return True
		return False
	
	def ee_state_callback(self, msg):
		self.ur3_state = msg

	def get_state(self):
		return np.array([self.ur3_state.pose.position.x, self.ur3_state.pose.position.y, self.ur3_state.twist.linear.x, self.ur3_state.twist.linear.y])

	def save_experience(self, interaction):
		self.agent.memory.add(*interaction)

	def human_callback(self, msg):
		self.human_action = msg.linear.x / 5

	def grad_updates(self):
		update_cycles = int(self.update_cycles)
		start_grad_updates = rospy.get_time()
		rospy.loginfo('Performing {} updates'.format(update_cycles))
		for _ in tqdm(range(update_cycles)):
			self.agent.learn()
			self.agent.soft_update_target()
		end_grad_updates = rospy.get_time()
		return end_grad_updates - start_grad_updates

	def test(self):
		self.test_agent_flag = True
		rospy.loginfo("Begin testing")
		for test_i_episode in range(1, self.test_max_episodes+1):
			self.test_count += 1
			self.reset()
			self.run(test_i_episode)
		self.test_agent_flag = False

	def train(self, i_episode):
		if self.train_model and i_episode >= self.start_training_on_episode:
			if i_episode % self.agent.update_interval == 0:
				start_training_time = rospy.get_time()
				train_msg = Bool()
				train_msg.data = True
				self.train_pub.publish(train_msg)
				self.compute_update_cycles()
				if self.update_cycles > 0:
					grad_updates_duration = self.grad_updates()
					self.agent.save_models()
				remaining_wait_time = self.rest_period - (rospy.get_time() - start_training_time)
				start_remaining_time = rospy.get_time()
				while rospy.get_time() - start_remaining_time < remaining_wait_time:
					pass
				train_msg = Bool()
				train_msg.data = False
				self.train_pub.publish(train_msg)

	def compute_update_cycles(self):
		if self.scheduling == 'uniform':
			self.update_cycles = math.ceil(self.total_update_cycles / math.ceil(self.max_episodes / self.agent.update_interval))
		elif self.scheduling == 'descending':
			self.update_cycles /= 2
		else:
			raise Exception("Choose a valid scheduling procedure")

def game_loop(game):
	if game.train_model:
		rospy.loginfo('Training')
		game.test()
		for i_episode in range(1, game.max_episodes+1):
			game.reset()
			game.run(i_episode)
			game.train(i_episode)
			if i_episode % game.test_interval == 0:
				game.test()
	else:
		rospy.loginfo('Testing')
		game.test()

if __name__ == "__main__":

	load_model_for_training = rospy.get_param("rl_control/Game/load_model_training", False)
	_, data_dir, plot_dir = get_save_dir(load_model_for_training)
	game = RL_Control()
	
	start_experiment_time = rospy.get_time()
	game_loop(game)
	end_experiment_time = rospy.get_time()
	game.reset()
	
	save_data(game, data_dir)
	plot_statistics(game, plot_dir)
	
	rospy.loginfo("Experiment Ended!!")
	rospy.loginfo("Total experiment duration: {} secs".format(end_experiment_time - start_experiment_time))