## FUNCTIONS FOR ANDREA BG MODEL ##
import 	nengo
from	nengo.dists import Uniform
import 	numpy as np
from 	nengo.utils.functions import piecewise
import  matplotlib.pyplot as plt
import  random as rn
import  collections
import  andrea_bg_func as abf
from    nengo.processes import WhiteSignal

## BG Model functions

# normalize input to BG
def normalize(x):
    OldRange = x.max() - x.min()
    if x.size == 1:
        return 0.8
    if OldRange == 0:
        x_new = np.ones(x.size)*0.8
    else:
        NewRange = (0.8 - 0.5)
        x_new = (((x - x.min())*0.3)/OldRange)+0.5
    return x_new

class EnvironmentReward():

        '''
        Here we define how the environment behaves
        '''
    
	def __init__(self):
		self.reward_gamble0_win = 1
		self.reward_gamble0_lose = -1
		self.reward_gamble0_win_prob = 0.3
		self.reward_gamble0_lose_prob = 1 - self.reward_gamble0_win_prob
		self.reward_gamble1_win = 0.2
		self.reward_gamble1_lose = -0.2
		self.reward_gamble1_win_prob = 0.7
		self.reward_gamble1_lose_prob = 1 - self.reward_gamble1_win_prob

		self.current_reward  =  [0, 0]
		self.choice_time     = 0
		self.reward_duration = 0.01
		self.len_act_rew     = 0.5
		self.len_stim        = 0.8
		self.len_pause       = 0.1
		self.more_than_one   = 0
		self.wait_time       = self.len_act_rew + self.len_pause + self.len_stim
		self.cur_choice      = [0,0]

	def gamble(self, t, x):
		self.cur_choice = x[0:2]
		
		# change prob of gambling
		#if t%15 == 0:
		#	self.reward_gamble0_win_prob = 0.7
		#	self.reward_gamble0_lose_prob = 1 - self.reward_gamble0_win_prob
		#	self.reward_gamble1_win_prob = 0.3
		#	self.reward_gamble1_lose_prob = 1 - self.reward_gamble1_win_prob
			
		if self.more_than_one is 0 and t > (self.choice_time + self.wait_time - self.len_act_rew):
			print 'First Trial!'
			## determine reward
			max_index  = np.argmax(self.cur_choice)
			if self.cur_choice[max_index] > 0.5:
				print 'a choice was made'
				self.more_than_one = 1
				self.choice_time = t
				if max_index == 1:   # gamble 2
					if np.random.random() < self.reward_gamble1_win_prob:
						self.current_reward[1] = x[2]
					else:
						self.current_reward[1] = x[3]
				elif max_index == 0: # gamble 1					
					if np.random.random() < self.reward_gamble0_win_prob:
						self.current_reward[0] = self.reward_gamble0_win
					else:
						self.current_reward[0] = self.reward_gamble0_lose
				else:
					self.current_reward =  [0, 0]
			else:
				print 'no choice was made'
				if self.cur_choice[0] < 0.2 and self.cur_choice[1] < 0.2:
					self.current_reward = [0, 0]
			return self.current_reward
			##
		elif (self.more_than_one is 1) and t > (self.choice_time + self.wait_time):
			print '2nd or higher trial!'
			## determine reward
			max_index  = np.argmax(self.cur_choice)
			if self.cur_choice[max_index] > 0.5:
				print 'a choice was made'
				self.choice_time = t
				if max_index == 1:   # gamble 2
					if np.random.random() < self.reward_gamble1_win_prob:
						self.current_reward[1] = x[2]
					else:
						self.current_reward[1] = x[3]
				elif max_index == 0: # gamble 1					
					if np.random.random() < self.reward_gamble0_win_prob:
						self.current_reward[0] = self.reward_gamble0_win
					else:
						self.current_reward[0] = self.reward_gamble0_lose
				else:
					self.current_reward =  [0, 0]
			else:
				print 'no choice was made'
				if self.cur_choice[0] < 0.2 and self.cur_choice[1] < 0.2:
					self.current_reward = [0, 0]
			return self.current_reward
			##
		elif (self.more_than_one is 1) and t > (self.choice_time + self.len_act_rew):
			self.current_reward = [0,0]
			return self.current_reward
		else:
			return self.current_reward
		
	def gamble_old(self, t, x):
	
		#x[0] = gamble0
		#x[1] = gamble1
		print t
		print x
		print self.current_reward
		if cmp(self.current_reward, [0,0]) is 0:
			print 'current reward is [0,0]!'
			cur_choice = x[0:2]
			max_index  = np.argmax(cur_choice)
			if cur_choice[max_index] > 0.5:
				print 'a choice was made'
				if max_index == 1:   # gamble 2
					if np.random.random() < self.reward_gamble1_win_prob:
						self.current_reward[1] = x[2]
					else:
						self.current_reward[1] = x[3]
				elif max_index == 0: # gamble 1
					
					if np.random.random() < self.reward_gamble0_win_prob:
						self.current_reward[0] = self.reward_gamble0_win
					else:
						self.current_reward[0] = self.reward_gamble0_lose
				else:
					self.current_reward =  [0, 0]
			else:
				print 'no choice was made'
				if self.cur_choice[0] < 0.2 and self.cur_choice[1] < 0.2:
					self.current_reward = [0, 0]
		if cmp(self.current_reward,[0,0]) is not 0:
			self.current_reward = [0,0]
			return self.current_reward
		else:
			return [0,0]
# ANDREA model functions
def get_maxrates(n,x,y): # do not understand this; why everywhere (?)s
    return ((np.random.rand(n)*x+y).astype(int))

def input_stim(cur_list,n_trials):
	"""Takes in a list of n same-length lists.
	Every of those lists has as elements values.
	E.g. put in a vector of possible gains and 
	possible losses.
	Samples from those vectors with replacement,
	to produce tupels which are of same dim as cur_list."""
	output_list = []
	
	for ii in range(n_trials):
		cur_trial = []
		for jj in range(len(cur_list)):
			cur_vec       = cur_list[jj,:]
			cur_trial.append(cur_vec[rn.randrange(1,len(cur_vec), 1)])
		output_list.append(cur_trial)
	return output_list

def stim_time(cur_list,stim_length,pause_length,len_act_rew, fix_time):
	"""Takes in a list of stimuli vectors (sim to tuples).
	Wants to know how long the stimuli tuples are shown.
	And length of pause.
	And length of action__reward_period.
	And how long each fixation time is.
	After every fixations will go to zero,
	to avoid PEs dependent of difference between,
	features. Give times so that 0 period is possible,
	towards end of trial."""
    # initialize the dictionary
	cur_dic = {0:0}
		
	# for every trial create the stimulation
	n_trials  = len(cur_list)
	cur_time  = 0
	for ii in range(n_trials):
		cur_stim          = cur_list[ii]
		cur_stim_length   = stim_length             # add jitter here later
		cur_pause_length  = pause_length            # add jitter here later
		cur_dic[cur_time] = 0
		cur_time          = cur_time + pause_length # first add the pause
		cur_base_time     = 0
		while cur_base_time < cur_stim_length:		# saccades only as long as overall stim time not over
			for kk in range(len(cur_stim)):
				if cur_base_time < cur_stim_length:
					cur_dic[cur_time] = cur_stim[kk]
					cur_time          = cur_time + fix_time 		# overall time
					cur_base_time     = cur_base_time + fix_time	# within-trial time
					cur_dic[cur_time] = 0                           # always set to zero after feature fix
					cur_time          = cur_time + fix_time
					cur_base_time     = cur_base_time + fix_time
		# measure how much time left to end of trial
		time_left = cur_stim_length - cur_base_time
		cur_time  = cur_time + time_left + len_act_rew
	return cur_dic
	
def control_time(n_trials,stim_length,pause_length,len_act_rew):
	"""Makes a dictionary to set control time for integrator.
	Pause time and lenght of action and reward: 1: resetting
	Value time: 								0: integrating.
	Starts with pause."""
    # initialize the dictionary
	cur_dic  = {0:1}
	cur_time = 0
		
	# for every trial create the stimulation
	for ii in range(n_trials):
		cur_stim_length   = stim_length                 # add jitter here later
		cur_pause_length  = pause_length                # add jitter here later
		cur_dic[cur_time+0.05] = 1
		cur_time          = cur_time + cur_pause_length # first add the pause
		cur_dic[cur_time+0.05] = 0
		cur_time          = cur_time + cur_stim_length # first add the pause
		cur_time          = cur_time + len_act_rew
	return cur_dic
	
def control_PE(cur_list,stim_length,pause_length, len_act_rew, fix_time):
	"""Makes a dictionary to set control time for ignoring/
	taking into account the OFC signal.
	ignore-time: 0
	eval time:   1
	To avoid taking into account PEs when just setting to baseline."""
    # initialize the dictionary
	cur_dic = {0:1}
		
	# for every trial create the stimulation
	n_trials  = len(cur_list)
	cur_time  = 0
	for ii in range(n_trials):
		cur_stim          = cur_list[ii]
		cur_stim_length   = stim_length             # add jitter here later
		cur_pause_length  = pause_length            # add jitter here later
		cur_dic[cur_time] = 0
		cur_time          = cur_time + pause_length # first add the pause
		cur_base_time     = 0
		while cur_base_time < cur_stim_length:		# saccades only as long as overall stim time not over
			for kk in range(len(cur_stim)):
				if cur_base_time < cur_stim_length:
					cur_dic[cur_time] = 0
					cur_time          = cur_time + fix_time 		# overall time
					cur_base_time     = cur_base_time + fix_time	# within-trial time
					cur_dic[cur_time] = 1                         # always set to zero after feature fix
					cur_time          = cur_time + fix_time
					cur_base_time     = cur_base_time + fix_time
		# measure how much time left to end of trial
		time_left = cur_stim_length - cur_base_time
		cur_time  = cur_time + time_left + len_act_rew
	return cur_dic

# FUNCTIONS FOR NEURONS #
def simple_product(x):
    return x[0]*x[1]
def simple_product_ctrl(x):
    return x[0]*x[1]
def simple_product_tau(x):
	return x[0]*x[1]*tau
def triple_product(x):
    return x[0]*x[1]*x[2]
def neg_triple_product(x):
    return (x[0]*x[1]*x[2])*(-1)
def neg_product(x):
    return -x[0]*x[1]
def neg_product_ctrl(x):
    return -x[0]*x[1]