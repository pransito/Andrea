import nengo
from nengo.processes import WhiteSignal
from nengo.utils.functions import piecewise
import numpy as np

###### GAME RULES
#
#     0                  1
#
#     ##                ##
#     ##                ##
#    stay             gamble
#     0             win or lose


# Network should start with random estimatives for the gamble rewards (given 
# by ANDREA)
# Should learn if it should gamble or not (have some output 'action' and 
# 'error' at each time step to be defined!)
######

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
    
model = nengo.Network(seed=1)
with model:
    
    class EnvironmentReward():

        '''
        Here we define how the environment behave
        '''
    
        def __init__(self):
            self.reward_gamble0_win = 1
            self.reward_gamble0_lose = -1
            self.reward_gamble0_win_prob = 0.1
            self.reward_gamble0_lose_prob = 1 - self.reward_gamble0_win_prob
            self.reward_gamble1_win = 1
            self.reward_gamble1_lose = -1
            self.reward_gamble1_win_prob = 0.9
            self.reward_gamble1_lose_prob = 1 - self.reward_gamble1_win_prob

            self.current_reward =  [0, 0]
            self.choice_time = 0
            self.reward_duration = 0.01
            self.choice_interval = 0.5

        def gamble(self, t, x):
        
            #x[0] = gamble0
            #x[1] = gamble1

            if cmp(self.current_reward, [0,0]) is 0:
                
                max_index = np.argmax(x)
                if x[max_index] > 0.5:
                    # self.choice_time = t   # time the choice is made
                    if max_index == 1:
                        if np.random.random() < self.reward_gamble1_win_prob:
                            self.current_reward[1] = self.reward_gamble1_win
                        else:
                            self.current_reward[1] = self.reward_gamble1_lose
                    elif max_index == 0:
                        if np.random.random() < self.reward_gamble0_win_prob:
                            self.current_reward[0] = self.reward_gamble0_win
                        else:
                            self.current_reward[0] = self.reward_gamble0_lose
                else:
                    self.current_reward =  [0, 0]
            else:
                #if t - self.choice_time > self.choice_interval:
                #    self.current_reward = [0, 0]
                if x[0] < 0.2 and x[1] < 0.2:
                    self.current_reward = [0, 0]
        
            if cmp(self.current_reward, [0, 0]) is not 0:
                return self.current_reward
            else:
                return [0, 0]
    gamble_fun = EnvironmentReward()
    
    #### ENVIRONMENT 
    reward = nengo.Node(gamble_fun.gamble, size_in = 2, size_out = 2)
    manual = nengo.Node([0,0])
    
    #### actions / rewards
    actions = nengo.Ensemble(100, 2, radius=1.4) # which action to choose
    errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=2)
    act_times = {0:1}
    pause = 0.5
    choice = 0.2
    time = 0
    for i in range(500):
        act_times[time] = 1
        time += pause
        act_times[time] = 0
        time += choice
        
    act = nengo.Node(piecewise(act_times))
    nengo.Connection(act, actions.neurons, transform=-2*np.ones((actions.n_neurons,1)))
    nengo.Connection(actions, reward)
    nengo.Connection(reward, errors.input, transform=-1)
    nengo.Connection(manual, actions)
    
    #### input to basal ganglia
    # random_thoughts = nengo.Node(np.sin)
    state = nengo.Ensemble(100, 2, radius=2)
    #state_integrator = nengo.Ensemble(100,1,radius=2)
    norm_state = nengo.Ensemble(100, 2, radius=2)
    # nengo.Connection(random_thoughts, state)
    nengo.Connection(state, norm_state, function=normalize)

    #### bg / thalamus / errors
    bg = nengo.networks.actionselection.BasalGanglia(2)
    thal = nengo.networks.actionselection.Thalamus(2)
    conn_gamble0 = nengo.Connection(state[0], norm_state[0], \
                        function=lambda x: 0.8, learning_rule_type=nengo.PES())
    conn_gamble1 = nengo.Connection(state[1], norm_state[1], \
                        function=lambda x: 0.4, learning_rule_type=nengo.PES())
    nengo.Connection(norm_state, bg.input, function=normalize)                    
    nengo.Connection(bg.input, errors.input, transform=1)
    #nengo.Connection(bg.input,state,synapse=0.1)
    nengo.Connection(bg.output, thal.input)
    nengo.Connection(errors.ensembles[0], conn_gamble0.learning_rule)
    nengo.Connection(errors.ensembles[1], conn_gamble1.learning_rule)
    nengo.Connection(act, errors.ensembles[0].neurons, \
                        transform = -2*np.ones((errors.ensembles[0].n_neurons, 1)))
    nengo.Connection(act, errors.ensembles[1].neurons, \
                        transform = -2*np.ones((errors.ensembles[1].n_neurons, 1)))
    # nengo.Connection(reward[0], conn_stay.learning_rule, transform=-1)#, function=punish_explore)
    # nengo.Connection(reward[1], conn_gamble.learning_rule, transform=-1)#, function=punish_explore)
    # nengo.Connection(bg.output[0], errors.ensembles[0].neurons, \
    #                                                  transform = np.ones((50,1))*4)    
    # nengo.Connection(bg.output[1], errors.ensembles[1].neurons, \
    #                                                 transform = np.ones((50,1))*4)    
 

    nengo.Connection(thal.output[0], actions, transform=[[1],[0]])
    nengo.Connection(thal.output[1], actions, transform=[[0],[1]])
    

    
    #def punish_explore(reward): ########
    #    if reward[0]<=0:
    #        return -0.5
    #    else:
    #        return 0

    # White Noise 
    WhiteNoise = nengo.Node(WhiteSignal(60, high=5, rms=0.1), size_out=2)
    nengo.Connection(WhiteNoise, norm_state)
    
sim = nengo.Simulator(model)
sim.run(5)