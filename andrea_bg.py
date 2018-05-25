## ANDREA BG model ##
# TASK #
# Two-armed bandit task
# Trial by trialwr
# Agent has to learn the probability of winning of either banditwa
# additional information on gamble is given on every trial
# this information is the amount of money to be won/lost, when gambling
# (no matter which of the two bandits)
# it needs to be incorporated by subject in making the decision within trial
# this is in addition to learning across trials the probability of winning 
# of either gamble

# MODEL #
# The evaluation of possible gain and loss is done by the ANDREA model
# Litt et al. (2008)
# The action selection is done by the basal ganglia and thalamus models
# Stewart et al. (2012)
# The learning across trials is inspired by Stewart et al. (2012)

# AUTHORS #
# Bruno del Papa, Alexander Genauck, Mariah Martin-Shein, Terrence Stewart

# DATE / PLACE #
# 17 June 2016
# Nengo Summer School 2016
# Waterloo University (ON), Canada

import 	nengo
from	nengo.dists import Uniform
import 	numpy as np
from 	nengo.utils.functions import piecewise
import  matplotlib.pyplot as plt
import  random as rn
import  collections
import  andrea_bg_func as abf
from    nengo.processes import WhiteSignal
import  nengo.spa as spa

# TRAIT PARAMETERS #
_beta	= 3    # DA connectivity to amy, reaction to appetitive stimuli
_gamma	= 5    # 5HT connectivity to amy, reaction to aversive stimuli (gamma > beta)? # draft paper says sigma = 0.75
_sigma  = 0.75 # < 1, determines the extent to which rectified information is # combined together
_alpha  = 0.3  # between 0 and 1; learning rate
_mu     = 2    # influence of dlpfc cost computation
_forget = 50    # forgetfulness of ofc integrator
_inflan = 0.6  # influence of andrea on action sel.

# NEURAL PARAMETERS #
N          = 500
n          = 200
rad        = 1
brad       = 4
syn        = 0.003
tau        = 0.1
neurontype = nengo.LIF()
maxrates_N = abf.get_maxrates(N,100,100)
maxrates_n = abf.get_maxrates(n,100,100)
sim_t      = 2.0

# EXPERIMENT PARAMETERS #
n_trials    = 500
len_stim    = 0.8 # watch period do not change!
len_pause   = 0.1 
len_fix     = 0.2 # fixation do not change!
len_act_rew = 0.6

# MODEL #
model = spa.SPA(seed=1)
model.config[nengo.Ensemble].neuron_type=nengo.LIF() # nengo.LIFRate

with model:
    model_ANDREA = nengo.Network(seed=312)
    with model_ANDREA:
    	# ANDREA #
        #nengo.neurons.LIF(tau_rc=0.01, tau_ref=0.001) # this is from my code (!)
        #model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
        
        # INPUTS / OUTPUTS #
    	# Amy
        stim_amyg = nengo.Node([1])
    	
    	# OFC
        possible_stim = np.vstack((np.linspace(0.0,0.9,num=10),
    	                np.linspace(0.0,0.9,num=10)*(-1)))
        all_stim 	  = abf.input_stim(possible_stim, n_trials)            # indicate number of trials
        cur_dic       = abf.stim_time(all_stim,len_stim,len_pause,len_act_rew,len_fix) # (len_stim, len_pause,len_fix)
        # mod((length of stim),(number of features plus 1))!=0
        cur_dic = collections.OrderedDict(sorted(cur_dic.items()))
    
        stim_ofc = nengo.Node(piecewise(cur_dic))
        
        # ENSEMBLES #
        # AMYG
        amyg = nengo.Ensemble(N, 1, radius = brad, max_rates=maxrates_N,
                                neuron_type=neurontype)
        
        # OFC
        ofc = nengo.Ensemble(N, 2, max_rates=maxrates_N, 
                            radius=brad, neuron_type=neurontype)
    
        # 5-HT   
        ht5_Eminus = nengo.Ensemble(n, 1, encoders=np.ones((n,1)), 
                                intercepts=Uniform(low=0.02, high=1.0), 
                                max_rates=maxrates_n,
                                radius=brad, neuron_type=neurontype)
        ht5_ht5 = nengo.Ensemble(n, 1, max_rates=maxrates_n,
                                radius=brad, neuron_type=neurontype)
        ht5_P = nengo.Ensemble(n, 1, max_rates=maxrates_n,
                                radius=brad, neuron_type=neurontype)
                                
        # DA
        da_Eplus = nengo.Ensemble(n, 1, encoders=np.ones((n,1)), 
                                intercepts=Uniform(low=0.02, high=1.0), 
                                max_rates=maxrates_n,
                                radius=brad, neuron_type=neurontype)
        da_da = nengo.Ensemble(n, 1, max_rates=maxrates_n,
                                radius=brad, neuron_type=neurontype)
        da_P = nengo.Ensemble(n, 1, max_rates=maxrates_n,
                                radius=brad, neuron_type=neurontype)
        
        # VS
        vs = nengo.Ensemble(N, 1, max_rates=maxrates_N,
                            radius=1, neuron_type=neurontype)
        
        # ACC - behaviour
        acc = nengo.Ensemble(N, 1, max_rates=maxrates_N,
                             neuron_type=neurontype)
        
        # DLPFC - which does nothing, really; YES it does.
        dlpfc = nengo.Ensemble(N, 1, encoders=np.ones((N,1)), 
                                intercepts=Uniform(low=0.02, high=1.0),
                                max_rates=maxrates_N,
                                neuron_type=neurontype)
    							
        # CONNECTIONS #
        
        # STIM AMY
        nengo.Connection(stim_amyg, amyg[0])
        
        # AMY
        nengo.Connection(amyg, ofc[1])
        
        # STIM OFC
        nengo.Connection(stim_ofc, ofc[0])
        
        # OFC
        nengo.Connection(ofc, ht5_Eminus, function=abf.neg_product, synapse=syn)
        nengo.Connection(ofc, da_Eplus, function=abf.simple_product, synapse=syn)
        
        # HT5_P
        nengo.Connection(ht5_P, ht5_Eminus, synapse=syn)
        nengo.Connection(ht5_P, ht5_P, synapse=tau)
        
        # HT5
        nengo.Connection(ht5_ht5, vs, transform=-1, synapse=syn)
        nengo.Connection(ht5_ht5, amyg, transform=_gamma, synapse=syn)
        nengo.Connection(ht5_ht5, dlpfc)
        
        # HT5_Eminus
        nengo.Connection(ht5_Eminus, ht5_ht5, transform=_sigma, synapse=syn)
        nengo.Connection(ht5_Eminus, da_da, transform=-(1-_sigma), synapse=syn)
        
        # DA_P
        nengo.Connection(da_P, da_Eplus, transform=-1, synapse=syn)
        nengo.Connection(da_P, da_P, synapse=tau)
        
        # DA_Eplus
        nengo.Connection(da_Eplus, ht5_ht5, transform=-(1-_sigma), synapse=syn)
        nengo.Connection(da_Eplus, da_da, transform=_sigma, synapse=syn)
        
        # VS
        nengo.Connection(vs, da_P, transform=_alpha, synapse=tau)
        nengo.Connection(vs, ht5_P, transform=_alpha, synapse=tau)
        
        # DA
        nengo.Connection(da_da, vs, synapse=syn)
        nengo.Connection(da_da, amyg, transform=_beta, synapse=syn)
        
        # DLPFC
        nengo.Connection(dlpfc, acc, transform = _mu)
        
        # ACC
        nengo.Connection(acc, amyg)
        
        # Output
        ofc_out = nengo.Ensemble(N, 1, max_rates=maxrates_N,
                        radius=brad,neuron_type=neurontype)
        nengo.Connection(ofc,ofc_out[0],function=abf.simple_product,synapse=syn)
        
        # Integrator of evaluation
        ofc_int = nengo.Ensemble(N, 1, max_rates=maxrates_N,radius=brad)
        nengo.Connection(ofc_out,ofc_int,synapse=tau,transform=tau)
        nengo.Connection(ofc_int,ofc_int,synapse=tau)
        
        # control of integration
        cur_dic = abf.control_time(n_trials,len_stim,len_pause,len_act_rew)
        control = nengo.Node(piecewise(cur_dic))
        nengo.Connection(control,ofc_int.neurons,synapse=syn,
        transform=np.ones((ofc_int.n_neurons,1))*(-2))
        cur_dic = abf.control_PE(all_stim,len_stim,len_pause,len_fix,len_act_rew)
        cur_dic = collections.OrderedDict(sorted(cur_dic.items()))
        ctrl_PE_int = nengo.Node(piecewise(cur_dic))
        nengo.Connection(ctrl_PE_int,ofc_out.neurons,synapse=syn,
        transform=np.ones((ofc_out.n_neurons,1))*(-2))

	## BG MODEL AND LEARNING ACROSS TRIALS ##
    gamble_fun = abf.EnvironmentReward()
    
    # ENVIRONMENT 
    reward = nengo.Node(gamble_fun.gamble, size_in = 4, size_out = 2)
    gamble_fun.len_pause   = len_pause
    gamble_fun.len_stim    = len_stim
    gamble_fun.len_act_rew = len_act_rew

    # ACTIONS / REWARDS
    actions = nengo.Ensemble(100, 2, radius=1.4) # which action to choose
    errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=2)
    act_times = {0:1}
    time = 0
    for i in range(n_trials):
        act_times[time] = 1
        time += len_pause + len_stim
        act_times[time] = 0
        time += len_act_rew
    act = nengo.Node(piecewise(act_times))
    nengo.Connection(act, actions.neurons, transform=-2*np.ones((actions.n_neurons,1)))
    nengo.Connection(actions, reward[0:2])
    nengo.Connection(reward[0], errors.ensembles[0], transform=-1)
    nengo.Connection(reward[1], errors.ensembles[1], transform=-1)
    
    ## INPUT TO BG
    state = nengo.Ensemble(100, 2, radius=2)
    norm_state = nengo.Ensemble(100, 2, radius=2)
    nengo.Connection(state, norm_state, function=abf.normalize)

    ## BG / THALAMUS / ERRORS
    bg = nengo.networks.actionselection.BasalGanglia(2)
    thal = nengo.networks.actionselection.Thalamus(2)
    conn_gamble0 = nengo.Connection(state[0], norm_state[0], \
                        function=lambda x: 0.5, learning_rule_type=nengo.PES())
    conn_gamble1 = nengo.Connection(state[1], norm_state[1], \
                        function=lambda x: 0.5, learning_rule_type=nengo.PES())
    nengo.Connection(norm_state, bg.input, function=abf.normalize)                    
    nengo.Connection(bg.input[0], errors.ensembles[0], transform=1)
    nengo.Connection(bg.input[1], errors.ensembles[1], transform=1)
    nengo.Connection(bg.output, thal.input)
    nengo.Connection(errors.ensembles[0], conn_gamble0.learning_rule)
    nengo.Connection(errors.ensembles[1], conn_gamble1.learning_rule)
    nengo.Connection(act, errors.ensembles[0].neurons, \
                        transform = -2*np.ones((errors.ensembles[0].n_neurons, 1)))
    nengo.Connection(act, errors.ensembles[1].neurons, \
                        transform = -2*np.ones((errors.ensembles[1].n_neurons, 1)))
    nengo.Connection(thal.output[0], actions, transform=[[1],[0]])
    nengo.Connection(thal.output[1], actions, transform=[[0],[1]])

    # White Noise 
    #WhiteNoise = nengo.Node(WhiteSignal(60, high=5, rms=0.1), size_out=2)
    #nengo.Connection(WhiteNoise, norm_state)
    
    ## CONNECTING ANDREA WITH BG LEARNING ##
    # Connect
    nengo.Connection(ofc_int,norm_state[1],transform=_inflan)
    
    # Make a stimulation of rewards for gamble 2
    choice2_rewards = {0:0}
    cur_time = 0
    for ii in range(n_trials):
        choice2_rewards[cur_time] = [0,0]
        cur_time += len_pause
        choice2_rewards[cur_time] = all_stim[ii]
        cur_time += len_stim + len_act_rew
    choice2_rewards = collections.OrderedDict(sorted(choice2_rewards.items()))
    reward_2_stim = nengo.Node(piecewise(choice2_rewards))
    nengo.Connection(reward_2_stim,reward[2:4])
    
    # Make a stimulation of rewards for gamble 2
    choice2_rewards_b = {0:0}
    cur_time = 0
    for ii in range(n_trials):
        choice2_rewards_b[cur_time] = [0,0]
        cur_time += len_pause
        choice2_rewards_b[cur_time] = all_stim[ii]
        cur_time += len_stim
        choice2_rewards_b[cur_time] = [0,0]
        cur_time += len_act_rew
    choice2_rewards_b = collections.OrderedDict(sorted(choice2_rewards_b.items()))
    reward_2_stim_b   = nengo.Node(piecewise(choice2_rewards_b))
   
    print(choice2_rewards)
    
    ## PROBING AND SIMULATION ##
    # Add Probes
    # Anything that is probed will collect the data it produces over time, allowing us to analyze and visualize it later.
    stim_pb    = nengo.Probe(stim_ofc) # The raw spikes from the neuron
    #spikes     = nengo.Probe(ofc_int.neurons) # The raw spikes from the neuron
    #voltage    = nengo.Probe(ofc_int.neurons, 'voltage')  # Subthreshold soma voltage of the neuron
    filtered   = nengo.Probe(ofc_out, synapse=0.01)       # Spikes filtered by a 10ms post-synaptic filter
    filt_ofc_i = nengo.Probe(ofc_int, synapse=0.01)       # Spikes filtered by a 10ms post-synaptic filter
    con_pb     = nengo.Probe(control, synapse=0.01)
    con_PE_pb  = nengo.Probe(ctrl_PE_int, synapse=0.01)
    
    # BG probe
    filt_bg_st = nengo.Probe(state,synapse=0.01)
    filt_act   = nengo.Probe(actions,synapse=0.01)
    
    #######################################
    # SPA stuff
    #######################################
    D = 32  # the dimensionality of the vectors
    voc = spa.Vocabulary(D)

    # instructor instructions: no inhibition: go!
    def instructor_fun(t, x):
        if x[0] < 0.5:
            instructor_fun._nengo_html_ = '<h2>GO!</h2>'
        else:
            if x[1] < 0.05 and x[2] < 0.05:
                instructor_fun._nengo_html_ = '<h2>Wait for it...</h2>'
            else:
                instructor_fun._nengo_html_ = '<h2></h2>'
                
            
    Instructor = nengo.Node(instructor_fun, size_in=3)
    nengo.Connection(act, Instructor[0])
    nengo.Connection(reward_2_stim,Instructor[1:3])

    # Watch
    def Watch0_fun(t, x):
        if x[0]>0.05 or x[1]>0.05:
            Watch0_fun._nengo_html_ = \
                   '<h2> <p>Gamble A</p><p>Win: %.0f</p> \n <p> Lose: %.0f</p></h2>'\
                   %(gamble_fun.reward_gamble0_win, gamble_fun.reward_gamble0_lose)
        else:
            Watch0_fun._nengo_html_ = '<h2></h2>'
    Watch0 = nengo.Node(Watch0_fun, size_in=2)
    
    def Watch1_fun(t, x):
        if abs(x[0])>0.05 and abs(x[1])>0.05:
            Watch1_fun._nengo_html_ = \
                       '<h2> <p>Gamble B</p><p>Win: %.0f </p> \n <p> Lose: %.0f</p></h2>'\
                       %(x[0]*10, x[1]*10)
        else:
            Watch1_fun._nengo_html_ = '<h2></h2>'
                   
    Watch1 = nengo.Node(Watch1_fun, size_in=2)
    #nengo.Connection(reward_2_stim,Watch1)
    nengo.Connection(reward_2_stim_b,Watch1)
    nengo.Connection(reward_2_stim_b,Watch0)
    

    # Choice
    model.Choice = spa.State(D, vocab=voc)
    def choice_fun(x):
        if x[0] > x[1]:
            return voc.parse('Gamble_A').v
        else:
            return voc.parse('Gamble_B').v
    nengo.Connection(actions, model.Choice.input, function=choice_fun)  
    
	# Outcome
    def reward_fun(t, x):
        if -0.1 < x[0] < 0.1 and -0.1 < x[1] < 0.1:
            reward_fun._nengo_html_ = '<h2> </h2>'
        elif x[0] > 0.1:
            reward_fun._nengo_html_ = '<h2>You win %.0f goat(s)! :)</h2>'\
                                               %(x[0]*10)
        elif x[0] < -0.1:
            reward_fun._nengo_html_ = '<h2>You lose %.0f goat(s)! :(</h2>' \
                                               %(-x[0]*10)
        elif x[1] < -0.1:
            reward_fun._nengo_html_ = '<h2>You lose %.0f goat(s)! :(</h2>' \
                                               %(-x[1]*10)
        elif x[1] > 0.1:
            reward_fun._nengo_html_ = '<h2>You won %.0f goat(s)! :)</h2>' \
                                                %(x[1]*10)    
	Outcome = nengo.Node(reward_fun, size_in=2)
	nengo.Connection(reward, Outcome)											
# Run the Model
sim = nengo.Simulator(model) # Create the simulator
sim.run(sim_t) # Run it for 1 seconds

plt.subplot(3,1,1)

# Plot the Results
# Plot the decoded output of the ensemble
plt.plot(sim.trange(), sim.data[filt_ofc_i])
plt.plot(sim.trange(), sim.data[stim_pb])
plt.plot(sim.trange(), sim.data[con_pb])
plt.plot(sim.trange(), sim.data[con_PE_pb])
plt.xlim(0,sim_t)
plt.xlabel('time (s)')
plt.ylabel('electric potential')
plt.title('Postsynaptic Potentials at OFC integrator')

plt.subplot(3,1,2)
plt.plot(sim.trange(), sim.data[filtered])
plt.xlim(0,sim_t)
plt.xlabel('time (s)')
plt.ylabel('decoded value')
plt.title('Postsynaptic Potentials at OFC ANDREA output')

plt.subplot(3,1,3)
plt.plot(sim.trange(), sim.data[filt_act])
plt.xlim(0,sim_t)
plt.xlabel('time (s)')
plt.ylabel('decoded value')
plt.title('Action chosen by agent')

plt.show()
