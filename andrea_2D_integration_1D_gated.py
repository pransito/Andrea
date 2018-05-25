import 	nengo
from	nengo.dists import Uniform
import 	numpy as np
from 	nengo.utils.functions import piecewise
import  matplotlib.pyplot as plt
import  random as rn
import collections

# FUNCTIONS #
def get_maxrates(n,x,y): # do not understand this; why everywhere (?)s
    return ((np.random.rand(n)*x+y).astype(int))

def input_stim(cur_list,n_trials):
	"""Takes in a list of three same-length lists.
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

def stim_time(cur_list,stim_length,pause_length, fix_time):
	"""Takes in a list of stimuli vectors (sim to tuples).
	Wants to know how long the stimuli tuples are shown.
	And length of pause.
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
		cur_time  = cur_time + time_left
	return cur_dic
	
def control_time(n_trials,stim_length,pause_length):
	"""Makes a dictionary to set control time for integrator.
	Pause time: 10: resetting
	Value time: -10: integrating.
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
	return cur_dic
	
def control_PE(cur_list,stim_length,pause_length, fix_time):
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
		cur_time  = cur_time + time_left
	return cur_dic

# TRAIT PARAMETERS #
_beta	= 3    # DA connectivity to amy, reaction to appetitive stimuli
_gamma	= 5    # 5HT connectivity to amy, reaction to aversive stimuli (gamma > beta)? # draft paper says sigma = 0.75
_sigma  = 0.75 # < 1, determines the extent to which rectified information is # combined together
_alpha  = 0.3  # between 0 and 1; learning rate
_mu     = 2    # influence of dlpfc cost computation
_forget = 50    # forgetfulness of ofc integrator

# NEURAL PARAMETERS #
N          = 1000
n          = 400
rad        = 1
brad       = 4
syn        = 0.003
tau        = 0.1
neurontype = nengo.LIF()

# EXPERIMENT PARAMTERS #
n_trials   = 7
len_stim   = 0.8
len_pause  = 0.6
len_fix    = 0.2

# MODEL #
model = nengo.Network(seed=410)
model.config[nengo.Ensemble].neuron_type=nengo.LIF() # nengo.LIFRate

with model:
    #nengo.neurons.LIF(tau_rc=0.01, tau_ref=0.001) # this is from my code (!)
    #model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    
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
    
    # INPUTS / OUTPUTS #
	# Amy
    stim_amyg = nengo.Node([1])
	
	# OFC
    possible_stim = np.vstack((np.linspace(0.1,0.9,num=7),
	                np.linspace(0.05,0.6,num=7)*(-1)))
    all_stim 	  = input_stim(possible_stim, n_trials) # indicate number of trials
    cur_dic = stim_time(all_stim,len_stim,len_pause,len_fix)      # (len_stim, len_pause,len_fix)
    # mod((length of stim),(number of features plus 1))!=0
    cur_dic = collections.OrderedDict(sorted(cur_dic.items()))

    stim_ofc = nengo.Node(piecewise(cur_dic))
    
    # ENSEMBLES #
    # AMYG
    amyg = nengo.Ensemble(N, 1, radius = brad, max_rates=get_maxrates(N,100,100),
                            neuron_type=neurontype)
    
    # OFC
    ofc = nengo.Ensemble(N, 2, max_rates=get_maxrates(N,100,100), 
                        radius=brad, neuron_type=neurontype)

    # 5-HT   
    ht5_Eminus = nengo.Ensemble(n, 1, encoders=np.ones((n,1)), 
                            intercepts=Uniform(low=0.02, high=1.0), 
                            max_rates=get_maxrates(n,100,100),
                            radius=brad, neuron_type=neurontype)
    ht5_ht5 = nengo.Ensemble(n, 1, max_rates=get_maxrates(n,100,100),
                            radius=brad, neuron_type=neurontype)
    ht5_P = nengo.Ensemble(n, 1, max_rates=get_maxrates(n,100,100),
                            radius=brad, neuron_type=neurontype)
                            
    # DA
    da_Eplus = nengo.Ensemble(n, 1, encoders=np.ones((n,1)), 
                            intercepts=Uniform(low=0.02, high=1.0), 
                            max_rates=get_maxrates(n,100,100),
                            radius=brad, neuron_type=neurontype)
    da_da = nengo.Ensemble(n, 1, max_rates=get_maxrates(n,100,100),
                            radius=brad, neuron_type=neurontype)
    da_P = nengo.Ensemble(n, 1, max_rates=get_maxrates(n,100,100),
                            radius=brad, neuron_type=neurontype)
    
    # VS
    vs = nengo.Ensemble(N, 1, max_rates=get_maxrates(N,100,100),
                        radius=1, neuron_type=neurontype)
    
    # ACC - behaviour
    acc = nengo.Ensemble(N, 1, max_rates=get_maxrates(N,100,100),
                         neuron_type=neurontype)
    
    # DLPFC - which does nothing, really; YES it does.
    dlpfc = nengo.Ensemble(N, 1, encoders=np.ones((N,1)), 
                            intercepts=Uniform(low=0.02, high=1.0),
                            max_rates=get_maxrates(N,100,100),
                            neuron_type=neurontype)
    
    
    # CONNECTIONS #
    
    # STIM AMY
    nengo.Connection(stim_amyg, amyg[0])
    
    # AMY
    nengo.Connection(amyg, ofc[1])
    
    # STIM OFC
    nengo.Connection(stim_ofc, ofc[0])
    
    # OFC
    nengo.Connection(ofc, ht5_Eminus, function=neg_product, synapse=syn)
    nengo.Connection(ofc, da_Eplus, function=simple_product, synapse=syn)
    
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
    
    # OUTPUT #
    # Output
    ofc_out      = nengo.Ensemble(N, 1, max_rates=get_maxrates(N,100,100),
                    radius=brad,neuron_type=neurontype)
    nengo.Connection(ofc,ofc_out[0],function=simple_product,synapse=syn)
    #nengo.Connection(stim_ofc,ofc_out)
    
    # integrator of evaluation
    ofc_int = nengo.Ensemble(N, 1, max_rates=get_maxrates(N,100,100),radius=brad)
    nengo.Connection(ofc_out,ofc_int,synapse=tau,transform=tau)
    nengo.Connection(ofc_int,ofc_int,synapse=tau)
    
    # control of integration
    cur_dic = control_time(n_trials,len_stim,len_pause)
    control = nengo.Node(piecewise(cur_dic))
    nengo.Connection(control,ofc_int.neurons,synapse=syn,
    transform=np.ones((ofc_int.n_neurons,1))*(-2))
    cur_dic = control_PE(all_stim,len_stim,len_pause,len_fix)
    cur_dic = collections.OrderedDict(sorted(cur_dic.items()))
    ctrl_PE_int = nengo.Node(piecewise(cur_dic))
    nengo.Connection(ctrl_PE_int,ofc_out.neurons,synapse=syn,
    transform=np.ones((ofc_out.n_neurons,1))*(-2))
	
    # PROBING
    # Add Probes
    # Anything that is probed will collect the data it produces over time, allowing us to analyze and visualize it later.
    stim_pb    = nengo.Probe(stim_ofc) # The raw spikes from the neuron
    #spikes     = nengo.Probe(ofc_int.neurons) # The raw spikes from the neuron
    #voltage    = nengo.Probe(ofc_int.neurons, 'voltage')  # Subthreshold soma voltage of the neuron
    filtered   = nengo.Probe(ofc_out, synapse=0.01)       # Spikes filtered by a 10ms post-synaptic filter
    filt_ofc_i = nengo.Probe(ofc_int, synapse=0.01)       # Spikes filtered by a 10ms post-synaptic filter
    con_pb     = nengo.Probe(control, synapse=0.01)
    con_PE_pb  = nengo.Probe(ctrl_PE_int, synapse=0.01)
	
# Run the Model
sim = nengo.Simulator(model) # Create the simulator
sim.run(8) # Run it for ... seconds

plt.subplot(2,1,1)

# 6: Plot the Results
# Plot the decoded output of the ensemble
plt.plot(sim.trange(), sim.data[filt_ofc_i])
plt.plot(sim.trange(), sim.data[stim_pb])
plt.plot(sim.trange(), sim.data[con_pb])
plt.plot(sim.trange(), sim.data[con_PE_pb])
plt.xlim(0,8)
plt.xlabel('time (s)')
plt.ylabel('electric potential')
plt.title('Postsynaptic Potentials at OFC integrator')

plt.subplot(2,1,2)
plt.plot(sim.trange(), sim.data[filtered])
plt.xlim(0,8)
plt.xlabel('time (s)')
plt.ylabel('electric potential')
plt.title('Postsynaptic Potentials at OFC ANDREA output')

# # Plot the spiking output of the ensemble
# from nengo.utils.matplotlib import rasterplot
# plt.figure(figsize=(10, 8))
# plt.subplot(221)
# rasterplot(sim.trange(), sim.data[spikes])
# plt.ylabel("VS")
# plt.xlim(0, 15)

# # Plot the soma voltages of the neurons
# plt.subplot(222)
# plt.plot(sim.trange(), sim.data[voltage][:,0], 'r')
# plt.xlim(0, 15);
plt.show()


    