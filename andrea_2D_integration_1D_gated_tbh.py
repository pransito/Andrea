

	
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


    