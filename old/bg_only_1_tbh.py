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
    
model = nengo.Network(seed=1)
with model:
    
    
sim = nengo.Simulator(model)
sim.run(5)