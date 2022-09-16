import numpy as np
import os

os.system("mkdir -p results/event/  results/npz/ results/tree/")

# (saveMode==0)
os.system("nice -20 julia Gen_Samples.jl --MassA=2e-5 --Axg=1e-18  --type=1 --probCutoff=1e-10 --numCutoff=5 --Nts=10 --seed=1769 --saveMode=0 --ftag=test0")
res = np.load("results/npz/tree_MassAx_2.0e-5_AxionG_1.0e-18_Ax_trajs_10_N_maxSample_6_num_cutoff_5_max_nodes_5_iseed_1769_test0_.npz")
event_num   = res[:, 0]
particle_id = np.array(res[:, 1], dtype=int)
thetaf      = res[:, 2] # Momentum theta
phif        = res[:, 3] # Momentum phi
thetafX     = res[:, 4] # Position theta
phifX       = res[:, 5]
absfX       = res[:, 6] # Final distance from NS
sln_prob    = res[:, 7] # Incoming axions per second
weight      = res[:, 8] # Weight normalised such that the weight of the
                        # incoming axion is 1
x0          = res[:, 9] # MC drawn position
y0          = res[:, 10] # MC drawn position
z0          = res[:, 11] # MC drawn position
delta_w     = res[:, 12] # Energy dispersion; currently NOT implemented

print("--saveMode=0")
print(res)

# Save more (saveMode>0)
os.system("nice -20 julia Gen_Samples.jl --MassA=2e-5 --Axg=1e-18  --type=1 --probCutoff=1e-10 --numCutoff=5 --Nts=10 --seed=1769 --saveMode=1 --ftag=test1")
res = np.load("results/npz/tree_MassAx_2.0e-5_AxionG_1.0e-18_Ax_trajs_10_N_maxSample_6_num_cutoff_5_max_nodes_5_iseed_1769_test1_.npz")
event_num   = res[:, 0]
particle_id = np.array(res[:, 1], dtype=int)
thetaf      = res[:, 2] # Momentum theta
phif        = res[:, 3] # Momentum phi
thetafX     = res[:, 4] # Position theta
phifX       = res[:, 5]
absfX       = res[:, 6] # Final distance from NS
sln_prob    = res[:, 7] # Incoming axions per second
weight      = res[:, 8] # Weight normalised such that the weight of the
                        # incoming axion is 1
x0          = res[:, 9] # MC drawn position
y0          = res[:, 10] # MC drawn position
z0          = res[:, 11] # MC drawn position
delta_w     = res[:, 12] # Energy dispersion; currently NOT implemented
tree_weight = res[:, 13] # Weight only from tree; currently SAME as weight
opticalDepth = res[:, 14] # Currently NOT implemented
weightC      = res[:, 15] # Currently NOT implemented
kx0          = res[:, 16] # MC drawn momentum
ky0          = res[:, 17] # MC drawn momentum
kz0          = res[:, 18] # MC drawn momentum

print("\n--saveMode=1")
print(res)
