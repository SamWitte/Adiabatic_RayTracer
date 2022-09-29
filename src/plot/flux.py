fname = "results/combined.npy"

import numpy as np
import matplotlib.pyplot as plt

res = np.load(fname)
event_num   = np.array(res[:, 0], dtype=int)
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


try:
    tree_weight  = res[:, 13] # Weight only from tree; currently SAME as weight
    opticalDepth = res[:, 14] # Currently NOT implemented
    weightC      = res[:, 15] # Currently NOT implemented
    kx0          = res[:, 16] # MC drawn momentum
    ky0          = res[:, 17] # MC drawn momentum
    kz0          = res[:, 18] # MC drawn momentum
    calpha       = res[:, 19]
    c            = np.abs(np.array(res[:, 20], dtype=int))
    info         = np.abs(np.array(res[:, 21], dtype=np.int64))

except:
    print("--savemode=0")
res = []

pps = weight*sln_prob # Particles per second?????

# --- Differential power ---
num_bins=50
plt.figure()
hist, bins = np.histogram(phif, bins=num_bins,
                    weights=pps*(particle_id == 1))
plt.step(bins[:-1], hist, label="photon")
hist, bins = np.histogram(phif, bins=num_bins,
                    weights=pps*(particle_id == 0))
plt.step(bins[:-1], hist, label="axion")
plt.xlabel(r"$\phi$")
plt.ylabel("Particles per second????")
plt.yscale("log")
plt.legend()

# --- Sub-branches considered ---
bins=np.arange(0, np.max(c))
plt.figure()
hist, bins = np.histogram(c, bins=bins,
                    weights=pps*(particle_id == 1))
plt.plot(bins[1:], hist, "^", label="photon")
hist, bins = np.histogram(c, bins=bins,
                    weights=pps*(particle_id == 0))
plt.plot(bins[1:], hist, "o", label="axion")
plt.xlabel(r"Number of considered sub-branches")
plt.ylabel("Particles per second????")
plt.yscale("log")
plt.axvline(10, color="k", linestyle="--", label="Monte Carlo threshold")
plt.legend()

# --- Sub-branches considered ---
flag1 = np.unique(event_num, return_index=True)[1]
flag2 = -np.unique(np.flip(event_num), return_index=True)[1]
flag = np.append(flag1,flag2)
c_tmp = c[flag]
bins=np.arange(0, np.max(c))
plt.figure()
hist, bins = np.histogram(c_tmp, bins=bins)
plt.plot(bins[1:], hist, "o")
plt.xlabel(r"Number of considered sub-branches")
plt.ylabel("Number of trees")
plt.yscale("log")
plt.axvline(10, color="k", linestyle="--", label="Monte Carlo threshold")
plt.legend()



# --- Stopping reason ---
num = event_num[-1]
print(f"Number of events:       {num}")
flag1 = np.unique(event_num, return_index=True)[1]
flag2 = -np.unique(np.flip(event_num), return_index=True)[1]
flag = np.append(flag1,flag2)
for i in range(len(info)):
    print(info[i], c[i])
print(f"Full trees:             {np.sum(np.abs(info[flag])==1)/2}")
print(f"Probability cutoff:     {np.sum(np.abs(info[flag])==2)/2}")
print(f"Num outgoing cutoff:    {np.sum(np.abs(info[flag])==3)/2}")
print(f"Max nodes cutoff:       {np.sum(np.abs(info[flag])==4)/2}")
print(f"Number MC used:         {np.sum(np.abs(info[flag])<0)/2}")

plt.show()

