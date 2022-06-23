import numpy as np
import sys

ofile = sys.argv[1]
files = sys.argv[3:]

data = np.loadtxt(files[0])
for i in range(1, len(files)):
    tmp = np.loadtxt(files[i])
    tmp[:,0] += data[-1,0] # num
    data = np.append(data, tmp, axis=0)

np.savetxt(ofile, data)
