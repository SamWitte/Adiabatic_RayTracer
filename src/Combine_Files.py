from numpy import load, append, save, cos
import sys
from glob import glob
from os.path import basename

print(f"The combined results will be stored in {sys.argv[1]}!")

data = None
nfiles = 0
for fin in sys.argv[2:]:
    for f in glob(fin):
        fname = basename(f)
        print(f"Adding {f}...")
        if not (fname[:5] == "tree_" and fname[-4:] == ".npy"):
            raise Exception("This script can only be used to combine npz-files "
                +f"from the Adiabatic Raytracer code. I do not recognize '{f}' "
                + "as such...")
        if data is None:
            data = load(f).T
        else:
            tmp = load(f).T
            tmp[0, :] += data[0, -1]
            data = append(data, tmp, axis=1)
        nfiles += 1
if nfiles == 0: raise Exception("No files given as input!")

# divide off by num files combining...
data[9, :] /= nfiles

print(f"Saving the results in {sys.argv[1]}...")
save(sys.argv[1], data.T)
