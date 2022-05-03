from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

cutoff = 0

def load_tree(filename):
    tree = []
    with open(filename) as f:
        line = f.readline()
        i = -1
        while line:
            tree.append({}); i += 1
            tree[i]["species"], w, wp = line.strip().split()
            tree[i]["weight"] = float(w)
            tree[i]["parent_weight"] = float(wp)
            lc = f.readline()
            if lc[0] == "-": # No splitting
                tree[i]["crossings"] = []
            else:
                tree[i]["crossings"] = [float(n) for n in lc.strip().split()]
            tree[i]["x"] = [float(n) for n in f.readline().strip().split()]
            tree[i]["y"] = [float(n) for n in f.readline().strip().split()]
            tree[i]["z"] = [float(n) for n in f.readline().strip().split()]
            line = f.readline()
    return tree

tree = load_tree("results/forward_5")
tree_b = load_tree("results/backward_5")


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(tree[0]["x"][0:1], tree[0]["y"][0:1], tree[0]["z"][0:1],
        marker="o", color="k")

NNN = 100

# Forwards in time
c = "C0"
for i in tree:
    if i["weight"] < cutoff: continue
    ls = "--" if i["species"][0]=="a" else "-"
    lc = int(0 if len(i["crossings"]) == 0 else i["crossings"][-1])
    N = min(lc + NNN, len(i["x"]))
    ax.plot3D(i["x"][:N], i["y"][:N], i["z"][:N], linestyle=ls, color=c)

# Backwards in time
c = "C1"
for i in tree_b:
    if i["weight"] < cutoff:
        ax.plot3D(i["x"][0:1], i["y"][0:1], i["z"][0:1], linestyle=ls, color=c,
                marker="*")
        continue
    alpha = np.log10(i["weight"])
    ls = ":" if i["species"][0]=="a" else "-."
    lc = int(0 if len(i["crossings"]) == 0 else i["crossings"][-1])
    N = min(lc + NNN, len(i["x"]))
    ax.plot3D(i["x"][:N], i["y"][:N], i["z"][:N], linestyle=ls, color=c)

#ax.set_xlim(tree[0]["x"][0]*0.9, tree[0]["x"][0]*1.1)
#ax.set_ylim(tree[0]["y"][0]*0.9, tree[0]["y"][0]*1.1)
#ax.set_zlim(tree[0]["z"][0]*0.9, tree[0]["z"][0]*1.1)

plt.show()
