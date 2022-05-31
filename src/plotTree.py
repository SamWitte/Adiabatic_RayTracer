from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

cutoff = -1


fontsize = 12
newparams = {'figure.figsize'        : (5, 4),
             'font.size'             : fontsize,
             'mathtext.fontset'      : 'stix',
             'font.family'           : 'STIXGeneral',
             'lines.markersize'      : 8,
             'lines.linewidth'       : 1.5,
             #'legend.frameon'        : False,
             #'legend.labelspacing'   : 0.1,
             #'legend.handletextpad'  : 0.2,
             #'legend.columnspacing'  : 1,
             #'legend.borderaxespad'  : 0.2,
             #'ytick.major.pad'       : 0,
             #'ytick.minor.pad'       : 0, 
             #'xtick.major.pad'       : 0,
             #'xtick.minor.pad'       : 0,
             #'axes.labelpad'         : -2.0,
             #'errorbar.capsize'      : 2,
             'markers.fillstyle'     : "none",
             'lines.markeredgewidth' : 1,
             #'lines.markeredgecolor' : "k",
             'xtick.bottom'          : True,
             'xtick.top'             : True,
             'ytick.left'            : True,
             'ytick.right'           : True,
             "ytick.direction"       : "in",
             "xtick.direction"       : "in"
             }
plt.rcParams.update(newparams)

rNS = 10
r_NS = 10

def load_tree(filename):
    tree = []
    with open(filename) as f:
        line = f.readline()
        i = -1
        while line:
            tree.append({}); i += 1
            tree[i]["species"], w, prob, wp = line.strip().split()
            tree[i]["weight"] = float(w)
            tree[i]["prob"] = 1 if float(wp) == -1 else float(prob)
            tree[i]["parent_weight"] = float(wp)
            lc = f.readline()
            if lc[0] == "-": # No splitting
                tree[i]["crossings_x"] = []
                tree[i]["crossings_y"] = []
                tree[i]["crossings_z"] = []
                tree[i]["final"] = True
                lc = f.readline()
                lc = f.readline()
            else:
                tree[i]["crossings_x"] = [float(n) for n in lc.strip().split()]
                lc = f.readline()
                tree[i]["crossings_y"] = [float(n) for n in lc.strip().split()]
                lc = f.readline()
                tree[i]["crossings_z"] = [float(n) for n in lc.strip().split()]
                tree[i]["final"] = False
            if i == 0: tree[i]["final"] = True # Main axion
            tree[i]["x"] = np.array(
                    [float(n) for n in f.readline().strip().split()])
            tree[i]["y"] = np.array(
                    [float(n) for n in f.readline().strip().split()])
            tree[i]["z"] = np.array(
                    [float(n) for n in f.readline().strip().split()])
            tree[i]["r"] = (tree[i]["x"]**2 + tree[i]["y"]**2 +
                            tree[i]["z"]**2)**.5
            tree[i]["NS"] = True if np.min(tree[i]["r"]) < 1.01*r_NS else False
            line = f.readline()
    return tree

        
tree = load_tree("results/tree__GR_1")

fig = plt.figure(figsize=(9, 7))
ax = plt.axes(projection='3d')


x0 = tree[0]["x"][0]
y0 = tree[0]["y"][0]
z0 = tree[0]["z"][0]
ax.plot3D(tree[0]["x"][0:1], tree[0]["y"][0:1], tree[0]["z"][0:1],
        marker="o", color="m", markersize=10)

NNN = 10

# Find outermost crossings
xmin = 1e100; xmax = 1e-100
ymin = 1e100; ymax = 1e-100
zmin = 1e100; zmax = 1e-100
for n in tree:
    for x in n["crossings_x"]:
        if x < xmin: xmin = x
        if x > xmax: xmax = x
    for y in n["crossings_y"]:
        if y < ymin: ymin = y
        if y > ymax: ymax = y
    for z in n["crossings_z"]:
        if z < zmin: zmin = z
        if z > zmax: zmax = z

add = 20
xmin = min(x0, xmin) - add
ymin = min(y0, ymin) - add
zmin = min(z0, zmin) - add
xmax = max(x0, xmax) + add
ymax = max(y0, ymax) + add
zmax = max(z0, zmax) + add


# Do not plot outide limits if final
for n in tree:
    print(n["final"])
    if n["final"] and (not n["NS"] or n["species"][0] == "a"):
        flagx = np.logical_or(n["x"]<xmin, n["x"]>xmax)
        flagy = np.logical_or(n["y"]<ymin, n["y"]>ymax)
        flagz = np.logical_or(n["z"]<zmin, n["z"]>zmax)
        i = np.where(np.logical_not(
            np.logical_or(flagx, np.logical_or(flagy, flagz))))
        n["x"] = n["x"][i]
        n["y"] = n["y"][i]
        n["z"] = n["z"][i]

# Forwards in time
c = "C0"

cmap = plt.get_cmap("copper").reversed()
vmin = np.log10(np.abs(np.min([n["weight"] for n in tree])))
if vmin == -np.inf: vmin = -10
print("vmin:", vmin)
def get_color(w0):
    w = w0
    if w0 == 0: w = 1e-10
    print(w)
    vmax = 0
    lw = np.log10(w)
    f = (lw - vmin)/(vmax - vmin)
    return cmap(f)

i = tree[0]
ls = ":" if i["species"][0]=="a" else "-."
c = "r"
ax.plot3D(i["x"][:], i["y"][:], i["z"][:], linestyle=ls, color=c)
ax.plot3D(i["crossings_x"], i["crossings_y"], i["crossings_z"],
        linestyle="", marker="*", color="r")
ax.plot3D([i["x"][-1]], [i["y"][-1]], [i["z"][-1]],
        linestyle="", marker="s", color="r", markersize=10)

for i in tree[1:]:
    if i["weight"] < cutoff: continue
    ls = "--" if i["species"][0]=="a" else "-"
    #lc = 100# int(0 if len(i["crossings_x"]) == 0 else i["crossings"][-1])
    #N = min(lc + NNN, len(i["x"]))
    c = get_color(abs(i["parent_weight"])*i["prob"])
    c = get_color(abs(i["parent_weight"])*i["weight"])
    ax.plot3D(i["x"][:], i["y"][:], i["z"][:], linestyle=ls, color=c)
    ax.plot3D(i["crossings_x"], i["crossings_y"], i["crossings_z"],
            linestyle="", marker="*", color="g")
    if not i["NS"] and i["final"]:
        ax.plot3D([i["x"][-1]], [i["y"][-1]], [i["z"][-1]],
            linestyle="", marker="s", color="b")


u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = rNS*np.cos(u)*np.sin(v)
y = rNS*np.sin(u)*np.sin(v)
z = rNS*np.cos(v)
ax.plot_surface(x, y, z, alpha=0.5, color="C0")

ax.set_xlim(min(-rNS, xmin), max(rNS, xmax))
ax.set_ylim(min(-rNS, ymin), max(rNS, ymax))
ax.set_zlim(min(-rNS, zmin), max(rNS, zmax))

sm = plt.cm.ScalarMappable(cmap=cmap)
sm._A = [0, vmin]
fig.colorbar(sm, label="Log probability")
fig.tight_layout()
ax.set_xlabel(r"$x/r_\mathrm{NS}$")
ax.set_ylabel(r"$y/r_\mathrm{NS}$")
ax.set_zlabel(r"$z/r_\mathrm{NS}$")
ax.plot([], [], linestyle="", marker="o", color="m",
        label="Sampled conversion point")
ax.plot([], [], linestyle=":", marker="", color="r",
        label="Backtraced axion")
ax.plot([], [], linestyle="", marker="*", color="g",
        label="Conversion point")
ax.plot([], [], linestyle="", marker="s", color="b",
        label="Escaping particle")
ax.plot([], [], linestyle="-", marker="", color="k",
        label="Photon")
ax.plot([], [], linestyle="--", marker="", color="k",
        label="Axion")

fig.legend()

plt.show()
