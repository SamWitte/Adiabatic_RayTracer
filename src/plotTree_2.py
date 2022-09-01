from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


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

savefig = True
showfig = True

fname = "tree__GR_18"
tree = load_tree("results/" + fname)

fig = plt.figure(figsize=(9, 7))
ax = plt.axes(projection='3d')


x0 = tree[0]["x"][0]
y0 = tree[0]["y"][0]
z0 = tree[0]["z"][0]
#ax.plot3D(tree[0]["x"][0:1], tree[0]["y"][0:1], tree[0]["z"][0:1], marker="o", color="m", markersize=10)
ax.plot3D(tree[0]["x"][0:1], tree[0]["y"][0:1], tree[0]["z"][0:1], marker="*", color="#69140E", markersize=10)

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
    if w0 == 1.0: return "k"
    w = w0
    if w0 == 0: w = 1e-10
    vmax = 0
    lw = np.log10(w)
    f = (lw - vmin)/(vmax - vmin)
    return cmap(f)

i = tree[0]
ls = ":" if i["species"][0]=="a" else "-."
c = "k"
ax.plot3D(i["x"][:], i["y"][:], i["z"][:], linestyle="--", color=c)
#ax.plot3D(i["crossings_x"], i["crossings_y"], i["crossings_z"], linestyle="", marker="*", color="r")
#ax.plot3D([i["x"][-1]], [i["y"][-1]], [i["z"][-1]], linestyle="", marker="s", color="r", markersize=10)

alphVs = np.ones(len(tree[1:]))
#for idx,i in enumerate(tree[1:]):
    # if i["weight"] < cutoff: continue
    # ls = "--" if i["species"][0]=="a" else "-"
    #lc = 100# int(0 if len(i["crossings_x"]) == 0 else i["crossings"][-1])
    #N = min(lc + NNN, len(i["x"]))
    
    # c = get_color(abs(i["parent_weight"])*i["prob"])
    # c = get_color(abs(i["parent_weight"])*i["weight"])
    
    # alphVs[idx] = np.log10(abs(i["parent_weight"])*i["weight"])
#    alphVs[idx] = np.log10(abs(i["parent_weight"])*i["prob"])
    
alphVs /= np.max(alphVs)

#for idx,i in enumerate(tree[1:]):
# skipList = [9,11]
skipList = []
for idx,i in enumerate(tree[1:]):
#    print(idx)
#    if idx not in skipList:
#        continue
    if i["species"][0] == "a":
        print("axion, weight ", i["weight"], i["final"])
    else:
        print("photon, weight ", i["weight"], i["final"])
    c = "#33658A" if i["species"][0]=="a" else "#F7996E"
    ax.plot3D(i["x"][:-1], i["y"][:-1], i["z"][:-1], color=c, alpha=alphVs[idx], lw=2)
    ax.plot3D(i["crossings_x"], i["crossings_y"], i["crossings_z"],
            linestyle="", marker="*", color="#69140E")
#    if not i["NS"] and i["final"]:
#        ax.plot3D([i["x"][-1]], [i["y"][-1]], [i["z"][-1]],
#            linestyle="", marker="s", color="b")

    a = Arrow3D([i["x"][-2], i["x"][-1]], [i["y"][-2], i["y"][-1]],
                    [i["z"][-2], i["z"][-1]], mutation_scale=20, alpha=alphVs[idx],
                    lw=2, arrowstyle="-|>", color=c)
    ax.add_artist(a)

u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = rNS*np.cos(u)*np.sin(v)
y = rNS*np.sin(u)*np.sin(v)
z = rNS*np.cos(v)
ax.plot_surface(x, y, z, alpha=0.5, color="#A7A5C6")

maxV = np.max(np.array([xmax, ymax, zmax]))
ax.set_xlim(-maxV, maxV)
ax.set_ylim(-maxV, maxV)
ax.set_zlim(-maxV, maxV)
#ax.set_xlim(min(-rNS, xmin), max(rNS, xmax))
#ax.set_ylim(min(-rNS, ymin), max(rNS, ymax))
#ax.set_zlim(min(-rNS, zmin), max(rNS, zmax))

sm = plt.cm.ScalarMappable(cmap=cmap)
sm._A = [0, vmin]
#fig.colorbar(sm, label="Log probability")
fig.tight_layout()
ax.set_xlabel(r"$x/r_\mathrm{NS}$")
ax.set_ylabel(r"$y/r_\mathrm{NS}$")
ax.set_zlabel(r"$z/r_\mathrm{NS}$")

ax.plot([], [], linestyle="--", marker="", color="k",
        label="In-falling Axion")
ax.plot([], [], linestyle="", marker="*", color="#69140E",
        label="Conversion point")
ax.plot([], [], linestyle="-",  color="#33658A",
        label="Sourced Axion")
ax.plot([], [], linestyle="-", color="#F7996E",
        label="Sourced Photon")
# ax.view_init(20, 70)
ax.view_init(10, 5)
fig.legend()

if showfig: plt.show()
if savefig: fig.savefig("figures/" + fname + ".pdf")
