from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
                tree[i]["crossings_x"] = []
                tree[i]["crossings_y"] = []
                tree[i]["crossings_z"] = []
                lc = f.readline()
                lc = f.readline()
            else:
                tree[i]["crossings_x"] = [float(n) for n in lc.strip().split()]
                lc = f.readline()
                tree[i]["crossings_y"] = [float(n) for n in lc.strip().split()]
                lc = f.readline()
                tree[i]["crossings_z"] = [float(n) for n in lc.strip().split()]
            tree[i]["x"] = np.array(
                    [float(n) for n in f.readline().strip().split()])
            tree[i]["y"] = np.array(
                    [float(n) for n in f.readline().strip().split()])
            tree[i]["z"] = np.array(
                    [float(n) for n in f.readline().strip().split()])
            tree[i]["r"] = (tree[i]["x"]**2 + tree[i]["y"]**2 +
                            tree[i]["z"]**2)**.5
            line = f.readline()
    return tree

tree = load_tree("results/forward_1")
#tree = load_tree("results/backward_1")


fig = plt.figure()
ax = plt.axes(projection='3d')

x0 = tree[0]["x"][0]
y0 = tree[0]["y"][0]
z0 = tree[0]["z"][0]
ax.plot3D(tree[0]["x"][0:1], tree[0]["y"][0:1], tree[0]["z"][0:1],
        marker="o", color="k")

NNN = 100

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


# Do not plot outide limits
for n in tree:
    flagx = np.logical_or(n["x"]<xmin, n["x"]>xmax)
    flagy = np.logical_or(n["y"]<ymin, n["y"]>ymax)
    flagz = np.logical_or(n["z"]<zmin, n["z"]>zmax)
    i = np.where(np.logical_or(flagx, np.logical_or(flagy, flagz)))
    n["x"][i] = np.nan
    n["y"][i] = np.nan
    n["z"][i] = np.nan

# Forwards in time
c = "C0"

cmap = plt.get_cmap("plasma")
vmin = np.log10(np.min([n["weight"] for n in tree]))
print(vmin)
def get_color(w):
    vmax = 0
    lw = np.log10(w)
    f = (lw - vmin)/(vmax - vmin)
    return cmap(1-f)

for i in tree:
    if i["weight"] < cutoff: continue
    ls = "--" if i["species"][0]=="a" else "-"
    #lc = 100# int(0 if len(i["crossings_x"]) == 0 else i["crossings"][-1])
    #N = min(lc + NNN, len(i["x"]))
    c = get_color(i["weight"])
    ax.plot3D(i["x"][:], i["y"][:], i["z"][:], linestyle=ls, color=c)
    ax.plot3D(i["crossings_x"], i["crossings_y"], i["crossings_z"],
            linestyle="", marker="*", color="k")

# Backwards in time
#c = "C1"
#for i in tree_b:
#    if i["weight"] < cutoff:
#        ax.plot3D(i["x"][0:1], i["y"][0:1], i["z"][0:1], linestyle=ls, color=c,
#                marker="*")
#        continue
#    alpha = np.log10(i["weight"])
#    ls = ":" if i["species"][0]=="a" else "-."
#    lc = int(0 if len(i["crossings"]) == 0 else i["crossings"][-1])
#    N = min(lc + NNN, len(i["x"]))
#    ax.plot3D(i["x"][:N], i["y"][:N], i["z"][:N], linestyle=ls, color=c)

rNS = 10
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


p_NS = 0; p_a = 0; p_p = 0
for n in tree:
    if n["r"][-1] < rNS*1.1: p_NS += n["weight"]
    elif n["species"][0] == "a": p_a += n["weight"]
    elif n["species"][0] == "p": p_a += n["weight"]
    else: raise Exception("Missing case")

print("Outcomes")
print("  photon: ", p_p)
print("  axion:  ", p_a)
print("  NS:     ", p_NS)


"""
def GJ_Model_wp_vec(x, t, θm, ω, B0, rNS)
    # For GJ model, return \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km],
    # origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]

    r = np.sqrt(np.sum(x**2, axis=1))
    
    phi   = np.atan2(x[1], x[0])
    theta = np.acos(x[2]/r)
    psi   = phi - w*t
    
    Bnorm = B0*(rNS/r)**3/2
    
    Br = 2*Bnorm*(cos(thetam)*cos(theta)   + sin(thetam)*sin(thetam)*cos(phi))
    Btheta = Bnorm*(cos(thetam)*sin(theta) - sin(thetam)*cos(theta)*cos(phi))
    Bphi = Bnorm*sin(thetam)*sin(psi)
    
    Bx = Br*sin(theta)*cos(phi)+Btheta*cos(theta)*cos(phi)-Bphi*sin(phi)
    By = Br*sin(theta)*sin(phi)+Btheta*cos(theta)*sin(phi)+Bphi*cos(phi)
    Bz = Br*cos(theta)-Btheta*sin(theta)
    
    nelec = np.abs((2.0*omega*Bz)/np.sqrt(4*np.pi/137)*1.95e-2*hbar)
    omegap = np.sqrt(4*np.pi*nelec/137/5.0e5)

    return ωp
"""
plt.show()
