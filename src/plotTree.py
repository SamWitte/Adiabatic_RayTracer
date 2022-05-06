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
            tree[i]["species"], w, prob, wp = line.strip().split()
            tree[i]["weight"] = float(w)
            tree[i]["prob"] = float(prob)
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


rNS = 10
plot = True
list_p_naive = [[], [], []]
list_p_in = [[], [], []]
list_p_out = [[], [], []]
for num in range(1, 17):

    print("\n--------- num: %i ---------"%(num))
        
    tree = load_tree("results/forward_%i"%(num))
    tree_b = load_tree("results/backward_%i"%(num))

    if plot:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        x0 = tree[0]["x"][0]
        y0 = tree[0]["y"][0]
        z0 = tree[0]["z"][0]
        ax.plot3D(tree[0]["x"][0:1], tree[0]["y"][0:1], tree[0]["z"][0:1],
                marker="o", color="r")

        NNN = 100

# Find outermost crossings
        xmin = 1e100; xmax = 1e-100
        ymin = 1e100; ymax = 1e-100
        zmin = 1e100; zmax = 1e-100
        for t in [tree, tree_b]:
            for n in t:
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
        for t in [tree, tree_b]:
            for n in t:
                flagx = np.logical_or(n["x"]<xmin, n["x"]>xmax)
                flagy = np.logical_or(n["y"]<ymin, n["y"]>ymax)
                flagz = np.logical_or(n["z"]<zmin, n["z"]>zmax)
                i = np.where(np.logical_or(flagx, np.logical_or(flagy, flagz)))
                n["x"][i] = np.nan
                n["y"][i] = np.nan
                n["z"][i] = np.nan

# Forwards in time
        c = "C0"

        cmap = plt.get_cmap("copper").reversed()
        vmin = np.log10(np.min([n["weight"] for n in tree]))
        print(vmin)
        def get_color(w):
            vmax = 0
            lw = np.log10(w)
            f = (lw - vmin)/(vmax - vmin)
            return cmap(f)

        for i in tree:
            if i["weight"] < cutoff: continue
            ls = "--" if i["species"][0]=="a" else "-"
            #lc = 100# int(0 if len(i["crossings_x"]) == 0 else i["crossings"][-1])
            #N = min(lc + NNN, len(i["x"]))
            c = get_color(i["weight"])
            ax.plot3D(i["x"][:], i["y"][:], i["z"][:], linestyle=ls, color=c)
            ax.plot3D(i["crossings_x"], i["crossings_y"], i["crossings_z"],
                    linestyle="", marker="*", color="k")

        for i in tree_b:
            if i["weight"] < cutoff: continue
            ls = ":" if i["species"][0]=="a" else "-."
            #lc = 100# int(0 if len(i["crossings_x"]) == 0 else i["crossings"][-1])
            #N = min(lc + NNN, len(i["x"]))
            c = get_color(i["weight"])
            ax.plot3D(i["x"][:], i["y"][:], i["z"][:], linestyle=ls, color=c)
            ax.plot3D(i["crossings_x"], i["crossings_y"], i["crossings_z"],
                    linestyle="", marker="*", color="k")

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


    p0 = tree[0]["prob"]


    p_NS = 0; p_a = 0; p_p = 0
    for n in tree:
        if n["r"][-1] < rNS*1.1: p_NS += n["weight"]
        elif n["species"][0] == "a": p_a += n["weight"]
        elif n["species"][0] == "p": p_p += n["weight"]
        else: raise Exception("Missing case")

    print("Out")
    list_p_out[0].append(p_p) 
    list_p_out[1].append(p_a) 
    list_p_out[2].append(p_NS) 
    print("  photon: %.10f"%(p0*p_p))
    print("  axion:  %.10f"%(p_a))
    print("  NS:     %.10f"%(p_NS))


    p_NS = 0; p_a = 0; p_p = 0
    for n in tree_b:
        if n["r"][-1] < rNS*1.1: p_NS += n["weight"]
        elif n["species"][0] == "a": p_a += n["weight"]
        elif n["species"][0] == "p": p_p += n["weight"]
        else: raise Exception("Missing case")

    print("In")
    list_p_in[0].append(p_p) 
    list_p_in[1].append(p_a) 
    list_p_in[2].append(p_NS) 
    print("  photon: %.10f"%(p_p))
    print("  axion:  %.10f"%(p_a))
    print("  NS:     %.10f"%(p_NS))

    print("Naive approach")
    P = p0 * (1 - p0)
    list_p_naive[0].append((1-p0)) 
    list_p_naive[1].append(p0) 
    print("  photon: %.10f"%(1 - p0))
    print("  axion:  %.10f"%(p0))

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
if plot: plt.show()

print("Summary: ")
print(" Photons out in naive approach: %.2e"%(np.mean(list_p_naive[0])))
print(" Photons out in full tree:      %.2e"%(np.mean(list_p_out[0])))
print(" Photons in NS in full tree:    %.2e"%(np.mean(list_p_out[2])))


#plt.figure()
#plt.hist(list_p_naive[0], histtype="step", label="Naive")
#plt.hist(list_p_out[0], histtype="step", label="Tree")
#plt.legend()
#plt.show()
