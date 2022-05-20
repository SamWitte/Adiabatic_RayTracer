from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

cutoff = 0


fontsize = 12
newparams = {'figure.figsize'        : (5, 4),
             'font.size'             : fontsize,
             'mathtext.fontset'      : 'stix',
             'font.family'           : 'STIXGeneral',
             'lines.markersize'      : 5,
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
            tree[i]["NS"] = True if np.min(tree[i]["r"]) < 1.1*r_NS else False
            line = f.readline()
    return tree


plot = True
savefig = False
showfig = True
runs = range(1, 40)
runs = [34]

list_p_naive = [[], [], []]
list_p_in = [[], [], []]
list_p_out = [[], [], []]
list_p_ain_pout = [[], [], []]
list_p_one_splitting = [[], [], []]
pout_weighted = [[], []]

for num in runs:

    print("\n--------- num: %i ---------"%(num))
        
    tree = load_tree("results/forward_%i"%(num))
    tree_b = load_tree("results/backward_%i"%(num))

    if plot:
        fig = plt.figure(figsize=(9, 7))
        ax = plt.axes(projection='3d')

        fig2d, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        x0 = tree[0]["x"][0]
        y0 = tree[0]["y"][0]
        z0 = tree[0]["z"][0]
        ax.plot3D(tree[0]["x"][0:1], tree[0]["y"][0:1], tree[0]["z"][0:1],
                marker="o", color="r")
        ax1.plot(tree[0]["x"][0:1], tree[0]["y"][0:1], marker="o", color="r") 
        ax2.plot(tree[0]["y"][0:1], tree[0]["z"][0:1], marker="o", color="r") 
        ax3.plot(tree[0]["z"][0:1], tree[0]["x"][0:1], marker="o", color="r") 
        

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


        # Do not plot outide limits if final
        for t in [tree, tree_b]:
            for n in t:
                if n["final"] and not n["NS"]:
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
        vmin = np.log10(np.min([n["weight"] for n in tree]))
        vmin = min(vmin, np.log10(np.min([n["weight"] for n in tree_b])))
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
            c = get_color(abs(i["parent_weight"])*i["prob"])
            ax.plot3D(i["x"][:], i["y"][:], i["z"][:], linestyle=ls, color=c)
            ax1.plot(i["x"][:], i["y"][:], linestyle=ls, color=c)
            ax2.plot(i["y"][:], i["z"][:], linestyle=ls, color=c)
            ax3.plot(i["z"][:], i["x"][:], linestyle=ls, color=c)
            ax.plot3D(i["crossings_x"], i["crossings_y"], i["crossings_z"],
                    linestyle="", marker="*", color="g")
            ax1.plot(i["crossings_x"], i["crossings_y"], linestyle="",
                    marker="*", color="g")
            ax2.plot(i["crossings_y"], i["crossings_z"], linestyle="",
                    marker="*", color="g")
            ax3.plot(i["crossings_z"], i["crossings_x"], linestyle="",
                    marker="*", color="g")
            if not i["NS"] and i["final"]:
                ax.plot3D([i["x"][-1]], [i["y"][-1]], [i["z"][-1]],
                    linestyle="", marker="s", color="b")
                ax1.plot([i["x"][-1]], [i["y"][-1]], linestyle="", marker="s",
                        color="b")
                ax2.plot([i["y"][-1]], [i["z"][-1]], linestyle="", marker="s",
                        color="b")
                ax3.plot([i["z"][-1]], [i["x"][-1]], linestyle="", marker="s",
                        color="b")

        for i in tree_b:
            if i["weight"] < cutoff: continue
            ls = ":" if i["species"][0]=="a" else "-."
            #lc = 100# int(0 if len(i["crossings_x"]) == 0 else i["crossings"][-1])
            #N = min(lc + NNN, len(i["x"]))
            c = get_color(abs(i["parent_weight"])*i["prob"])
            ax.plot3D(i["x"][:], i["y"][:], i["z"][:], linestyle=ls, color=c)
            ax1.plot(i["x"][:], i["y"][:], linestyle=ls, color=c)
            ax2.plot(i["y"][:], i["z"][:], linestyle=ls, color=c)
            ax3.plot(i["z"][:], i["x"][:], linestyle=ls, color=c)
            ax.plot3D(i["crossings_x"], i["crossings_y"], i["crossings_z"],
                    linestyle="", marker="*", color="g")
            ax1.plot(i["crossings_x"], i["crossings_y"], linestyle="",
                    marker="*", color="g")
            ax2.plot(i["crossings_y"], i["crossings_z"], linestyle="",
                    marker="*", color="g")
            ax3.plot(i["crossings_z"], i["crossings_x"], linestyle="",
                    marker="*", color="g")
            if not i["NS"] and i["final"]:
                ax.plot3D([i["x"][-1]], [i["y"][-1]], [i["z"][-1]],
                    linestyle="", marker="^", color="m")
                ax1.plot([i["x"][-1]], [i["y"][-1]], linestyle="", marker="^",
                        color="m")
                ax2.plot([i["y"][-1]], [i["z"][-1]], linestyle="", marker="^",
                        color="m")
                ax3.plot([i["z"][-1]], [i["x"][-1]], linestyle="", marker="^",
                        color="m")


        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = rNS*np.cos(u)*np.sin(v)
        y = rNS*np.sin(u)*np.sin(v)
        z = rNS*np.cos(v)
        ax.plot_surface(x, y, z, alpha=0.5, color="C0")

        ax.set_xlim(min(-rNS, xmin), max(rNS, xmax))
        ax1.set_xlim(min(-rNS, xmin), max(rNS, xmax))
        ax2.set_xlim(min(-rNS, ymin), max(rNS, ymax))
        ax3.set_xlim(min(-rNS, zmin), max(rNS, zmax))
        ax.set_ylim(min(-rNS, ymin), max(rNS, ymax))
        ax1.set_ylim(min(-rNS, ymin), max(rNS, ymax))
        ax2.set_ylim(min(-rNS, zmin), max(rNS, zmax))
        ax3.set_ylim(min(-rNS, xmin), max(rNS, xmax))
        ax.set_zlim(min(-rNS, zmin), max(rNS, zmax))

        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm._A = [0, vmin]
        fig.colorbar(sm, label="Log probability")
        fig.tight_layout()
        fig2d.tight_layout()
        ax.set_xlabel(r"$x/r_\mathrm{NS}$")
        ax1.set_xlabel(r"$x/r_\mathrm{NS}$")
        ax2.set_xlabel(r"$y/r_\mathrm{NS}$")
        ax3.set_xlabel(r"$z/r_\mathrm{NS}$")
        ax.set_ylabel(r"$y/r_\mathrm{NS}$")
        ax1.set_ylabel(r"$y/r_\mathrm{NS}$")
        ax2.set_ylabel(r"$z/r_\mathrm{NS}$")
        ax3.set_ylabel(r"$x/r_\mathrm{NS}$")
        ax.set_zlabel(r"$z/r_\mathrm{NS}$")
        ax.plot([], [], linestyle="", marker="o", color="r",
                label="Initial conversion")
        ax.plot([], [], linestyle="", marker="*", color="g",
                label="Level crossing")
        ax.plot([], [], linestyle="", marker="s", color="b",
                label="Escaping particle")
        ax.plot([], [], linestyle="", marker="^", color="m",
                label="Approaching particle")
        ax.plot([], [], linestyle="-", marker="", color="k",
                label="Photon (forward)")
        ax.plot([], [], linestyle="--", marker="", color="k",
                label="Axion (forward)")
        ax.plot([], [], linestyle="-.", marker="", color="k",
                label="Photon (backward)")
        ax.plot([], [], linestyle=":", marker="", color="k",
                label="Axion (backward)")

        fig.legend()
        if savefig: fig.savefig("figures/%i.pdf"%(num)) 
        if savefig: fig2d.savefig("figures/2d-%i.pdf"%(num)) 

    # Play around with statistics!
    ###########################################################################
    p0 = tree[0]["prob"]
    p_NS = 0; p_a = 0; p_p = 0
    p_tmp = 0
    for n in tree:
        if n["final"]:
            if n["r"][-1] < rNS*1.1: p_NS += n["weight"]
            elif n["species"][0] == "a": p_a += n["weight"]
            elif n["species"][0] == "p":
                p_p += n["weight"]
                p_tmp += n["weight"]
            else: raise Exception("Missing case")


    print("Out")
    list_p_out[0].append(p_p) 
    list_p_out[1].append(p_a) 
    list_p_out[2].append(p_NS) 
    print("  photon: %.10f"%(p0*p_p))
    print("  axion:  %.10f"%(p_a))
    print("  NS:     %.10f"%(p_NS))


    p_NS = 0; p_a = 0; p_p = 0
    a_tmp = 0
    for n in tree_b:
        if n["final"]:
            if n["r"][-1] < rNS*1.1: p_NS += n["weight"]
            elif n["species"][0] == "a":
                p_a += n["weight"]
                a_tmp += n["weight"]
            elif n["species"][0] == "p": p_p += n["weight"]
            else: raise Exception("Missing case")
    
    pout_weighted[0].append(p_tmp) # Probability for photon out
    pout_weighted[1].append(a_tmp) # Probability for axion in
    print(p_tmp, a_tmp)

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

    if len(tree) == 3:
        p_NS = 0; p_a = 0; p_p = 0
        for n in tree:
            if n["final"]:
                if n["r"][-1] < rNS*1.1: p_NS += n["weight"]
                elif n["species"][0] == "a": p_a += n["weight"]
                elif n["species"][0] == "p": p_p += n["weight"]
                else: raise Exception("Missing case")
        list_p_one_splitting[0].append(p_p) 
        list_p_one_splitting[1].append(p_a) 
        list_p_one_splitting[2].append(p_NS)

    if list_p_in[0][-1] < 0.1:# and list_p_out[0][-1] > 0.9:
        list_p_ain_pout[0].append(list_p_out[0][-1])
        list_p_ain_pout[1].append(list_p_out[1][-1])
        list_p_ain_pout[2].append(list_p_out[2][-1])


if plot and showfig:
    plt.show()

print("Summary: ")
print(" Photons out in naive approach:      %.2e"%(np.mean(list_p_naive[0])))
print(" Photons out in full tree:           %.2e"%(np.mean(list_p_out[0])))
print(" Photons in NS in full tree:         %.2e"%(np.mean(list_p_out[2])))
print(" Full tree, only 'nice' tracks:      %.2e"%(
    np.mean(list_p_one_splitting[0])))
print(" Photons, when  a in:                %.2e"%(np.mean(list_p_ain_pout[2])))

print()
print(np.average(pout_weighted[0], weights=pout_weighted[1]))

#plt.figure()
#plt.hist(list_p_naive[0], histtype="step", label="Naive")
#plt.hist(list_p_out[0], histtype="step", label="Tree")
#plt.legend()
#plt.show()
