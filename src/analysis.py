import numpy as np
import matplotlib.pyplot as plt

tag = "test"
Mass_a = 2e-5
c_km = 2.99792e5

def load_event_info(tag=""):
    data     = np.loadtxt("results/event_"+tag+"_GR_")
    num      = data[:,0]
    vIfty    = data[:,1:4]
    sln_prob = data[:,4]
    x_in     = data[:,5:9]
    k_in     = data[:,9:13]
    x0       = data[:,13:17]
    k0       = data[:,17:-1]
    time     = data[:,-1]
    return num, vIfty, sln_prob, x_in, k_in, x0, k0, time

def load_final_info(tag=""):
    data     = np.loadtxt("results/final_"+tag+"_GR_")
    num      = np.array(data[:,0], dtype=int)
    weight   = data[:,1]
    species  = data[:,2]
    theta_f  = data[:,3]
    phi_f    = data[:,4]
    abs_f   = data[:,5]
    theta_Xf = data[:,6]
    phi_Xf   = data[:,7]
    abs_Xf  = data[:,8]
    t = data[:, 9]
    return num, weight, species, theta_f, phi_f, abs_f, theta_Xf,phi_Xf,abs_Xf,t

for tag in ["low_coupling", "1e-10","1e-11","1e-12","1e-13","1e-14","1e-15"]:


    try:
        num0, vIfty, sln_prob, x_in, k_in, x0, k0, time = load_event_info(tag)
        num, weight, species, theta_f, phi_f, abs_f, theta_Xf,phi_Xf,abs_Xf,t =\
            load_final_info(tag)
    except Exception as e:
        print("Error with ", tag, ". Skipping!")
        print("The error raised is: ", e)
        continue

    # Time as a function of the velocity at infinity
    vIfty_abs = (vIfty[:,0]**2 + vIfty[:,1]**2 + vIfty[:,2]**2)**.5
    theta = np.arctan(vIfty[:,1]/vIfty[:,0])
    phi = np.arccos(vIfty[:,2]/vIfty_abs)


    plt.figure()
    plt.plot(time[1:], vIfty_abs[1:], "^")
    plt.plot(time[1:], vIfty[1:,0], "o")
    plt.plot(time[1:], vIfty[1:,1], "o")
    plt.plot(time[1:], vIfty[1:,2], "o")
    #plt.plot(time[1:], phi[1:], "s")
    #plt.plot(time[1:], theta[1:], "d")
    plt.ylabel(r"$v_\infty$ [km/s]")
    plt.xlabel("Computation time [s]")
    plt.title(tag)
    #plt.close()


    # Differential power
    plt.figure()
    vIfty_mag = (vIfty[:,0]**2 + vIfty[:,1]**2 + vIfty[:,2]**2)**.5
    gammaA = 1/np.sqrt(1.0 - (vIfty_mag / c_km)**2 )
    erg_inf_ini = Mass_a * np.sqrt(1 + (vIfty_mag / c_km * gammaA)**2)
    P = (sln_prob*erg_inf_ini)[num - 1]
    n_photons = np.sum((species==1)*weight)
    w = P*weight
    for i, label in enumerate(["Axion", "Photon"]):
        flag = np.array(species==i, dtype=int)
        N = np.sum(w*flag)
        y, bins = np.histogram(theta_f, weights=w*flag, bins=30)    
        bc = bins[1:]*.5 + bins[:-1]*.5
        bw = bins[1:]-bins[:-1]
        y = y/(bw*num[-1])
        plt.plot(bc, y, label=label)
    plt.yscale("log")
    plt.title(tag)
    plt.legend()

plt.show()
