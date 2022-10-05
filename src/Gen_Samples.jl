using ArgParse
using SpecialFunctions
using LinearAlgebra
using NPZ
using Dates
using Statistics
using Base
using Random
#using Profile
include("RayTracer.jl")
include("MainRunner.jl")



function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--ThetaM"
            help = "misalignment angle in rad"
            arg_type = Float64
            default = 0.2
        "--Nts"
            help = "number photon trajectories"
            arg_type = Int
            default = 100
        "--ftag"
            help = "file tag"
            arg_type = String
            default = ""
        "--rotW"
            help = "rotational freq NS in 1/s"
            arg_type = Float64
            default = 1.0
        "--MassA"
            help = "axion mass in eV"
            arg_type = Float64
            default = 2e-5
        "--Axg"
            help = "coupling in 1/GeV"
            arg_type = Float64
            default = 1e-12
        "--B0"
            help = "surface magnetic field in G"
            arg_type = Float64
            default = 1e14
        "--run_RT"
            help = "should we run ray tracer?"
            arg_type = Int
            default = 1
        "--run_Combine"
            help = "should we combine file runs"
            arg_type = Int
            default = 0
        "--side_runs"
            help = "how many runs do we combine?"
            arg_type = Int
            default = 0
        # Not used in the generation of trees
        #"--batchSize"
        #    help = "what batchsize is used? Batchsize=1 by hand when the full"*
        #           "tree is generated"
        #    arg_type = Int
        #    default = 4
        "--rNS"
            help = "radius NS in km"
            arg_type = Float64
            default = 10.0
        "--Mass_NS"
            help = "Mass NS in solar masses"
            arg_type = Float64
            default = 1.0
        "--vNS_x"
            help = "vel NS x in c"
            arg_type = Float64
            default = 0.0
        "--vNS_y"
            help = "vel NS y in c"
            arg_type = Float64
            default = 0.0
        "--vNS_z"
            help = "vel NS z in c"
            arg_type = Float64
            default = 0.0
        # ---- Tree parameters ----
        "--saveMode"
            help = "What data do we store? "                                   *
                   " 0: Only the essentials in a npy file "                    *
                   " 1: More information in the npy file "                     *
                   " 2: Save also in clear text with more information "        *
                   " 3: Save entire tree"
            arg_type = Int
            default = 0
        "--probCutoff"
            help = "Stop the generation of the tree when the total "           *
                   "probability/weight of all outgoing particles has reached " *
                   "'prob_cutoff'. The final error will be <= prob_cutoff. "   *
                   "This should be equal to the final uncertainty we want to " *
                   "achieve."
            arg_type = Float64
            default = 1e-10
        "--numCutoff"
            help = "Stops when num_cutoff outgoing particles has been found. " *
                   "This should be as large as possible. It can be used to "   *
                   "cut off some large trees. The final error will be "        *
                   "smaller than 2^-num_cutoff."
            arg_type = Int
            default = 5
        "--MCNodes"
            help = "The number of total subbrances to compute before the "     *
                   "generation of the tree is transitioned to a 'pure' MC "    *
                   "selection, i.e. the number of particles propagated "       *
                   "between subtrees." 
            arg_type = Int
            default = 5
        "--maxNodes"
            help = "The number of total subbrances to compute before the "     *
                   "generation of the tree is stopped."
            arg_type = Int
            default = 50
        "--seed"
            help = "Seed for random number generator. Use seed=-1 for a "*
                   "random seed"
            arg_type = Int
            default = -1 # random seed
    end

    return parse_args(s)
end

# User changable parameters
parsed_args = parse_commandline()
Mass_a = parsed_args["MassA"]; # eV
Ax_g = parsed_args["Axg"]; # 1/GeV
θm = parsed_args["ThetaM"]; # rad
ωPul = parsed_args["rotW"]; # 1/s
B0 = parsed_args["B0"]; # G
rNS = parsed_args["rNS"]; # km
Mass_NS = parsed_args["Mass_NS"]; # solar mass
Ntajs = parsed_args["Nts"];
#batchSize = parsed_args["batchSize"]; # how to batch runs
file_tag = parsed_args["ftag"]
vNS = [parsed_args["vNS_x"] parsed_args["vNS_y"] parsed_args["vNS_z"]]
# --- Tree parameters ---
saveMode = parsed_args["saveMode"]
num_cutoff = parsed_args["numCutoff"]
prob_cutoff = parsed_args["probCutoff"]
MC_nodes = parsed_args["MCNodes"]
max_nodes = parsed_args["maxNodes"]
seed = parsed_args["seed"]

# Fixed parameters
ωProp = "Simple"; # NR: Standard. Dispersion relation
gammaF = [1.0, 1.0]
CLen_Scale = false # if true, perform cut due to de-phasing
cutT = 10000; # keep highest weight 'cutT' each batch
fix_time = 0.0; # eval at fixed time = 0?
ode_err = 1e-6; # need strong error
ntimes = 100 # how many points on photon traj to keep
flat = false; # flat space or schwartzchild
isotropic = true; # default is anisotropic
melrose = true; # keep true, more efficient
ntimes_ax = 500; # vector scan for resonance
vmean_ax = 220.0
dir_tag = "results"
rho_DM = 0.3
thick_surface=true
n_maxSample=6

print("Axion parameters: ", Mass_a, "\n", Ax_g, "\n")

time0=Dates.now()

if parsed_args["run_RT"] == 1
@inbounds @fastmath main_runner_tree(Mass_a, Ax_g, θm, ωPul, B0, rNS,
              Mass_NS, ωProp, Ntajs, gammaF;
              flat=flat, isotropic=isotropic, melrose=melrose,
              thick_surface=thick_surface, ode_err=ode_err,
              cutT=cutT, fix_time=fix_time, CLen_Scale=CLen_Scale,
              file_tag=file_tag, ntimes=ntimes, v_NS=vNS, rho_DM=rho_DM,
              ntimes_ax=ntimes_ax, vmean_ax=vmean_ax, dir_tag=dir_tag,
              n_maxSample=n_maxSample,
              saveMode=saveMode, num_cutoff=num_cutoff,
              prob_cutoff=prob_cutoff, iseed=seed, MC_nodes=MC_nodes,
              max_nodes=max_nodes)
end


function combine_files(Mass_a, Ax_g, θm, ωPul, B0, Ntajs, Nruns, ode_err, fix_time, file_tag, ntimes, v_NS, dir_tag, num_cutoff, MC_nodes, max_nodes)
   
    fileL = String[];
    
    for i = 0:(Nruns-1)
    
        fileN = dir_tag*"/npy/tree_"
        fileN *= "MassAx_"*string(Mass_a)*"_AxionG_"*string(Ax_g)
        fileN *="_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0)
        fileN *= "_Ax_trajs_"*string(Ntajs)
        fileN *= "_N_Times_"*string(ntimes);
        fileN *= "_num_cutoff_"*string(num_cutoff)
        fileN *= "_MC_nodes_"*string(MC_nodes)
        fileN *= "_max_nodes_"*string(max_nodes)
        fileN *= "_"*file_tag*string(i)*".npy"
        
        push!(fileL, fileN);
    end
    
    hold = npzread(fileL[1]);
    
    # divide off by num files combining...
    for i = 2:Nruns
        hold = vcat(hold, npzread(fileL[i]));
    end
    hold[:, 8] ./= Nruns;
    
    
    fileN = dir_tag*"/"
    fileN *= "MassAx_"*string(Mass_a)*"_AxionG_"*string(Ax_g)
    fileN *="_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0)
    fileN *= "_Ax_trajs_"*string(Ntajs * Nruns)
    fileN *= "_N_Times_"*string(ntimes);
    fileN *= "_num_cutoff_"*string(num_cutoff)
    fileN *= "_MC_nodes_"*string(MC_nodes)
    fileN *= "_max_nodes_"*string(max_nodes)
    fileN *= "_"*file_tag*".npy"

    npzwrite(fileN, hold);
    
    for i = 1:Nruns
        Base.Filesystem.rm(fileL[i])
    end
    
end


if parsed_args["run_Combine"] == 1
    combine_files(Mass_a, Ax_g, θm, ωPul, B0, Ntajs, parsed_args["side_runs"], ode_err, fix_time, file_tag, ntimes, vNS, dir_tag, num_cutoff, MC_nodes, max_nodes);
end




time1=Dates.now()
print("\n")
print("time diff: ", time1-time0)
print("\n")

