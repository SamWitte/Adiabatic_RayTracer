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
        # misalignment angle
        "--ThetaM"
            arg_type = Float64
            default = 0.2

        # number photon trajectories
        "--Nts"
            arg_type = Int
            default = 100
        # file tage
        "--ftag"
            arg_type = String
            default = ""
        # rotational freq NS
        "--rotW"
            arg_type = Float64
            default = 1.0
        # axion mass
        "--MassA"
            arg_type = Float64
            default = 2e-5
        # coupling
        "--Axg"
            arg_type = Float64
            default = 1e-12
        # surface magnetic field
        "--B0"
            arg_type = Float64
            default = 1e14
        # should we run ray tracer?
        "--run_RT"
            arg_type = Int
            default = 1
        # should we combine file runs
        "--run_Combine"
            arg_type = Int
            default = 0
        # how many runs do we combine
        "--side_runs"
            arg_type = Int
            default = 0
        # what batchsize is used?
        "--batchSize"
            arg_type = Int
            default = 4
        # radius NS
        "--rNS"
            arg_type = Float64
            default = 10.0
        # Mass  NS
        "--Mass_NS"
            arg_type = Float64
            default = 1.0
        # vel NS x
        "--vNS_x"
            arg_type = Float64
            default = 0.0
        # vel NS y
        "--vNS_y"
            arg_type = Float64
            default = 0.0
        # vel NS z
        "--vNS_z"
            arg_type = Float64
            default = 0.0
        ########################################################################
        # ---- Tree parameters ----
        # create tree or standard run (call main_runner or main_runner_tree)
        # 0->standard, 1->tree, 2->single event with init conditions, tree
        "--type"
            arg_type = Int
            default = 0
        # seed
        "--seed"
            arg_type = Int
            default = -1 # random seed
        "--saveTree"
            arg_type = Int
            default = 1
        "--probCutoff"
            arg_type = Float64
            default = 1e-10
        "--numCutoff"
            arg_type = Int
            default = 5
        "--maxNodes"
            arg_type = Int
            default = 5
        # ---- parameters for single run ----
        "--species"
            arg_type = String
            default = "axion"
        "--x0"
            arg_type = Float64
            default = 100.
        "--y0"
            arg_type = Float64
            default = 100.
        "--z0"
            arg_type = Float64
            default = 100.
        "--kx0"
            arg_type = Float64
            default = -1.
        "--ky0"
            arg_type = Float64
            default = -1.
        "--kz0"
            arg_type = Float64
            default = -1.
        "--forward" 
            arg_type = Bool
            default = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

Mass_a = parsed_args["MassA"]; # eV
Ax_g = parsed_args["Axg"]; # 1/GeV
θm = parsed_args["ThetaM"]; # rad
ωPul = parsed_args["rotW"]; # 1/s
B0 = parsed_args["B0"]; # G
rNS = parsed_args["rNS"]; # km
Mass_NS = parsed_args["Mass_NS"]; # solar mass
ωProp = "Simple"; # NR: Standard. Dispersion relation
Ntajs = parsed_args["Nts"];
gammaF = [1.0, 1.0]
batchSize = parsed_args["batchSize"]; # how to batch runs
CLen_Scale = false # if true, perform cut due to de-phasing
cutT = 10000; # keep highest weight 'cutT' each batch
fix_time = 0.0; # eval at fixed time = 0?
file_tag = parsed_args["ftag"];  # if you dont want to cut on Lc "_NoCutLc_";
ode_err = 1e-6; # need strong error
ntimes = 100 # how many points on photon traj to keep
vNS = [parsed_args["vNS_x"] parsed_args["vNS_y"] parsed_args["vNS_z"]]; # relative neutron star velocity
flat = false; # flat space or schwartzchild
isotropic = true; # default is anisotropic
melrose = true; # keep true, more efficient
ntimes_ax = 500; # vector scan for resonance

# Tree parameters
saveTree = parsed_args["saveTree"]
num_cutoff = parsed_args["numCutoff"]
prob_cutoff = parsed_args["probCutoff"]
max_nodes = parsed_args["maxNodes"]
seed = parsed_args["seed"]
# Single run parameters
species = parsed_args["species"]
x0 = parsed_args["x0"]
y0 = parsed_args["y0"]
z0 = parsed_args["z0"]
kx0 = parsed_args["kx0"]
ky0 = parsed_args["ky0"]
kz0 = parsed_args["kz0"]

print("Parameters: ", Mass_a, "\n")
print(Ax_g, "\n")
print(θm, "\n")
print(ωPul, "\n")
print(B0, "\n")
print(rNS, "\n")
print(Mass_NS, "\n")
print(ωProp, "\n")
print(Ntajs, "\n")
print(gammaF, "\n")
print(batchSize, "\n")
print(cutT, "\n")

time0=Dates.now()

if parsed_args["run_RT"] == 1
  if parsed_args["type"] == 0
    @inbounds @fastmath main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS,
              Mass_NS, ωProp, Ntajs, gammaF, batchSize;
              flat=flat, isotropic=isotropic, melrose=melrose, ode_err=ode_err,
              cutT=cutT, fix_time=fix_time, CLen_Scale=CLen_Scale,
              file_tag=file_tag, ntimes=ntimes, v_NS=vNS, ntimes_ax=ntimes_ax);
  elseif parsed_args["type"] == 1
    @inbounds @fastmath main_runner_tree(Mass_a, Ax_g, θm, ωPul, B0, rNS,
              Mass_NS, ωProp, Ntajs, gammaF, batchSize;
              flat=flat, isotropic=isotropic, melrose=melrose, ode_err=ode_err,
              cutT=cutT, fix_time=fix_time, CLen_Scale=CLen_Scale,
              file_tag=file_tag, ntimes=ntimes, v_NS=vNS, ntimes_ax=ntimes_ax,
              saveTree=saveTree, num_cutoff=num_cutoff, 
              prob_cutoff=prob_cutoff, iseed=seed, max_nodes=max_nodes)
  else 
    @inbounds @fastmath single_runner(
              species, x0, y0, z0, kx0, ky0, kz0,
              Mass_a, Ax_g, θm, ωPul, B0, rNS,
              Mass_NS, ωProp, Ntajs, gammaF, batchSize;
              flat=flat, isotropic=isotropic, melrose=melrose, ode_err=ode_err,
              cutT=cutT, fix_time=fix_time, CLen_Scale=CLen_Scale,
              file_tag=file_tag, ntimes=ntimes, v_NS=vNS, ntimes_ax=ntimes_ax,
              saveTree=saveTree, num_cutoff=num_cutoff, iseed=seed, 
              prob_cutoff=prob_cutoff, forwards=parsed_args["forward"])

  end
end


function combine_files(Mass_a, Ax_g, θm, ωPul, B0, Ntajs, Nruns, ode_err, fix_time, file_tag, ntimes, v_NS)
   
    fileL = String[];
    fileL_EV = String[];
    
    for i = 0:(Nruns-1)
        # file_tagL = file_tag * "_" * string(i)
        # if fix_time != Nothing
        #     file_tagL *= "_fixed_time_"*string(fix_time);
        # end
        # file_tagL *= "_odeErr_"*string(ode_err);
        # file_tagL *= "_vxNS_"*string(v_NS[1]);
        # file_tagL *= "_vyNS_"*string(v_NS[2]);
        # file_tagL *= "_vzNS_"*string(v_NS[3]);
        # fileN = "results/Fast_Trajectories_MassAx_"*string(Mass_a)*"_AxionG_"*string(Ax_g)*"_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0);
        # fileN *= "_Ax_trajs_"*string(Ntajs);
        # fileN *= "_N_Times_"*string(ntimes)*"_"*file_tagL*"_.npz";
        fileN = "results/final_" * file_tag * string(i)
        push!(fileL, fileN);
    end
    
    # hold = npzread(fileL[1]);
    
    # divide off by num files combining...
    for i = 2:Nruns
        hold = vcat(hold, npzread(fileL[i]));
    end
    hold[:, 6] ./= Nruns;
    
#    for i = 1:Nruns
#        Base.Filesystem.rm(fileL[i])
#    end
    
    fileN = "results/Fast_Trajectories_MassAx_"*string(Mass_a)*"_AxionG_"*string(Ax_g)*"_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0);
    fileN *= "_Ax_trajs_"*string(Ntajs * Nruns);
    fileN *= "_N_Times_"*string(ntimes)*"_"*file_tag*"_.npz";
    npzwrite(fileN, hold);
end

if parsed_args["run_Combine"] == 1
    combine_files(Mass_a, Ax_g, θm, ωPul, B0, Ntajs, parsed_args["side_runs"], ode_err, fix_time, file_tag, ntimes, vNS);
end


time1=Dates.now()
print("\n")
print("time diff: ", time1-time0)
print("\n")

