# Adiabatic Ray Tracer

```
mkdir -p results/npz/ results/tree/ results/event/
```

## Usage

The easiest way to use the Adiabatic Ray Tracer, is to run it from `Gen_Samples.jl`. 

```
>>> julia Gen_Samples.jl --help

usage: Gen_Samples.jl [--ThetaM THETAM] [--Nts NTS] [--ftag FTAG]
                      [--rotW ROTW] [--MassA MASSA] [--Axg AXG]
                      [--B0 B0] [--run_RT RUN_RT]
                      [--run_Combine RUN_COMBINE]
                      [--side_runs SIDE_RUNS] [--rNS RNS]
                      [--Mass_NS MASS_NS] [--vNS_x VNS_X]
                      [--vNS_y VNS_Y] [--vNS_z VNS_Z]
                      [--saveMode SAVEMODE] [--probCutoff PROBCUTOFF]
                      [--numCutoff NUMCUTOFF] [--MCNodes MCNODES]
                      [--maxNodes MAXNODES] [--seed SEED] [-h]

optional arguments:
  --ThetaM THETAM       misalignment angle in rad (type: Float64,
                        default: 0.2)
  --Nts NTS             number photon trajectories (type: Int64,
                        default: 100)
  --ftag FTAG           file tag (default: "")
  --rotW ROTW           rotational freq NS in 1/s (type: Float64,
                        default: 1.0)
  --MassA MASSA         axion mass in eV (type: Float64, default:
                        2.0e-5)
  --Axg AXG             coupling in 1/GeV (type: Float64, default:
                        1.0e-12)
  --B0 B0               surface magnetic field in G (type: Float64,
                        default: 1.0e14)
  --run_RT RUN_RT       should we run ray tracer? (type: Int64,
                        default: 1)
  --run_Combine RUN_COMBINE
                        should we combine file runs (type: Int64,
                        default: 0)
  --side_runs SIDE_RUNS
                        how many runs do we combine? (type: Int64,
                        default: 0)
  --rNS RNS             radius NS in km (type: Float64, default: 10.0)
  --Mass_NS MASS_NS     Mass NS in solar masses (type: Float64,
                        default: 1.0)
  --vNS_x VNS_X         vel NS x in c (type: Float64, default: 0.0)
  --vNS_y VNS_Y         vel NS y in c (type: Float64, default: 0.0)
  --vNS_z VNS_Z         vel NS z in c (type: Float64, default: 0.0)
  --saveMode SAVEMODE   What data do we store?  0: Only the essentials
                        in a npy file  1: More information in the npy
                        file  2: Save also in clear text with more
                        information  3: Save entire tree (type: Int64,
                        default: 0)
  --probCutoff PROBCUTOFF
                        Stop the generation of the tree when the total
                        probability/weight of all outgoing particles
                        has reached 'prob_cutoff'. The final error
                        will be <= prob_cutoff. This should be equal
                        to the final uncertainty we want to achieve.
                        (type: Float64, default: 1.0e-10)
  --numCutoff NUMCUTOFF
                        Stops when num_cutoff outgoing particles has
                        been found. This should be as large as
                        possible. It can be used to cut off some large
                        trees. The final error will be smaller than
                        2^-num_cutoff. (type: Int64, default: 5)
  --MCNodes MCNODES     The number of total subbrances to compute
                        before the generation of the tree is
                        transitioned to a 'pure' MC selection, i.e.
                        the number of particles propagated between
                        subtrees. (type: Int64, default: 5)
  --maxNodes MAXNODES   The number of total subbrances to compute
                        before the generation of the tree is stopped.
                        (type: Int64, default: 50)
  --seed SEED           Seed for random number generator. Use seed=-1
                        for a random seed (type: Int64, default: -1)
  -h, --help            show this help message and exit
```

Before anything is run, make sure that the output folders are created!

```
mkdir -p results/npy/ results/tree/ results/event/
```

## Output files

* `--saveMode 0`  
    The results are stored in an npy-file: 
    `[photon_trajs id θf ϕf θfX ϕfX absfX sln_prob[1] weight xpos[1] xpos[2] xpos[3] Δω]`
    + `photon_trajs`: event number
    + `id`: 1: photon, 0: axion  
    + `[θf ϕf]`: angles of final momentum  
    + `[θfX ϕfX absfX]`: final position in polar coordinates  
    + `sln_prob`: incoming axions per second  
    + `weight`: weight of outgoing particle. E.g. `sum(weight[id==1]*sln_prob)` is the total number of outgoing photons per second.  
    + `[xpos]`: level crossing drawn in the MC. Note that the initial axion is backtraces. Thus, this is not necessarily the *first*   crossing.
    + `Δω`: energy dispetsion **Not yet implemented**

* `--saveMode 1`  
    More results are stored in the npy-file:
    `[photon_trajs id θf ϕf θfX ϕfX absfX sln_prob[1] weight xpos[1] xpos[2] xpos[3] Δω tree_weight opticalDepth weightC k_init[1] k_init[2] k_init[3] calpha[1], c, info]`
    + `photon_trajs`: event number
    + `id`: 1: photon, 0: axion  
    + `[θf ϕf]`: angles of final momentum  
    + `[θfX ϕfX absfX]`: final position in polar coordinates  
    + `sln_prob`: incoming axions per second  
    + `weight`: weight of outgoing particle. E.g. `sum(weight[id==1]*sln_prob)` is the total number of outgoing photons per second.  
    + `[xpos]`: level crossing drawn in the MC. Note that the initial axion is backtraces. Thus, this is not necessarily the *first*   crossing.
    + `Δω`: energy dispetsion **Not yet implemented**
    + `tree_weight`: Weight comming only from tree **Currently same as `weight`**
    + `opticalDepth`: Optical depth **Currently not implemented**
    + `weightC`: ??? **Currently not implemented**
    + `[k_init]`: MC selected momentum
    + `calpha`: ???
    + `c`: number of subranches that have been considered in the call to `generate_tree`. Note that `generate_tree` is called seperately for the photon and axion in the first crossing.
    + `info`: stop reason. 1: no cutoff, 2: probCutoff, 3: numCutoff, 4: maxNodes, negative: MCNodes reached

* `--saveMode 2`  
    In addition to the npy-file, some additional information is stored in clear text in the `results/event/` folder.  
    + `final_*` contains the information about outgoing particles. The content
    is written to the file on line 631 in MainRunner.jl.
    `Event number | final weight | species (0: axion, 1: photon) | final momentum in spherical coordinates [theta|phi|abs] | final position in spherical coordinates [theta|phi|abs] | time at final crossing (was used for some testing) | Computation time`

    + `event_*` contains information about the MC drawing. The content is written to the file on line 566 and 652.
    `Event number | velocity at infinity (vIfty) in cartesian coordinates [x|y|z] | incoming axions per second (sln_prob) | position and momentum of incoming (fully backtraced) axion in cartesian coordinates [x|y|z|kx|ky|kz] | position and momentum of the MC selection [x|y|z|kx|ky|kz]`


* `--saveMode 3`
    In adiition to the event information, the trajectories of each particle in the tree is stored in `results/tree/`. See `plot/plot_tree.py` for more information.

## TODO
- Check time-input and output of "propagate" for unaligned rotators
- Include opticalDepth and weightC. This should be done for every photon in the tree, e.g. by including a new weight variable in the struct node which is inherited by its parent.
- Take into account the energy dispersion Δω for misaligned
rotators. This must be stored in node and be inherited. Propagate must be edited to be able to accept the energy change as input.
