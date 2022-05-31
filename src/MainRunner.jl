RT = RayTracerGR; # define ray tracer module
func_use = RT.ωNR_e
func_use_SPHERE = RT.ωSimple_SPHERE

function printTree(n::Array)
    print("\n")
    weight_tot = 0
    for i in 1:length(n)
    	print(n[i].species, "  ", n[i].weight, "\n")
    	weight_tot = weight_tot + n[i].weight
    end
    print("Total weight: ", weight_tot, "\n")
    print("\n")
end

function saveTree(n::Array, filename::String="tree.txt", info_level::Int=5)
    if info_level >= 4
      open(filename, "w") do f
        for i in 1:length(n)
          write(f, n[i].species, " ", string(n[i].weight), " ",
                string(n[i].prob), " ",
                string(n[i].parent_weight), "\n")
          if length(n[i].xc) > 0
            for j in 1:length(n[i].xc)
              write(f, " ", string(n[i].xc[j]))
            end
            write(f, "\n")
            for j in 1:length(n[i].yc)
              write(f, " ", string(n[i].yc[j]))
            end
            write(f, "\n")
            for j in 1:length(n[i].zc)
              write(f, " ", string(n[i].zc[j]))            
            end
          else
            write(f, "-\n-\n-")
          end
          write(f, "\n")
          if length(n[i].traj) > 0 && info_level >= 5 # All traj
            for j in 1:length(n[i].traj[:, 1])
              write(f, " ", string(n[i].traj[j, 1]))
            end
            write(f, "\n")
            for j in 1:length(n[i].traj[:, 2])
              write(f, " ", string(n[i].traj[j, 2]))
            end
            write(f, "\n")
            for j in 1:length(n[i].traj[:, 3])
              write(f, " ", string(n[i].traj[j, 3]))
            end
            write(f, "\n")
          elseif length(n[i].traj) > 0 && info_level >= 4 # Only final point
            write(f, " ", string(n[i].traj[-1, 1]))
            write(f, "\n")
            write(f, " ", string(n[i].traj[-1, 2]))
            write(f, "\n")
            write(f, " ", string(n[i].traj[-1, 3]))
            write(f, "\n")
          else
            write(f, string(n[i].x))
            write(f, "\n")
            write(f, string(n[i].y))
            write(f, "\n")
            write(f, string(n[i].z))
            write(f, "\n")
          end
        end 
      end
    elseif info_level == 0
      # Not the best way to store information as a list of
      # nodes, but this is far from the bottleneck of the problem.
      open(filename, "w") do f
        # Initial particle
        write(f, n[1].species, " ", string(n[1].weight), " ",
          string(n[1].prob), " ", string(n[1].parent_weight), " ",
          string(n[1].x),  " ", string(n[1].y),  " ", string(n[1].z),  " ",
          string(n[1].kx), " ", string(n[1].ky), " ", string(n[1].kz), "\n"
         )
        for i in 2:length(n)
          # Only store if it is an outgoing particle
          if length(n[i].xc) == 0
            write(f, n[i].species, " ", string(n[i].weight), " ",
                string(n[i].prob), " ", string(n[i].parent_weight), "\n")
          end
        end
      end
    else
      print("UNKNOWN INFOLEVEL ", info_level, " IN saveTree.")
      print("\nThe data is not stored!!!!! \n")
    end
end

function saveNode(f, n)
  write(f, n.species, " ", string(n.weight), " ",
              string(n.prob), " ", string(n.parent_weight), "\n")
  if length(n.xc) > 0
    for j in 1:length(n.xc)
      write(f, " ", string(n.xc[j]))
    end
    write(f, "\n")
    for j in 1:length(n.yc)
       write(f, " ", string(n.yc[j]))
    end
    write(f, "\n")
    for j in 1:length(n.zc)
       write(f, " ", string(n.zc[j]))            
    end
  else
    write(f, "-\n-\n-")
  end
  write(f, "\n")
  if length(n.traj) > 0
    for j in 1:length(n.traj[:, 1])
      write(f, " ", string(n.traj[j, 1]))
    end
    write(f, "\n")
    for j in 1:length(n.traj[:, 2])
      write(f, " ", string(n.traj[j, 2]))
    end
    write(f, "\n")
    for j in 1:length(n.traj[:, 3])
      write(f, " ", string(n.traj[j, 3]))
    end
    write(f, "\n")
  else
    write(f, string(n.x))
    write(f, "\n")
    write(f, string(n.y))
    write(f, "\n")
    write(f, string(n.z))
    write(f, "\n")
  end
end

function get_Prob_nonAD(pos::Array, kpos::Array,
    Mass_a,Ax_g,θm,ωPul,B0,rNS,erg_inf_ini,vIfty_mag)

  Nc = length(pos[:, 1])
  rmag = sqrt.(sum(pos.^ 2, dims=2));
  vmag = sqrt.(2 * GNew .* Mass_NS ./ rmag) ; # km/s 
  vmag_tot = sqrt.(vmag .^ 2 .+ vIfty_mag.^2); # km/s
  Bvec, ωp = RT.GJ_Model_vec(pos, zeros(Nc), θm, ωPul, B0, rNS);
  Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
  newV = kpos
  cθ = sum(newV .* Bvec, dims=2) ./ Bmag
 
  erg_ax = erg_inf_ini ./ sqrt.(1.0 .- 2 * GNew .* Mass_NS ./ rmag ./
                                c_km.^2 );
  B_tot = Bmag .* (1.95e-20) ; # GeV^2
  
  MagnetoVars =  [θm, ωPul, B0, rNS, [1.0 1.0], zeros(Nc), erg_ax]
  sln_δk = RT.dk_ds(pos, kpos, [func_use, MagnetoVars]);
  conversion_F = sln_δk ./  (6.58e-16 .* 2.998e5) # 1/km^2;

  Prob_nonAD = (π ./ 2 .* (Ax_g .* B_tot) .^2 ./ conversion_F .*
          (1e9 .^2) ./ (vmag_tot ./ 2.998e5) .^2 ./
          ((2.998e5 .* 6.58e-16) .^2) ./ sin.(acos.(cθ)).^4) #unitless

end

function get_tree(first::RT.node, erg_inf_ini, vIfty_mag,
    Mass_a,Ax_g,θm,ωPul,B0,rNS,Mass_NS,gammaF,flat,isotropic,melrose,
    NumerPass; num_cutoff=5, prob_cutoff=1e-10,splittings_cutoff=-1,
    ax_num=100)

  # Accuracy parameters 
  # -------------------
  #tot_prob_cutoff    = 1 - 1e- # Cutoff total probability
  #prob_cutoff        = 1e-100    # Cutoff probability for single photons
  #splittings_cutoff  = -1       # Max number of splittings for each particles
                                # If negative: one splitting but stores the
                                # original as well to be rerun later
  #num_cutoff         = 2000     # Max number of total particles (must be large!)
  #num_main           = 5       # Max number of escaped main branch particles
  # ^^^^^^^^^^^^^^^^^^^

  #splittings_cutoff = -1 # The code is no longer optimal for a different number

  # Initial conversion probability
  pos = [first.x first.y first.z]
  kpos = [first.kx first.ky first.kz]
  Prob_nonAD = get_Prob_nonAD(pos,kpos,Mass_a,Ax_g,θm,ωPul,B0,rNS,
                                  erg_inf_ini, vIfty_mag)
  Prob = 1 .- exp.(-Prob_nonAD)
  first.prob = Prob[1]

  batchsize = 1 # Only one parent photon

  events = [first]
  tree = []

  tot_prob = 0 # Total probability in tree

  count = -1
  count_main = 0
  
  #DEBUG
  print("Initial conversion probability: ", Prob, "\n")
  print("prob_cutoff: ", prob_cutoff, "\n")
  print("num_cutoff: ", num_cutoff, "\n")

  while length(events) > 0
    
    count += 1
    
    event = last(events)
    pop!(events)
    
    pos0 = [event.x event.y event.z]
    k0 = [event.kx event.ky event.kz]

    # DEBUG
    print(count_main, " ", event.species, " ", event.weight, " ",
          tot_prob, " ", sum(pos0.^2), "\n")

    # propagate photon or axion
    if event.species == "photon"
      Mvars = [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS,
               [erg_inf_ini], flat, isotropic, melrose]
      x_e, k_e, t_e, err_e, cut_short, xc, yc, zc, kxc, kyc, kzc = RT.propagate(
                      func_use_SPHERE, pos0, k0,
                      ax_num, Mvars, NumerPass, RT.func!,
                      true, false, Mass_a, splittings_cutoff)
    else      
      Mvars = [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS,
               [erg_inf_ini], flat, isotropic, melrose, Mass_a]
      x_e, k_e, t_e, err_e, cut_short, xc, yc, zc, kxc, kyc, kzc = RT.propagate(
                        func_use_SPHERE, pos0, k0,
                        ax_num, Mvars, NumerPass, RT.func_axion!,
                        true, true, Mass_a, splittings_cutoff)
    end
    pos = transpose(x_e[1, :, :])
    kpos = transpose(k_e[1, :, :])
    event.traj = pos


    if length(xc) < 1  # No crossings
        # Since we are considering the most probable first
        count_main += 1
        tot_prob += event.weight
    else

      if any(abs.(kxc) .> 1) || any(abs.(kyc) .> 1) || any(abs.(kzc) .> 1)
        print("A rare fail occured, and I do not know why...\n")
        print("   xc:   ", xc, "\n")
        print("   yc:   ", yc, "\n")
        print("   zc:   ", zc, "\n")
        print("   kxc:  ", kxc, "\n")
        print("   kyc:  ", kyc, "\n")
        print("   kzc:  ", kzc, "\n")
        push!(tree, event)
        tot_prob += event.weight
        continue
      end

      # If two crossings are close, it is likely only one crossing
      # This happens (likely only) close to the neutron star surface
      if length(xc) > 1
        epsabs = 1e-4 # ... as ode_err 
        r = sqrt.(xc.^2 + yc.^2 + zc.^2)
        tmp = sqrt.(abs.(diff(xc)).^2 .+ abs.(diff(yc)).^2 .+ abs.(diff(zc)).^2)
        if any(tmp .< epsabs)
          flag = push!((tmp .> epsabs), 1)
          print("Two crossings occur at the same point. Deleting one of them\n")
          print(sqrt.(xc.^2 + yc.^2 + zc.^2), "\n")
          xc = xc[flag]
          yc = yc[flag]
          zc = zc[flag]
          kxc = kxc[flag]
          kyc = kyc[flag]
          kzc = kzc[flag]
          print(sqrt.(xc.^2 + yc.^2 + zc.^2), "\n")
        end
      end

      # store level crossings
      Nc = length(xc)
      pos = zeros(Nc, 3);
      pos[:, 1] .= xc; pos[:, 2] = yc; pos[:, 3] = zc
      kpos = zeros(Nc, 3);
      kpos[:, 1] .= kxc; kpos[:, 2] .= kyc; kpos[:, 3] .= kzc
      event.xc = xc
      event.yc = yc
      event.zc = zc
      event.kxc = kxc
      event.kyc = kyc
      event.kzc = kzc

      # Conversion probability
      Prob_nonAD = get_Prob_nonAD(pos,kpos,Mass_a,Ax_g,θm,ωPul,B0,rNS,
                                  erg_inf_ini, vIfty_mag)
      Prob = 1 .- exp.(-Prob_nonAD)
      event.Pc = Prob

      # Find "ID" of new particle              
      if event.species == "photon"
        new_species = "axion"
      else
        new_species = "photon"
      end

      # Add all crossings to the tree
      print("Number of crossings: ", Nc, "\n") #DEBUG
      for j in 1:Nc 
        #if Prob[j]*event.weight > prob_cutoff # Cutoff
          push!(events, RT.node(xc[j], yc[j], zc[j], kxc[j], kyc[j],
                    kzc[j], new_species, Prob[j],
                    Prob[j]*event.weight, event.weight,[],[],[],[],[],[],[],[]))
          if splittings_cutoff <= 0 
            push!(events, RT.node(xc[j], yc[j], zc[j], kxc[j], kyc[j],
                kzc[j], event.species, 1-Prob[j],
                (1-Prob[j])*event.weight, event.weight,[],[],[],[],[],[],[],[]))
          else
        #end
            # Re-evaluate weight of parent
            event.weight = event.weight*(1-Prob[j])
          end
      end
      
      if splittings_cutoff > 0 # Only final particles should count
        tot_prob += event.weight
      end

    end

    # Add to tree
    push!(tree, event)

    # Cutoff
    if tot_prob >= 1 - prob_cutoff
      break
    end
    if num_cutoff <= 0 && splittings_cutoff > 0
      break
    end
    if count_main >= num_cutoff
      break
    end

    # Sort events to consider the most likely first
    sort!(events, by = events->events.weight)

  end

  return tree

end


function main_runner_tree(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, ωProp,
    Ntajs, gammaF, batchsize; flat=true, isotropic=false, melrose=false,
    ode_err=1e-5, cutT=100000, fix_time=Nothing, CLen_Scale=true, file_tag="",
    ntimes=1000, v_NS=[0 0 0], rho_DM=0.3, save_more=false, vmean_ax=220.0,
    ntimes_ax=10000, dir_tag="results", iseed=-1, num_cutoff=5,
    prob_cutoff=1e-10, info_level=5)

    if iseed < 0
      iseed = rand(0:1000000)
      print("Using seed ", iseed, "\n")
      Random.seed!(iseed)
    elseif iseed == 0
      Random.seed!()
    else
      print("Using seed ", iseed, "\n")
      Random.seed!(iseed)
    end


    # Identify the maximum distance of the conversion surface from NS
    maxR = RT.Find_Conversion_Surface(Mass_a, fix_time, θm, ωPul, B0,
                                      rNS, 1, false)
    maxR_tag = "";

    # check if NS allows for conversion
    if maxR < rNS
        print("Too small Max R.... quitting.... \n")
        omegaP_test = RT.GJ_Model_ωp_scalar(rNS .* [sin.(θm) 0.0 cos.(θm)], 
                                            0.0, θm, ωPul, B0, rNS);
        print("Max omegaP found... \t", omegaP_test,
              "Max radius found...\t", maxR, "\n")
        return
    end


    photon_trajs = 1
    desired_trajs = Ntajs
    f_inx = 0;

    # define arrays that are used in surface area sampling
    tt_ax = LinRange(-2*maxR, 2*maxR, ntimes_ax); # Not a real physical time
    t_diff = tt_ax[2] - tt_ax[1];
    tt_ax_zoom = LinRange(-2*t_diff, 2*t_diff, ntimes_ax);

    # define min and max time to propagate photons
    ln_t_start = -22;
    ln_t_end = log.(1 ./ ωPul);
    NumerPass = [ln_t_start, ln_t_end, ode_err];
    ttΔω = exp.(LinRange(ln_t_start, ln_t_end, ntimes));

    if (v_NS[1] == 0)&&(v_NS[1] == 0)&&(v_NS[1] == 0)
        phaseApprox = true;
    else
        phaseApprox = false;
    end
    vNS_mag = sqrt.(sum(v_NS.^2));
    if vNS_mag .> 0
        vNS_theta = acos.(v_NS[3] ./ vNS_mag);
        vNS_phi = atan.(v_NS[2], v_NS[1]);
    end
    
    # init some arrays
    t0_ax = zeros(batchsize);
    xpos_flat = zeros(batchsize, 3);
    R_sample = zeros(batchsize);
    mcmc_weights = zeros(batchsize);
    filled_positions = false;
    fill_indx = 1;
    Ncx_max = 1;
    
    while photon_trajs < desired_trajs
        
        # First part of code here is just written to generate evenly spaced
        # samples of conversion surface
        while !filled_positions
            xv, Rv, numV, weights = RT.find_samples(maxR, ntimes_ax, θm, ωPul,
                                                    B0, rNS, Mass_a, Mass_NS)
            f_inx += 2;
            
            if numV == 0
                continue
            end
            
            # for i in 1:1
            for i in 1:numV # Keep more?
                if weights[i] .> Ncx_max
                    Ncx_max = weights[i]
                end
                if fill_indx <= batchsize

                    xpos_flat[fill_indx, :] .= xv[i, :];
                    R_sample[fill_indx] = Rv[i];
                    mcmc_weights[fill_indx] = weights[i];
                    fill_indx += 1
                    
                end
            end
            
            if fill_indx > batchsize
                filled_positions = true
                fill_indx = 1
                f_inx -= 1;
            end
        end
        #    and not interpolation of the path
        filled_positions = false;
        
        rmag = sqrt.(sum(xpos_flat.^ 2, dims=2));
        vmag = sqrt.(2 * GNew .* Mass_NS ./ rmag) ; # km/s
        
        # resample angle (this will be axion velocity at conversion surface)
        θi = acos.(1.0 .- 2.0 .* rand(length(rmag)));
        ϕi = rand(length(rmag)) .* 2π;
        newV = [sin.(θi) .* cos.(ϕi) sin.(θi) .* sin.(ϕi) cos.(θi)];
        
        
        
        # define angle between surface normal and velocity
        calpha = RT.surfNorm(xpos_flat, newV, [func_use, [θm, ωPul, B0, rNS,
                  gammaF, zeros(batchsize), Mass_NS]], return_cos=true); # alpha
        weight_angle = abs.(calpha);

        # sample asymptotic velocity
        vIfty=erfinv.(2 .* rand(length(vmag),3) .- 1.) .* vmean_ax .+ v_NS#km/s
        vIfty_mag = sqrt.(sum(vIfty.^2, dims=2));
        vel_eng = sum((vIfty ./ 2.998e5).^ 2, dims = 2) ./ 2;
        gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ./ c_km).^2 )
        erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag ./ c_km .* gammaA).^2)
        
        # define initial momentum (magnitude)
        k_init = RT.k_norm_Cart(xpos_flat, newV,  0.0, erg_inf_ini, θm, ωPul,
              B0, rNS, Mass_NS, melrose=melrose, isotropic=isotropic, flat=flat)
        MagnetoVars = [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS,
                       erg_inf_ini, flat, isotropic, melrose]
                   # θm, ωPul, B0, rNS, gamma factors, Time = 0, mass_ns, erg ax
        


        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # to-do: 
        #  - Check physics of EoM of axion
        
        for i in 3:batchsize
          fname = dir_tag * "/tree_" * file_tag * string(photon_trajs) 
          f = open(fname, "w")


          print(photon_trajs, " backward in time\n------------------\n") # DEBUG
          # Find previous crossings...
          # Backwards in time equivalent to setting k->-k and vecB->-vecB
          parent = RT.node(xpos_flat[i, 1], xpos_flat[i, 2], xpos_flat[i, 3],
                -k_init[i, 1], -k_init[i, 2], -k_init[i, 3],
                "axion", 1.0, 1.0, -1.0, [],[],[],[],[],[],[],[])
          # The simplest is always the best: make use of existing code
          nb = get_tree(parent,erg_inf_ini[i],vIfty_mag[i],
                Mass_a,Ax_g,θm,ωPul,-B0,rNS,Mass_NS,gammaF,
                flat,isotropic,melrose,NumerPass;prob_cutoff=prob_cutoff,
                num_cutoff=0, splittings_cutoff=100000, ax_num=100)[1]
          saveNode(f, nb)

          print(photon_trajs, " forward in time\n--------------------\n") #DEBUG
          # Forward propagation of a photon from the last node
          if length(nb.xc) == 0
            nb.xc = [xpos_flat[i, 1]]
            nb.yc = [xpos_flat[i, 2]]
            nb.zc = [xpos_flat[i, 3]]
            nb.kxc = [-k_init[i, 1]]
            nb.kyc = [-k_init[i, 2]] # "-" since it was backtraced
            nb.kzc = [-k_init[i, 3]]
            nb.Pc  = [nb.prob]
          end
          species = ["axion*" "photon"]
          probs = [1 - nb.Pc[end], nb.Pc[end]]
          for j in [1 2]
            if probs[j] > prob_cutoff # Skip if unlikely
              parent = RT.node( nb.xc[end],   nb.yc[end],   nb.zc[end],
                          -nb.kxc[end], -nb.kyc[end], -nb.kzc[end], # to forward
                          species[j], probs[j], probs[j], 1.0,
                          [],[],[],[],[],[],[],[])
              tree = get_tree(parent,erg_inf_ini[i],vIfty_mag[i],
                  Mass_a,Ax_g,θm,ωPul,B0,rNS,Mass_NS,gammaF,
                  flat,isotropic,melrose,NumerPass;prob_cutoff=prob_cutoff,
                  num_cutoff=num_cutoff,ax_num=100)
              for ii in 1:length(tree)
                saveNode(f, tree[ii])
              end
            end
          end

          photon_trajs += 1

          close(f)
        end


        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    end
end


function single_runner(species, x0, y0, z0, kx0, ky0, kz0,
    Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, ωProp,
    Ntajs, gammaF, batchsize; flat=true, isotropic=false, melrose=false,
    ode_err=1e-5, cutT=100000, fix_time=Nothing, CLen_Scale=true, file_tag="",
    ntimes=1000, v_NS=[0 0 0], rho_DM=0.3, save_more=false, vmean_ax=220.0,
    ntimes_ax=10000, dir_tag="results", iseed=-1, num_cutoff=5,
    prob_cutoff=1e-10, info_level=5,
    forwards=true)

    rmag = sqrt(x0^2 + y0^2 + z0^2)
    vmag = sqrt.(2 * GNew .* Mass_NS ./ rmag) ; # km/s
    
    # resample angle (this will be axion velocity at conversion surface)
    #θi = acos.(1.0 .- 2.0 .* rand(length(rmag)));
    #ϕi = rand(length(rmag)) .* 2π;
    #newV = [sin.(θi) .* cos.(ϕi) sin.(θi) .* sin.(ϕi) cos.(θi)];
    
    # define angle between surface normal and velocity
    #calpha = RT.surfNorm(xpos_flat, newV, [func_use, [θm, ωPul, B0, rNS,
    #          gammaF, zeros(batchsize), Mass_NS]], return_cos=true); # alpha
    #weight_angle = abs.(calpha);

    # sample asymptotic velocity
    vIfty=erfinv.(2 .* rand(length(vmag),3) .- 1.) .* vmean_ax .+ v_NS#km/s
    vIfty_mag = sqrt.(sum(vIfty.^2));
    #vel_eng = sum((vIfty ./ 2.998e5).^ 2, dims = 2) ./ 2;
    gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ./ c_km).^2 )
    erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag ./ c_km .* gammaA).^2)
        
    # define min and max time to propagate photons
    ln_t_start = -22;
    ln_t_end = log.(1 ./ ωPul);
    NumerPass = vec([ln_t_start, ln_t_end, ode_err])

    # Normalise k
    norm = sqrt(kx0^2 + ky0^2 + kz0^2)
    kx0 = kx0/norm
    ky0 = ky0/norm
    kz0 = kz0/norm

    if forwards
      parent = RT.node(x0, y0, z0, kx0, ky0, kz0,
            species, 1.0, 1.0, -1.0, [], [], [], [])
      tree = get_tree(parent,erg_inf_ini,vIfty_mag,
            Mass_a,Ax_g,θm,ωPul,B0,rNS,Mass_NS,gammaF,
            flat,isotropic,melrose,NumerPass;prob_cutoff=prob_cutoff,
            num_cutoff=num_cutoff)
      saveTree(tree, dir_tag * "/forward_" * file_tag * "single",
               info_level)
    else
      parent = RT.node(x0, y0, z0, kx0, ky0, kz0,
            species, 1.0, 1.0, -1.0, [], [], [], [])
      tree_backwards = get_tree(parent,erg_inf_ini,vIfty_mag,
            Mass_a,Ax_g,θm,ωPul,-B0,rNS,Mass_NS,gammaF,
            flat,isotropic,melrose,NumerPass;prob_cutoff=prob_cutoff,
            num_cutoff=num_cutoff)
      saveTree(tree_backwards,
               dir_tag * "/backward_" * file_tag * "single",
              info_level)
    end
end
