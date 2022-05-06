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

function saveTree(n::Array, filename::String="tree.txt")
    open(filename, "w") do f
      for i in 1:length(n)
        write(f, n[i].species, " ", string(n[i].weight), " ",
              string(n[i].prob), " ",
              string(n[i].parent_weight), "\n")
        if length(n[i].level_crossings_x) > 0
          for j in 1:length(n[i].level_crossings_x)
            write(f, " ", string(n[i].level_crossings_x[j]))
          end
          write(f, "\n")
          for j in 1:length(n[i].level_crossings_y)
            write(f, " ", string(n[i].level_crossings_y[j]))
          end
          write(f, "\n")
          for j in 1:length(n[i].level_crossings_z)
            write(f, " ", string(n[i].level_crossings_z[j]))            
          end
        else
          write(f, "-\n-\n-")
        end
        write(f, "\n")
        if length(n[i].traj) > 0
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
    NumerPass)

  # Accuracy parameters 
  # -------------------
  tot_prob_cutoff    = 1 - 1e-10 # Cutoff total probability
  prob_cutoff        = 1e-100    # Cutoff probability for single photons
  splittings_cutoff  = -1       # Max number of splittings for each particles
                                # If negative: one splitting but stores the
                                # original as well to be rerun later
  num_cutoff         = 2000     # Max number of total particles (must be large!)
  num_main           = 100       # Max number of escaped main branch particles
  # ^^^^^^^^^^^^^^^^^^^

  # Initial conversion probability
  pos = [first.x first.y first.z]
  kpos = [first.kx first.ky first.kz]
  Prob_nonAD = get_Prob_nonAD(pos,kpos,Mass_a,Ax_g,θm,ωPul,B0,rNS,
                                  erg_inf_ini, vIfty_mag)
  Prob = exp.(-Prob_nonAD)
  first.prob = Prob[1]

  batchsize = 1 # Only one parent photon

  events = [first]
  tree = []

  tot_prob = 0 # Total probability in tree

  count = -1
  count_main = 0
  
  #DEBUG
  print("Initial conversion probability: ", Prob, "\n")

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
                      1000, Mvars, NumerPass, RT.func!,
                      true, false, Mass_a, splittings_cutoff)
    else      
      Mvars = [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS,
               [erg_inf_ini], flat, isotropic, melrose, Mass_a]
      x_e, k_e, t_e, err_e, cut_short, xc, yc, zc, kxc, kyc, kzc = RT.propagate(
                        func_use_SPHERE, pos0, k0,
                        1000, Mvars, NumerPass, RT.func_axion!,
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

      # find level crossings OLD
      #logdiff = (log.(RT.GJ_Model_ωp_vec(pos, 0.0, θm, ωPul, B0, rNS))
      #           .- log.(Mass_a))
      #cxing_st = RT.get_crossings(logdiff)
      #xpos = RT.apply(cxing_st,  pos[:, 1])
      #ypos = RT.apply(cxing_st,  pos[:, 2])
      #zpos = RT.apply(cxing_st,  pos[:, 3])
      #kx = RT.apply(cxing_st,  kpos[:, 1])
      #ky = RT.apply(cxing_st,  kpos[:, 2])
      #kz = RT.apply(cxing_st,  kpos[:, 3])
      
      Nc = length(xc)
      pos = zeros(Nc, 3);
      pos[:, 1] .= xc; pos[:, 2] = yc; pos[:, 3] = zc
      kpos = zeros(Nc, 3);
      kpos[:, 1] .= kxc; kpos[:, 2] .= kyc; kpos[:, 3] .= kzc
      
      event.level_crossings_x = xc
      event.level_crossings_y = yc
      event.level_crossings_z = zc


      # Conversion probability
      Prob_nonAD = get_Prob_nonAD(pos,kpos,Mass_a,Ax_g,θm,ωPul,B0,rNS,
                                  erg_inf_ini, vIfty_mag)
      Prob = exp.(-Prob_nonAD)

      # Find "ID" of new particle              
      if event.species == "axion"
        new_species = "photon"
      elseif event.species == "photon"
        new_species = "axion"
      end

      # Add all crossings to the tree
      for j in 1:Nc 
        if Prob[j]*event.weight > prob_cutoff # Cutoff
          push!(events, RT.node(xc[j], yc[j], zc[j], kxc[j], kyc[j],
                    kzc[j], new_species, Prob[j],
                    Prob[j]*event.weight, event.weight, [], [], [], []))
          if splittings_cutoff <= 0 
            push!(events, RT.node(xc[j], yc[j], zc[j], kxc[j], kyc[j],
                    kzc[j], event.species, 1-Prob[j],
                    (1 - Prob[j])*event.weight, event.weight, [], [], [], []))
          end
        end

        # Re-evaluate weight of parent
        event.weight = event.weight*(1-Prob[j])
      end
      
      if splittings_cutoff > 0 # Only final particles should count
        tot_prob += event.weight
      end

    end

    # Add to tree
    push!(tree, event)

    # Cutoff
    if tot_prob >= tot_prob_cutoff
      break
    end
    if count >= num_cutoff
      break
    end
    if count_main >= num_main 
      break
    end

    # Sort events to consider the most likely first
    sort!(events, by = events->events.weight)

  end

  return tree

end



function main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, ωProp, Ntajs, gammaF, batchsize; flat=true, isotropic=false, melrose=false, ode_err=1e-5, cutT=100000, fix_time=Nothing, CLen_Scale=true, file_tag="", ntimes=1000, v_NS=[0 0 0], rho_DM=0.3, save_more=false, vmean_ax=220.0, ntimes_ax=10000, dir_tag="results", iseed=-1)

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


    # axion mass [eV], axion-photon coupling [1/GeV], misalignment angle (rot-B field) [rad], rotational freq pulars [1/s]
    # magnetic field strengh at surface [G], radius NS [km], mass NS [solar mass], dispersion relations
    # number of axion trajectories to generate
    
    # This next part is out-dated and irrelevant (haven't removed because i have to re-write functions)
    # ~~~~~~~~~
    



    # Identify the maximum distance of the conversion surface from NS
    maxR = RT.Find_Conversion_Surface(Mass_a, fix_time, θm, ωPul, B0, rNS, 1, false)
    maxR_tag = "";

    # check if NS allows for conversion
    if maxR < rNS
        print("Too small Max R.... quitting.... \n")
        omegaP_test = RT.GJ_Model_ωp_scalar(rNS .* [sin.(θm) 0.0 cos.(θm)], 0.0, θm, ωPul, B0, rNS);
        print("Max omegaP found... \t", omegaP_test, "Max radius found...\t", maxR, "\n")
        return
    end


    photon_trajs = 1
    desired_trajs = Ntajs
    # assumes desired_trajs large!
    save_more=true;
    if save_more
        SaveAll = zeros(desired_trajs * 2, 18);
    else
        SaveAll = zeros(desired_trajs * 2, 11);
    end
    f_inx = 0;

    # define arrays that are used in surface area sampling
    tt_ax = LinRange(-2*maxR, 2*maxR, ntimes_ax); # Not a real physical time -- just used to get trajectory crossing
    t_diff = tt_ax[2] - tt_ax[1];
    tt_ax_zoom = LinRange(-2*t_diff, 2*t_diff, ntimes_ax);

    # define min and max time to propagate photons
    ln_t_start = -22;
    ln_t_end = log.(1 ./ ωPul);
    NumerPass = [ln_t_start, ln_t_end, ode_err];
    ttΔω = exp.(LinRange(ln_t_start, ln_t_end, ntimes));

    if fix_time != Nothing
        file_tag *= "_fixed_time_"*string(fix_time);
    end

    file_tag *= "_odeErr_"*string(ode_err);
    
    file_tag *= "_vxNS_"*string(v_NS[1]);
    file_tag *= "_vyNS_"*string(v_NS[2]);
    file_tag *= "_vzNS_"*string(v_NS[3]);
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
        # First part of code here is just written to generate evenly spaced samples of conversion surface
        while !filled_positions
            xv, Rv, numV, weights = RT.find_samples(maxR, ntimes_ax, θm, ωPul, B0, rNS, Mass_a, Mass_NS)
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
        calpha = RT.surfNorm(xpos_flat, newV, [func_use, [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS]], return_cos=true); # alpha
        weight_angle = abs.(calpha);

        # sample asymptotic velocity
        vIfty = erfinv.(2 .* rand(length(vmag), 3) .- 1.0) .* vmean_ax .+ v_NS # km /s
        vIfty_mag = sqrt.(sum(vIfty.^2, dims=2));
        vel_eng = sum((vIfty ./ 2.998e5).^ 2, dims = 2) ./ 2;
        gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ./ c_km).^2 )
        erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag ./ c_km .* gammaA).^2)
        
        # define initial momentum (magnitude)
        k_init = RT.k_norm_Cart(xpos_flat, newV,  0.0, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, melrose=melrose)
        MagnetoVars = [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS, erg_inf_ini, flat, isotropic, melrose] # θm, ωPul, B0, rNS, gamma factors, Time = 0, mass_ns, erg ax
        


        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # to-do: 
        #  - reduced effective mass of NS when the axion passes through
        #  - Check physics of EoM backwards in time
        #  - Check physics of EoM of axion
        for i in 1:batchsize
          parent = RT.node(xpos_flat[i, 1], xpos_flat[i, 2], xpos_flat[i, 3],
                k_init[i, 1], k_init[i, 2], k_init[i, 3],
                "photon", 1.0, 1.0, -1.0, [], [], [], [])
                               # Parent weight: -1 indicates first
                               # The last 7 elements are updated in "get_tree"
          print(i, " forward in time\n---------------------\n")
          tree = get_tree(parent,erg_inf_ini[i],vIfty_mag[i],
                Mass_a,Ax_g,θm,ωPul,B0,rNS,Mass_NS,gammaF,
                flat,isotropic,melrose,NumerPass)
          printTree(tree)
          saveTree(tree, "results/forward_" * string(i))



          print(i, " backward in time\n---------------------\n")
          # Backwards in time equivalent to setting k->-k and vecB->-vecB (???)
          parent = RT.node(xpos_flat[i, 1], xpos_flat[i, 2], xpos_flat[i, 3],
                -k_init[i, 1], -k_init[i, 2], -k_init[i, 3],
                "axion", 1.0, 1.0, -1.0, [], [], [], [])
          tree_backwards = get_tree(parent,erg_inf_ini[i],vIfty_mag[i],
                Mass_a,Ax_g,θm,ωPul,-B0,rNS,Mass_NS,gammaF,
                flat,isotropic,melrose,NumerPass)
          printTree(tree_backwards)    
          saveTree(tree_backwards, "results/backward_" * string(i))

        end
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # send to ray tracer
        # note, some rays pass through NS, these get removed internally (so need to redefine some stuff)
        xF, kF, tF, fail_indx = RT.propagate(func_use_SPHERE, xpos_flat, k_init, ntimes, MagnetoVars, NumerPass);
      
        vmag_tot = sqrt.(vmag .^ 2 .+ vIfty_mag.^2); # km/s
        Bvec, ωp = RT.GJ_Model_vec(xpos_flat, zeros(batchsize), θm, ωPul, B0, rNS);
        Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
        cθ = sum(newV .* Bvec, dims=2) ./ Bmag

        erg_ax = erg_inf_ini ./ sqrt.(1.0 .- 2 * GNew .* Mass_NS ./ rmag ./ c_km.^2 );
        B_tot = Bmag .* (1.95e-20) ; # GeV^2
        
        MagnetoVars =  [θm, ωPul, B0, rNS, [1.0 1.0], zeros(batchsize), erg_ax]
        sln_δk = RT.dk_ds(xpos_flat, k_init, [func_use, MagnetoVars]);
        conversion_F = sln_δk ./  (6.58e-16 .* 2.998e5) # 1/km^2;
        

        # compute conversion prob
        Prob_nonAD = π ./ 2 .* (Ax_g .* B_tot) .^2 ./ conversion_F .* (1e9 .^2) ./ (vmag_tot ./ 2.998e5) .^2 ./ ((2.998e5 .* 6.58e-16) .^2) ./ sin.(acos.(cθ)).^4; #unitless
        Prob = (1.0 .- exp.(-Prob_nonAD));

        # phase space factors, first assumes vNS = 0, second more general but needs more samples
        if phaseApprox
            phaseS = (π .* maxR .* R_sample .* 2.0) .* rho_DM .* Prob ./ Mass_a .* (vmag_tot ./ c_km) .^ 2 ./ sqrt.(sum( (vIfty ./ c_km) .^ 2, dims=2))
        else
            phaseS = (π .* maxR .* R_sample .* 2.0) .* rho_DM .* Prob ./ Mass_a
            # vmag is vmin [km/s]
            # vmag_tot is v [km/s]
            rhat = xpos_flat ./ sqrt.(sum(xpos_flat.^2, dims=2));
            vnear = vvec_flat .* vmag_tot;
            vinf_mag = sqrt.(sum( (vIfty) .^ 2, dims=2));
            vinf_gf = (vinf_mag.^2 .* vnear .+ vinf_mag .* vmag.^2 ./ 2 .* rhat .- vinf_mag .* vnear .* sum(vnear .* rhat, dims=2)) ./ (vinf_mag.^2 .+ vmag.^2 ./ 2 .- vinf_mag .* sum(vnear .* rhat, dims=2))

            phaseS .*= vmag_tot.^2 .* exp.(- sum((vinf_gf .- v_NS).^2, dims=2) ./ vmean_ax.^2 ) ./ (sqrt.(vmag_tot.^2 .- vmag.^2) .* exp.(- sum((vIfty .- v_NS).^2, dims=2) ./ vmean_ax.^2)) ./ c_km
        end
        
        sln_prob = weight_angle .* phaseS .* (1e5 .^ 2) .* c_km .* 1e5 .* mcmc_weights .* fail_indx ; # photons / second

        # archaic re definition from old feature that has been removed
        sln_k = k_init;
        sln_x = xpos_flat;
        sln_vInf = vel_eng ;
        sln_t = zeros(batchsize);
        sln_ConVL = sqrt.(π ./ conversion_F);


        # extract final angle in sky and photon direction
        ϕf = atan.(view(kF, :, 2, ntimes), view(kF, :, 1, ntimes));
        ϕfX = atan.(view(xF, :, 2, ntimes), view(xF, :, 1, ntimes));
        θf = acos.(view(kF, :, 3, ntimes) ./ sqrt.(sum(view(kF, :, :, ntimes) .^2, dims=2)));
        θfX = acos.(view(xF, :, 3, ntimes) ./ sqrt.(sum(view(xF, :, :, ntimes) .^2, dims=2)));

        # compute energy dispersion (ωf - ωi) / ωi
        MagnetoVars =  [θm, ωPul, B0, rNS, [1.0 1.0], zeros(batchsize)]
        passA = [func_use, MagnetoVars];
        Δω = tF[:, end] ./ Mass_a .+ vel_eng[:];
        
        # comptue optical depth, for now not needed
#        opticalDepth = RT.tau_cyc(xF, kF, ttΔω, passA, Mass_a);
        opticalDepth = zeros(length(sln_prob))

        # should we apply Lc cut?
        num_photons = length(ϕf)
        passA2 = [func_use, MagnetoVars, Mass_a];
        # this is hand set to off for now
        CLen_Scale = false
        if CLen_Scale
            weightC = ConvL_weights(xF, kF, vmag_tot ./ c_km, ttΔω, sln_ConVL, passA2)
        else
            weightC = ones(num_photons)
        end
        
        # cut out spurious features from high Lc cut
        weightC[weightC[:] .> 1.0] .= 1.0;
        
        # Save info
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 1] .= view(θf, :); # final momentum theta
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 2] .= view(ϕf,:); # final momentum phi
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 3] .= view(θfX, :); # final position theta
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 4] .= view(ϕfX, :); # final position phi
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 5] .= sqrt.(sum(xF[:, :, end] .^2, dims=2))[:]; # final distance NS
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 6] .= sln_prob[:] .* weightC .^ 2 .* exp.(-opticalDepth[:]); #  num photons / second
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 7] .= Δω[:]; # (ωf - ωi) / ωi
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 8] .= sln_ConVL[:]; # conversion length
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 9] .= xpos_flat[:, 1]; # initial x
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 10] .= xpos_flat[:, 2]; # initial y
        SaveAll[photon_trajs:photon_trajs + num_photons - 1, 11] .= xpos_flat[:, 3]; # initial z
        if save_more
            SaveAll[photon_trajs:photon_trajs + num_photons - 1, 12] .= k_init[:, 1]; # initial kx
            SaveAll[photon_trajs:photon_trajs + num_photons - 1, 13] .= k_init[:, 2]; # initial ky
            SaveAll[photon_trajs:photon_trajs + num_photons - 1, 14] .= k_init[:, 3]; # initial kz
            SaveAll[photon_trajs:photon_trajs + num_photons - 1, 15] .= opticalDepth[:]; # optical depth
            SaveAll[photon_trajs:photon_trajs + num_photons - 1, 16] .= weightC[:]; # Lc weight
            SaveAll[photon_trajs:photon_trajs + num_photons - 1, 17] .= Prob[:]; # optical depth
            SaveAll[photon_trajs:photon_trajs + num_photons - 1, 18] .= calpha[:]; # surf norm angle
        end
        
        photon_trajs += num_photons;
        
        
        GC.gc();
    end

    
    # cut out unused elements
    SaveAll = SaveAll[SaveAll[:,6] .> 0, :];
    SaveAll[:,6] ./= (float(f_inx) .* float(Ncx_max)); # divide off by N trajectories sampled

    fileN = "results/Fast_Trajectories_MassAx_"*string(Mass_a)*"_AxionG_"*string(Ax_g)*"_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0);
    fileN *= "_Ax_trajs_"*string(Ntajs);
    fileN *= "_N_Times_"*string(ntimes)*"_"*file_tag*"_.npz";
    npzwrite(fileN, SaveAll)

end
