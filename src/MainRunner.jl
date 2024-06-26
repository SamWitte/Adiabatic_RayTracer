__precompile__()

RT = RayTracerGR; # define ray tracer module


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

function saveNode(f, n)
  write(f, n.species, " ", string(n.weight), " ",
              string(n.prob), " ", string(n.parent_weight), "\n")
  if length(n.xc) > 0
    for j in 1:length(n.xc)
      write(f, "  ", string(n.xc[j]))
    end
    write(f, "\n")
    for j in 1:length(n.yc)
       write(f, "  ", string(n.yc[j]))
    end
    write(f, "\n")
    for j in 1:length(n.zc)
       write(f, "  ", string(n.zc[j]))            
    end
    write(f, "\n")
    for j in 1:length(n.tc)
       write(f, "  ", string(n.tc[j]))            
    end
  else
    write(f, "-\n-\n-")
  end
  write(f, "\n")
  if length(n.traj) > 0
    for j in 1:length(n.traj[:, 1])
      write(f, "  ", string(n.traj[j, 1]))
    end
    write(f, "\n")
    for j in 1:length(n.traj[:, 2])
      write(f, "  ", string(n.traj[j, 2]))
    end
    write(f, "\n")
    for j in 1:length(n.traj[:, 3])
      write(f, "  ", string(n.traj[j, 3]))
    end
    write(f, "\n")
    for j in 1:length(n.times)
      write(f, "  ", string(n.times[j]))
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
    Mass_a,Ax_g,θm,ωPul,B0,rNS,erg_inf_ini,vIfty_mag,flat,isotropic, bndry_lyr)

    Nc = length(pos[:, 1])
    rmag = sqrt.(sum(pos.^ 2, dims=2));
    theta_sf = acos.(pos[:,3] ./ rmag)
    x0_pl = [rmag theta_sf atan.(pos[:,2], pos[:,1])]
    t_start = zeros(Nc)
    g_tt, g_rr, g_thth, g_pp = RT.g_schwartz(x0_pl, Mass_NS);
    Bsphere = RT.GJ_Model_Sphereical(pos, t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat)
    ksphere = RT.k_sphere(pos, kpos, θm, ωPul, B0, rNS, Nc, Mass_NS, Mass_a, flat, bndry_lyr=bndry_lyr)
    Bmag = sqrt.(RT.spatial_dot(Bsphere, Bsphere, Nc, x0_pl, Mass_NS)) .* 1.95e-2; # eV^2
    kmag = sqrt.(RT.spatial_dot(ksphere, ksphere, Nc, x0_pl, Mass_NS));
    ctheta_B = RT.spatial_dot(Bsphere, ksphere, Nc, x0_pl, Mass_NS) .* 1.95e-2 ./ (kmag .* Bmag)
    stheta_B = sin.(acos.(ctheta_B))
    if isotropic
        ctheta_B .*= 0.0
        stheta_B ./= stheta_B
    end
    
    v_group = RT.grad(RT.omega_function(x0_pl, RT.seed(ksphere), t_start, -erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a, flat=flat, iso=isotropic, melrose=true))
    
    v_group[:, 1] ./= g_rr
    v_group[:, 2] ./= g_thth
    v_group[:, 3] ./= g_pp
    vgNorm = sqrt.(RT.spatial_dot(v_group, v_group, Nc, x0_pl, Mass_NS));
        
    erg_ax = erg_inf_ini ./ sqrt.(1.0 .- 2 * GNew .* Mass_NS ./ rmag ./ c_km.^2 );
    Mvars =  [θm, ωPul, B0, rNS, [1.0, 1.0], t_start, Mass_NS, Mass_a, flat, isotropic, erg_ax, bndry_lyr]
    # sln_δwp, sln_δk, sln_δE, cos_w, vgNorm, dk_vg, dE_vg, k_vg = RT.dwp_ds(pos, ksphere,  Mvars)
    # sln_δw, angleVal, k_dot_N, dwdk_snorm, vg_mu, vgN  = RT.dwp_ds(pos, ksphere, Mvars)
    
    ωp = RT.GJ_Model_ωp_vecSPH(x0_pl, t_start, θm, ωPul, B0, rNS, zeroIn=true, bndry_lyr=bndry_lyr)
    local_vars = [ωp, Bsphere, Bmag, ksphere, kmag, vgNorm, ctheta_B, stheta_B]
    Prob_nonAD, sln_δE, cos_w, grad_Emag, cos_w_2, grad_Emag_2 = RT.conversion_prob(Ax_g, x0_pl, Mvars, local_vars, one_D=false)
    # print(Prob_nonAD,"\n")
    conversion_F = sln_δE .*  (hbar .* c_km) # eV^2;
    
    return Prob_nonAD
#    # extra_term = Mass_a.^5 ./ (kmag.^2 .+ Mass_a.^2 .* stheta_B.^2).^2
#    # Prob_nonAD = π ./ 2 .* (Ax_g .* 1e-9 .* Bmag).^2 .* Mass_a.^2 ./ (conversion_F .* kmag.^2); #unitless
    
#    vmag = sqrt.(2 * GNew .* Mass_NS ./ rmag) ; # km/s
#    vmag_tot = sqrt.(vmag .^ 2 .+ vIfty_mag.^2); # km/s
#    Bvec, ωp = RT.GJ_Model_vec(pos, zeros(Nc), θm, ωPul, B0, rNS);
#    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
#    newV = kpos
#    cθ = sum(newV .* Bvec, dims=2) ./ Bmag
    
#    B_tot = Bmag .* (1.95e-20) ; # GeV^2
#    MagnetoVars =  [θm, ωPul, B0, rNS, [1.0 1.0], zeros(Nc), erg_ax]
#    sln_δk = RT.dk_ds(pos, kpos, [ MagnetoVars]);
#    conversion_F = sln_δk ./  (6.58e-16 .* 2.998e5) # 1/km^2;
#    Prob_nonAD = (π ./ 2 .* (Ax_g .* B_tot) .^2 ./ conversion_F .*
#          (1e9 .^2) ./ (vmag_tot ./ 2.998e5) .^2 ./
#          ((2.998e5 .* 6.58e-16) .^2) ./ sin.(acos.(cθ)).^4) #unitless

end

function get_tree(first::RT.node, erg_inf_ini, vIfty_mag,
    Mass_a,Ax_g,θm,ωPul,B0,rNS,Mass_NS,gammaF,flat,isotropic,melrose,bndry_lyr,
    NumerPass_in; num_cutoff=5, prob_cutoff=1e-10,splittings_cutoff=-1,
    ax_num=100, MC_nodes=5, max_nodes=50)

  # Initial conversion probability
  pos = [first.x first.y first.z]
  kpos = [first.kx first.ky first.kz]
  Prob_nonAD = get_Prob_nonAD(pos,kpos,Mass_a,Ax_g,θm,ωPul,B0,rNS,
                      erg_inf_ini .* abs.(first.Δω), vIfty_mag, flat, isotropic, bndry_lyr)
  Prob = 1 .- exp.(-Prob_nonAD)
  first.prob = Prob[1]

  batchsize = 1 # Only one parent photon

  events = [first]
  tree = []

  tot_prob = 0 # Total probability in tree
  # Note: turned of to make bound orbits more likely
  # tot_prob = 1 - first.prob

  count = 0
  count_main = 0
  info = 1
 
  NumerPass = copy(NumerPass_in)
  dt0 = exp(NumerPass[1])

  while length(events) > 0
    
    count += 1
    
    event = last(events)
    pop!(events)
    
    pos0 = [event.x event.y event.z]
    k0 = [event.kx event.ky event.kz]

    Δω = event.Δω[end]      
    NumerPass[1] = log(max(event.t, dt0)) # Time of start

    if (Δω > -0.5 || Δω < -2.0)
      print("The energy is changed by a factor ", -Δω, "... ")
      print("Something is probably wrong!\n")
    end


    # propagate photon or axion
    if event.species == "photon"
      # print("propagate photon \n")
      Mvars = [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS, Mass_a,
               [erg_inf_ini], flat, isotropic, melrose, bndry_lyr]
      x_e,k_e,t_e,err_e,cut_short,xc,yc,zc,kxc,kyc,kzc,tc,Δωc,times = RT.propagate(
                    pos0, k0,
                    ax_num, Mvars, NumerPass, RT.func!,
                    true, false, Mass_a, splittings_cutoff, Δω)
    else
      # print("propagate axion \n")
      Mvars = [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS,
               [erg_inf_ini], flat, isotropic, melrose, Mass_a, bndry_lyr]
      x_e,k_e,t_e,err_e,cut_short,xc,yc,zc,kxc,kyc,kzc,tc,Δωc,times = RT.propagate(
                        pos0, k0,
                        ax_num, Mvars, NumerPass, RT.func_axion!,
                        true, true, Mass_a, splittings_cutoff, Δω)
    end
    pos = transpose(x_e[1, :, :])
    kpos = transpose(k_e[1, :, :])
   
    event.traj = pos
    event.mom = kpos
    event.erg = transpose(t_e[1, :])
    event.times = times
   
    if length(xc) < 1  # No crossings
        # Since we are considering the most probable first
        count_main += 1
        tot_prob += event.weight
        # It is "final" if it is not killed by the NS
        if sum(pos[end, :].^2)^.5 > rNS*1.1
          event.is_final = true
        end
    else

      ##########################################################################
      # I have not experienced this in a long time. Still, I do not dare to
      # delete anything...
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
        epsabs = 1e-5 # ... as ode_err
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
          tc = tc[flag]
          Δωc = Δωc[flag]
          print(sqrt.(xc.^2 + yc.^2 + zc.^2), "\n")
        end
      end
      ##########################################################################

      # store level crossings
      Nc = length(xc)
      pos = zeros(Nc, 3);
      pos[:, 1] .= xc; pos[:, 2] = yc; pos[:, 3] = zc
      kpos = zeros(Nc, 3);
      kpos[:, 1] .= kxc; kpos[:, 2] .= kyc; kpos[:, 3] .= kzc
      #resontant conversion: "plasma mass"=axion mass
      event.xc = xc
      event.yc = yc
      event.zc = zc
      event.kxc = kxc
      event.kyc = kyc
      event.kzc = kzc
      event.tc = tc 
      event.Δωc = Δωc

      # Conversion probability
      Prob_nonAD = get_Prob_nonAD(pos,kpos,Mass_a,Ax_g,θm,ωPul,B0,rNS,
                          erg_inf_ini .* abs.(Δωc), vIfty_mag, flat, isotropic, bndry_lyr)
      Prob = 1 .- exp.(-Prob_nonAD)
      event.Pc = Prob

      # Find "ID" of new particle              
      if event.species == "photon"
        new_species = "axion"
      else
        new_species = "photon"
      end


      if splittings_cutoff <= 0 # stop at each crossing

        # This event is taking too long! We transition to a pure MC
        if count > MC_nodes
          r = rand(Float64)
          #print("MC: ", r, " ", Prob[1], "\n")
          if r < Prob[1] # New particle
            push!(events, RT.node(xc[1], yc[1], zc[1], kxc[1], kyc[1], kzc[1],
               tc[1], Δωc[1], new_species, Prob[1], event.weight, event.weight,
               Prob[1],Prob[1]))
          else # No conversion
            push!(events, RT.node(xc[1], yc[1], zc[1], kxc[1], kyc[1], kzc[1],
                            tc[1], Δωc[1], event.species, 1-Prob[1],
                            event.weight, event.weight,Prob[1],event.prob_conv))
          end

        else

          # Store full tree
          push!(events, RT.node(xc[1], yc[1], zc[1], kxc[1], kyc[1], kzc[1],
                    tc[1], Δωc[1], new_species, Prob[1], Prob[1]*event.weight,
                    event.weight, Prob[1], Prob[1])) # New particle
          push!(events, RT.node(xc[1], yc[1], zc[1], kxc[1], kyc[1], kzc[1],
                                tc[1], Δωc[1], event.species, 1-Prob[1],
                    (1-Prob[1])*event.weight,
                    event.weight, Prob[1], event.prob_conv)) # No conversion

        end

      else # Follow one particle for more than one crossing
        
        for j in 1:Nc 
            push!(events, RT.node(xc[j], yc[j], zc[j], kxc[j], kyc[j], kzc[j],
                    tc[j], Δωc[j], new_species, Prob[j], Prob[j]*event.weight,
                    event.weight,Prob[1], Prob[1]))
            event.weight = event.weight*(1-Prob[j]) # Re-weight of parent
        end
        tot_prob += event.weight

      end
  
    end

    # Add to tree
    push!(tree, event)

    if tot_prob >= 1 - prob_cutoff
      info = 2
      break
    end
    if num_cutoff <= 0 && splittings_cutoff > 0
      break
    end
    if count_main >= num_cutoff
      info = 3
      break
    end

    if count > max_nodes
      info = 4
      break
    end

    # Sort events to consider the most likely first
    sort!(events, by = events->events.weight)

  end

  if count > MC_nodes
    info = -abs(info)
  end

  return tree, count, info 

end


function main_runner_tree(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, ωProp,
    Ntajs, gammaF; flat=true, isotropic=false, melrose=false,
    thick_surface=true, bndry_lyr=-1,
    ode_err=1e-6, cutT=100000, fix_time=0.0, CLen_Scale=true, file_tag="",
    ntimes=1000, v_NS=[0 0 0], rho_DM=0.45, vmean_ax=220.0, saveMode=0,
    ntimes_ax=1000, dir_tag="results", n_maxSample=6, iseed=-1,
    num_cutoff=5,
    MC_nodes=5, max_nodes=50,
    prob_cutoff=1e-10)

    if iseed < 0
      iseed = rand(0:100000000)
      Random.seed!(iseed)
    elseif iseed == 0
      Random.seed!()
    else
      Random.seed!(iseed)
    end

    print("Using seed ", iseed, "\n")

    batchsize=1
    saveAll = nothing

    if saveMode < 3 
      ntimes = 3 # Times to store in ODE
    end

    # Identify the maximum distance of the conversion surface from NS
    maxR = RT.Find_Conversion_Surface(Mass_a, fix_time, θm, ωPul, B0,
                                      rNS, 1, false)
    maxR_tag = "";

    # check if NS allows for conversion
    if maxR < rNS
        print("Too small Max R.... quitting.... \n")
        omegaP_test = RT.GJ_Model_ωp_scalar(rNS .* [sin.(θm) 0.0 cos.(θm)], 
                                            0.0, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr);
        print("Max omegaP found... \t", omegaP_test,
              "Max radius found...\t", maxR, "\n")
        return
    end


    photon_trajs = 1
    desired_trajs = Ntajs
    f_inx = 0;
    accur_threshold = 1e-4

    # define arrays that are used in surface area sampling
    tt_ax = LinRange(-2*maxR, 2*maxR, ntimes_ax); # Not a real physical time
    t_diff = tt_ax[2] - tt_ax[1];
    tt_ax_zoom = LinRange(-2*t_diff, 2*t_diff, ntimes_ax);

    # define min and max time to propagate photons
    # ln_t_start = -15;
    ln_t_start = -30;
    ln_t_end = log.(1 ./ ωPul);
    NumerPass = [ln_t_start, ln_t_end, ode_err];
    ttΔω = exp.(LinRange(ln_t_start, ln_t_end, ntimes));

    phase_approx = false # Put by hand to false

    vNS_mag = sqrt.(sum(v_NS.^2));
    if vNS_mag .> 0
        vNS_theta = acos.(v_NS[3] ./ vNS_mag);
        vNS_phi = atan.(v_NS[2], v_NS[1]);
    end
    
    # init some arrays
    t0_ax = zeros(batchsize);
    xpos_flat = zeros(batchsize, 3);
    velNorm_flat = zeros(batchsize, 3);
    vIfty = zeros(batchsize, 3);
    R_sample = zeros(batchsize);
    mcmc_weights = zeros(batchsize);
    filled_positions = false;
    fill_indx = 1;
    Ncx_max = 1;
    
    if saveMode > 1 # Clear text
      # Information about each final particle
      fname = dir_tag * "/event/final_" * file_tag
      f_final = open(fname, "w")
      close(f_final)
      # Information about the event
      fname = dir_tag * "/event/event_" * file_tag 
      f_event = open(fname, "w")
      close(f_event)
    end

    small_batch = 1
    
    tot_count = 0
    # Random.seed!(iseed)
    while photon_trajs < desired_trajs

        # --- DEBUG ---
        #iseed = rand(0:100000000)
        #iseed = 51471221
        #Random.seed!(iseed)
        #print(photon_trajs, " ")
        #print(iseed, "\n")
        # -------------

        # First part of code here is just written to generate evenly spaced
        # samples of conversion surface
        
        while !filled_positions
            xv, Rv, numV, weights, vvec_in, vIfty_in = RT.find_samples_new(maxR,
                      θm, ωPul, B0, rNS, Mass_a, Mass_NS;
                      n_max=n_maxSample, batchsize=small_batch,
                      thick_surface=thick_surface, iso=isotropic, melrose=false,
                      bndry_lyr=bndry_lyr)
            f_inx += small_batch
            
            if numV == 0
                continue
            end
          

            for i in 1:Int(numV) # Keep more?
                f_inx -= 1
                if fill_indx <= batchsize
            
                    xpos_flat[fill_indx, :] .= xv[i, :];
                    R_sample[fill_indx] = Rv[i];
                    mcmc_weights[fill_indx] = n_maxSample;
                    velNorm_flat[fill_indx, :] .= vvec_in[i, :];
                    vIfty[fill_indx, :] .= vIfty_in[i, :];
                    fill_indx += 1
                    
                end
            end
            
            if fill_indx > batchsize
                filled_positions = true
                fill_indx = 1
            end
        end
        filled_positions = false;
        
        # vIfty = erfinv.(2 .* rand(batchsize, 3) .- 1.0) .* vmean_ax .+ v_NS#km/s
        rmag = sqrt.(sum(xpos_flat.^ 2, dims=2));
        vmag = sqrt.(2 * GNew .* Mass_NS ./ rmag) ; # km/s

        # print(xpos_flat, "\n", velNorm_flat, "\n\n")
        
        jacVs = zeros(length(rmag))
        
        ϕ = atan.(view(xpos_flat, :, 2), view(xpos_flat, :, 1))
        θ = acos.(view(xpos_flat, :, 3)./ rmag)
        
        for i in 1:length(rmag)
            jacVs[i] = RT.jacobian_fv(xpos_flat[i, :], velNorm_flat[i, :])
        end
        
        
        # define angle between surface normal and velocity
        # calpha = RT.surfNorm(xpos_flat, velNorm_flat, [func_use,
        #       [θm, ωPul, B0, rNS, gammaF, ]],
        #                      return_cos=true); # alpha
        # weight_angle = abs.(calpha);
        
    
        # sample asymptotic velocity
        vIfty_mag = sqrt.(sum(vIfty.^2, dims=2));
        vel_eng = sum((vIfty ./ c_km).^ 2, dims = 2) ./ 2;
        gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ).^2 )
        erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag .* gammaA).^2)
        erg_ax = erg_inf_ini ./ sqrt.(1.0 .- 2 * GNew .* Mass_NS ./ rmag ./ c_km.^2 );
        
        
        # define initial momentum (magnitude)
        k_init = RT.k_norm_Cart(xpos_flat, velNorm_flat,  0.0, erg_inf_ini, θm,
                                ωPul, B0, rNS, Mass_NS, Mass_a, melrose=melrose,
                                isotropic=isotropic, flat=flat, ax_fix=true)
        
        ksphere = RT.k_sphere(xpos_flat, k_init, θm, ωPul, B0, rNS, zeros(batchsize), Mass_NS, Mass_a, flat, bndry_lyr=bndry_lyr)
   
        Mvars =  [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS, Mass_a, flat, isotropic, erg_ax, bndry_lyr]
        sln_δwp, sln_δk, sln_δE, cos_w, vgNorm, dk_vg, dE_vg, k_vg = RT.dwp_ds(xpos_flat, ksphere,  Mvars)
        # sln_δw, angleVal, k_dot_N, dwdk_snorm, vg_mu, vgN  = RT.dwp_ds(xpos_flat, ksphere,  Mvars)
        # calpha = cos.(angleVal)
        # weight_angle = abs.(calpha)
        
        MagnetoVars = [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS,
                       erg_inf_ini, flat, isotropic, melrose]

        
        # Needed for differential power
        # Note: optical depth and weightC are neglected here
        
        theta_sf = acos.(xpos_flat[:,3] ./ rmag)
        x0_pl = [rmag theta_sf atan.(xpos_flat[:,2], xpos_flat[:,1])]
        jacobian_GR = RT.g_det(x0_pl, zeros(batchsize), θm, ωPul, B0, rNS, Mass_NS, Mass_a; flat=flat, bndry_lyr=bndry_lyr); # unitless
        
        dense_extra = 2 ./ sqrt.(π) * (1.0 ./ (220.0 ./ c_km)) .* sqrt.(2.0 * Mass_NS * GNew / c_km^2 ./ rmag)
        # phaseS = jacVs.*phaseS.*jacobian_GR
        redshift_factor = sqrt.(1.0 .- 2 * GNew .* Mass_NS ./ rmag ./ c_km.^2 );
        phaseS = dense_extra .* (2 .* π .* maxR.^2) .* (rho_DM .* 1e9)  ./ Mass_a .* jacobian_GR
        sln_prob = abs.(cos_w) .* redshift_factor .* phaseS .* (1e5 .^ 2) .* c_km .* 1e5 .*
                   mcmc_weights # axions in per second
        # print(sln_prob, "\t", abs.(k_dot_N), "\t", maxR, "\t", jacobian_GR, "\t", dense_extra,  "\n")
        # print(rho_DM, "\t", Mass_a, "\t", mcmc_weights, "\n")
        
        for i in 1:batchsize

          time0=time()

          if saveMode > 1
            # Information about each final particle
            fname = dir_tag * "/event/final_" * file_tag
            f_final = open(fname, "a")
            # Information about the event
            fname = dir_tag * "/event/event_" * file_tag 
            f_event = open(fname, "a")
          end
          if saveMode > 2 # Entire tree
            # Entire tree; one file each
            fname = dir_tag * "/tree/tree_" * file_tag * string(photon_trajs) 
            f = open(fname, "w")
          end

          # Find previous crossings...
          # Backwards in time equivalent to setting k->-k and vecB->-vecB
          parent = RT.node(xpos_flat[i, 1], xpos_flat[i, 2], xpos_flat[i, 3],
                -k_init[i, 1], -k_init[i, 2], -k_init[i, 3], 0, -1.0, # Δω0
                "axion", 1.0, 1.0, -1.0, -1.0, -1.0)
          # The simplest is always the best: make use of existing code
          nb, c_bck, _ = get_tree(parent,erg_inf_ini[i],vIfty_mag[i],
                Mass_a,Ax_g,θm,ωPul,-B0,rNS,Mass_NS,gammaF,
                flat,isotropic, melrose, bndry_lyr, NumerPass;
                prob_cutoff=prob_cutoff, num_cutoff=0, splittings_cutoff=100000,
                ax_num=ntimes)

          nb = nb[1] 
          
          if saveMode > 1
            # Store event information
            write(f_event,
                string(photon_trajs), " ", # Event number
                string(vIfty[1]), " ",string(vIfty[2]), " ", string(vIfty[3]),
                   " ", # Velocity at infinity
                string(sln_prob[1]),     " ", # Incoming axions per second 
                # Incoming axion
                string(nb.x[end]),  " ", string(nb.y[end]),  " ",
                string(nb.z[end]),  " ",
                string(nb.kx[end]), " ", string(nb.ky[end]), " ",
                string(nb.kz[end]), " ",
                # MC selection info
                string(xpos_flat[i,1]), " ", string(xpos_flat[i,2]), " ",
                string(xpos_flat[i,3]), " ",
                string(k_init[i,1]),    " ", string(k_init[i,2]),    " ", 
                string(k_init[i,3])
               )
          end
          if saveMode > 2 saveNode(f, nb) end # Store entire tree

          if length(nb.xc) == 0 # In case the selected conversion is the first
            nb.xc = [xpos_flat[i, 1]]
            nb.yc = [xpos_flat[i, 2]]
            nb.zc = [xpos_flat[i, 3]]
            nb.kxc = [-k_init[i, 1]]
            nb.kyc = [-k_init[i, 2]] # "-" since it should have been backtraced
            nb.kzc = [-k_init[i, 3]]
            nb.tc  = [0.0]
            nb.Δωc = [-1.0]
            nb.Pc  = [nb.prob]
          end

          Prob_nonAD_0 = nb.prob
          nb.tc .-= nb.tc[end] # We define t=0 at the first conversion
          nb.tc .*= -1

          samp_back_weight = nb.prob*nb.weight

          #print(samp_back_weight, "\n") #DEBUG

          # Forward propagation of a photon from the last node
          #species = ["axion*" "photon"]
          #probs = [1 - nb.Pc[end], nb.Pc[end]]
          count = 0 
          #for j in [1 2]
            # Remove comment to have a prob_cutoff for entire tree, and not
            # for each of the two subtrees
            ##if probs[j] > prob_cutoff # Skip if unlikely
              #if j == 1 # axion
              #  parent = RT.node( nb.xc[end],   nb.yc[end],   nb.zc[end],
              #        -nb.kxc[end], -nb.kyc[end], -nb.kzc[end], nb.tc[end],
              #        nb.Δωc[end], species[j], probs[j], probs[j], 1.0,
              #        nb.Pc[end], -1.0)
              #                    # Original axion
              #else      # photon
                #parent = RT.node( nb.xc[end],   nb.yc[end],   nb.zc[end],
                #      -nb.kxc[end], -nb.kyc[end], -nb.kzc[end], nb.tc[end],
                #      nb.Δωc[end], species[j], probs[j], probs[j], 1.0,
                #      nb.Pc[end], nb.Pc[end] )
                parent = RT.node( xpos_flat[i, 1], xpos_flat[i, 2],
                                 xpos_flat[i, 3], k_init[i, 1], k_init[i, 2],
                                 k_init[i, 3], 0,
                                 -1.0, "photon", 1.0, 1.0,
                                 -1.0, -1.0, -1.0 )
              #end
              tree, c, info = get_tree(
                  parent,erg_inf_ini[i],vIfty_mag[i],
                  Mass_a,Ax_g,θm,ωPul,B0,rNS,Mass_NS,gammaF,
                  flat,isotropic, melrose, bndry_lyr, NumerPass;prob_cutoff=prob_cutoff,
                  num_cutoff=num_cutoff,ax_num=ntimes,MC_nodes=MC_nodes,
                  max_nodes=max_nodes)
              count += c
              
              tot_count += length(tree)
              
              # Store results
              for ii in 1:length(tree)
                if saveMode>2 saveNode(f, tree[ii]) end
                if tree[ii].is_final
                  absf  = sqrt(sum(tree[ii].mom[end,:].^2))
                  absfX = sqrt(sum(tree[ii].traj[end,:].^2))
                  ϕf  = atan(tree[ii].mom[end,2], tree[ii].mom[end,1])
                  ϕfX = atan(tree[ii].traj[end,2], tree[ii].traj[end,1])
                  θf  = acos(tree[ii].mom[end,3]/absf)
                  θfX = acos(tree[ii].traj[end,3]/absfX)
                  if tree[ii].species == "axion*" || tree[ii].species == "axion"
                    id = 0
                  else
                    id = 1
                  end 
                 
                  # Re-weight axion with bracktraced weight
                  tree[ii].weight *= samp_back_weight

                  # Save information
                  if saveMode > 1
                    write(f_final,
                        string(photon_trajs),     " ", # Event number
                        string(tree[ii].weight),  " ", # Final weight
                        string(id),  " ", # Species
                        string(θf),  " ", string(ϕf),  " ",
                            string(absf),  " ", # Final momentum
                        string(θfX), " ", string(ϕfX), " ",
                          string(absfX), " ", # Final position
                        string(tree[ii].t), # Time (at final crossing)
                        "\n"
                       )
                  end
                  # ---
                  # See TODOs
                  opticalDepth = 0
                  weightC = 1
                  # ---
                  weight_tmp = tree[ii].weight * (
                                          weightC^2 * exp.(-opticalDepth) )
                  Δω = tree[ii].erg[end] ./ Mass_a .+ vel_eng[:] # Energy change

                  if id == 1
                      f_inx += 1
                  end
                  if saveMode > 0 # Save more
                    row = [photon_trajs id θf ϕf θfX ϕfX absfX sln_prob[1] weight_tmp xpos_flat[i,1] xpos_flat[i,2] xpos_flat[i,3] Δω[1] tree[ii].weight opticalDepth weightC k_init[i,1] k_init[i,2] k_init[i,3] cos_w[1] c info tree[ii].prob tree[ii].prob_conv tree[ii].prob_conv0 samp_back_weight absfX c_bck Prob_nonAD_0]

                  else
                    # print(θfX, "\t", ϕfX, "\n")

                    row = [photon_trajs id θf ϕf θfX ϕfX absfX sln_prob[1] weight_tmp xpos_flat[i,1] xpos_flat[i,2] xpos_flat[i,3] Δω[1]]
                  end
                  if isnothing(saveAll)
                    saveAll = row
                  else
                    saveAll = [saveAll; row]
                  end
                
                end
              end
            #end
          #end j

          photon_trajs += 1

          if saveMode > 2 close(f) end
          if saveMode > 1
            write(f_event, " ", string(time() - time0),
                  " ", string(count), "\n")
            close(f_final)
            close(f_event)
          end

        end # Batchsize, in any case 1...

      GC.gc() # Manually delete trashed memory

    end # while
    
    saveAll[:, 8] ./= float(f_inx) # divide off by N trajectories sampled
    fileN = dir_tag*"/npy/tree_"
    fileN *= "MassAx_"*string(Mass_a)*"_AxionG_"*string(Ax_g)
    fileN *="_ThetaM_"*string(θm)*"_rotPulsar_"*string(ωPul)*"_B0_"*string(B0)
    fileN *= "_Ax_trajs_"*string(Ntajs)
    fileN *= "_N_Times_"*string(ntimes);
    #fileN *= "_N_maxSample_"*string(n_maxSample)
    fileN *= "_num_cutoff_"*string(num_cutoff)
    fileN *= "_MC_nodes_"*string(MC_nodes)
    fileN *= "_max_nodes_"*string(max_nodes)
    # fileN *= "_iseed_"*string(iseed)
    fileN *= "_"*file_tag*".npy"
    npzwrite(fileN, saveAll)

    return tot_count

end
