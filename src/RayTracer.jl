__precompile__()


include("Constants.jl")
import .Constants: c_km, hbar, GNew

module RayTracerGR
import ..Constants: c_km, hbar, GNew

using ForwardDiff: gradient, derivative, Dual, Partials, hessian
using OrdinaryDiffEq
using LSODA
using DiffEqBase
using SpecialFunctions
using LinearAlgebra: cross, det
using NLsolve

# CuArrays.allowscalar(false)

### Parallelized derivatives with ForwardDiff.Dual

# Seed 3-dim vector with dual partials for gradient calculation
seed = x -> [map(y -> Dual(y, (1., 0., 0.)), x[:,1]) map(y -> Dual(y, (0., 1., 0.)), x[:,2]) map(y -> Dual(y, (0., 0., 1.)), x[:,3])]

# Extract gradient from dual
grad = x -> [map(x -> x.partials[1], x) map(x -> x.partials[2], x) map(x -> x.partials[3], x)]


### Parallelized crossing calculations

struct Crossings
    i1
    i2
    weight
end

"""
Calculate values of matrix X at crossing points
"""
function apply(c::Crossings, A)
    A[c.i1] .* c.weight .+ A[c.i2] .* (1 .- c.weight)
end

"""
calcuates crossings along 2 axis
"""
function get_crossings(A; keep_all=true)
    # Matrix with 2 for upward and -2 for downward crossings
    sign_A = sign.(A)
    #cross = sign_A[:, 2:end] - sign_A[:, 1:end-1]
    cross = sign_A[2:end] - sign_A[1:end-1]
    #print(cross)
    # Index just before crossing
    if keep_all
        i1 = Array(findall(x -> x .!= 0., cross))
    else
        i1 = Array(findall(x -> x .> 0., cross))
    end

    # Index just behind crossing
    #i2 = map(x -> x + CartesianIndex(0, 1), i1)
    i2 = i1 .+ 1

    # Estimate weight for linear interpolation
    weight = A[i2] ./ (A[i2] .- A[i1])

    return Crossings(i1, i2, weight)
end



# compute photon trajectories
function func!(du, u, Mvars, lnt)
    @inbounds begin
        t = exp.(lnt);
        
        Mass_NS = 1.0;
        ω, Mvars2 = Mvars;
        θm, ωPul, B0, rNS, gammaF, time0, Mass_NS, erg, flat, isotropic, melrose, bndry_lyr = Mvars2;
        if flat
            Mass_NS = 0.0;
        end
        time = time0 .+  t;
        
        g_tt, g_rr, g_thth, g_pp = g_schwartz(view(u, :, 1:3), Mass_NS);
        
        du[:, 4:6] .= -grad(hamiltonian(seed(view(u, :, 1:3)), view(u, :, 4:6) .* erg , time[1], -view(u, :, 7), θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=melrose, bndry_lyr=bndry_lyr)) .* c_km .* t .* (g_rr ./ erg) ./ erg;
        du[:, 1:3] .= grad(hamiltonian(view(u, :, 1:3), seed(view(u, :, 4:6)  .* erg ), time[1], -view(u, :, 7), θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=melrose, bndry_lyr=bndry_lyr)) .* c_km .* t .* (g_rr ./ erg);
        du[u[:,1] .<= rNS, :] .= 0.0;
        
        du[:,7 ] .= derivative(tI -> hamiltonian(view(u, :, 1:3), view(u, :, 4:6)  .* erg , tI, -view(u, :, 7), θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=melrose, bndry_lyr=bndry_lyr), time[1])[:] .* t .* (g_rr[:] ./ -view(u, :, 7));
        
    end
end

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# compute axion trajectories
function func_axion!(du, u, Mvars, lnt)
    @inbounds begin
        t = exp.(lnt);
        
        Mass_NS = 1.0;
        ω, Mvars2 = Mvars;
        θm, ωPul, B0, rNS, gammaF, time0, Mass_NS, erg, flat, isotropic, melrose, mass_axion = Mvars2;
        if flat
            Mass_NS = 0.0;
        end
        time = time0 .+  t;
       
        r = u[:, 1]
        Mass_NS = Mass_NS*ones(length(r))
        Mass_NS[r .< rNS] = r[r .< rNS].^3/rNS^3

        g_tt, g_rr, g_thth, g_pp = g_schwartz(view(u, :, 1:3), Mass_NS);


        du[:, 4:6] .= -grad(hamiltonian_axion(seed(view(u, :, 1:3)),
                       view(u, :, 4:6) .* erg , time[1], erg, θm, ωPul, B0, rNS,
                       Mass_NS, mass_axion, iso=isotropic, melrose=melrose)) .*
                           c_km .* t .* (g_rr ./ erg) ./ erg;
        du[:, 1:3] .= grad(hamiltonian_axion(view(u, :, 1:3),
              seed(view(u, :, 4:6)  .* erg ), time[1], erg, θm, ωPul,
              B0, rNS, Mass_NS, mass_axion, iso=isotropic, melrose=melrose)) .*
                    c_km .* t .* (g_rr ./ erg);
        #du[u[:,1] .<= rNS, :] .= 0.0;
        
        
    end
end

# Struct for conversion points in trajectory
mutable struct node
  x    # Conversion position/initial conditions
  y
  z
  kx
  ky
  kz
  t
  Δω
  species # Axion or photon?
  prob   # Last instance in weight
  weight
  parent_weight
  prob_conv  # Photon conversion probability at last level crossing, -1: parent
  prob_conv0 # Photon conversion probability at level crossing wherein the
             # current particle species was produced, -1: parent
  xc # Level crossings
  yc
  zc
  kxc
  kyc
  kzc
  tc
  Δωc
  Pc # Conversion probability
  is_final
  traj    # Used to store entire trajectory
  mom    # Used to store entire momentum
  erg
end
# Constructor
node(x0=0.,y0=0.,z0=0.,kx0=0.,ky0=0.,kz0=0.,t0=0.,Δω0=-1.0,
     species0="axion",prob0=0.,
     weight0=0.,parent_weight0=0.,prob_conv=0.,prob_conv0=0.) = node(
      x0,y0,z0,kx0,ky0,kz0,t0,Δω0,species0,
      prob0,weight0,parent_weight0,prob_conv,prob_conv0,
      [],[],[],[],[],[],[],[],[],false,[],[],[])
#node(x=0.,y=0.,z=0.,kx=0.,ky=0.,kz=0.,t=0.,species="axion",prob=0.,weight=0.,
#     parent_weight=0.,xc=[],yc=[],zc=[],kxc=[],kyc=[],kzc=[],tc=[],Pc=[],
#    is_final=false,traj=[],mom=[]) = node(x,y,z,kx,ky,kz,t,species,prob,weight,
#     parent_weight,xc,yc,zc,kxc,kyc,kzc,tc,Pc,is_final,traj,mom)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# propogate photon module
function propagate(ω, x0::Matrix, k0::Matrix,  nsteps, Mvars, NumerP, rhs=func!,
    make_tree=false, is_axion=false, Mass_a=1e-6, max_crossings=3, Δω=-1.0; bndry_lyr=false)
    ln_tstart, ln_tend, ode_err = NumerP
    
    tspan = (ln_tstart, ln_tend)
    saveat = (tspan[2] .- tspan[1]) ./ (nsteps-1)
    
    if is_axion 
      θm,ωPul,B0,rNS,gammaF,time0,Mass_NS,erg,flat,isotropic,melrose,Mass_a=Mvars;
      k0 = k_norm_Cart(x0, k0, time0, erg, θm, ωPul, B0, rNS, Mass_NS, Mass_a, melrose=melrose, flat=flat, isotropic=isotropic, is_photon=false)
    else
      θm,ωPul,B0,rNS,gammaF,time0,Mass_NS,erg,flat,isotropic,melrose=Mvars;
      k0 = k_norm_Cart(x0, k0, time0, erg, θm, ωPul, B0, rNS, Mass_NS, Mass_a, melrose=melrose, flat=flat, isotropic=isotropic, is_photon=true)
    end
    if flat
        Mass_NS = 0.0;
    end
    

    # Define the Schwarzschild radius of the NS (in km)
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2

    # Switch to polar coordinates
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    
    omP = GJ_Model_ωp_vecSPH(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=true, bndry_lyr=bndry_lyr);
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* k0, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* k0[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* k0[:,1] .+ x0[:,1] .* k0[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    
    # Switch to celerity in polar coordinates
    AA = (1.0 .- r_s0 ./ rr)
    
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    
    
#    if !is_axion
#        val = hamiltonian(x0_pl, w0_pl, time0, erg, θm, ωPul, B0, rNS, Mass_NS; iso=isotropic, melrose=melrose) ./ (erg ./ sqrt.(AA)).^2
#        print(val, "\n\n")
#    end
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    # kpar = K_par(x0_pl, w0_pl, [θm, ωPul, B0, rNS, time0, Mass_NS])
    # if isotropic
    #     kpar .*= 0.0
    # end
    # NrmSq = (-erg.^2 .* g_tt .- omP.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp .- omP.^2 .* kpar.^2 ./ (erg.^2 ./ g_rr))
    # NrmSq = (-erg.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp )
    w0_pl .*= 1.0 ./ erg # switch to dt and renormalize for order 1 vals
    
    # Define initial conditions so that u0[1] returns a list of x positions (again, 1 entry for each axion trajectory) etc.
    # Δω: relative energy change (is negative)
    u0 = ([x0_pl w0_pl erg .* Δω])
    #u0 = ([x0_pl w0_pl -erg])
    # u0 = ([x0_pl w0_pl])
   
    bounce_threshold = 0.0

    function floor_aff!(int)
    
        r_s0 = 2.0 * Mass_NS * GNew / c_km^2
        AA = sqrt.(1.0 .- r_s0 ./ int.u[:, 1])
        
        test = (-int.u[:, 7] ./ AA .- GJ_Model_ωp_vecSPH(int.u[:, 1:3], exp.(int.t), θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr) ) ./ erg # if negative we have problem
        fail_indx = [if test[i] .< bounce_threshold i else -1 end for i in 1:length(int.u[:,1])]; # define when to bounce
        fail_indx = fail_indx[ fail_indx .> 0];
       
        if int.dt > 1e-10
            set_proposed_dt!(int,(int.t-int.tprev)/10)
        end
        
        if length(fail_indx) .> 0
            g_tt, g_rr, g_thth, g_pp = g_schwartz(int.u[fail_indx, 1:3], Mass_NS);
                        
            dωdr_grd = -grad(GJ_Model_ωp_vecSPH(seed(int.u[fail_indx, 1:3]), exp.(int.t), θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr)); # [eV / km, eV, eV]
            dωdr_grd ./= sqrt.(dωdr_grd[:, 1].^2 .* g_rr .+ dωdr_grd[:, 2].^2 .* g_thth  .+ dωdr_grd[:, 3].^2 .* g_pp) # eV / km, net: [ , km , km]

            # int.u[fail_indx, 4:6] .= int.u[fail_indx, 4:6] .- 2.0 .* (int.u[fail_indx, 4] .* dωdr_grd[:, 1] .* g_rr .+ int.u[fail_indx, 5] .* dωdr_grd[:, 2] .* g_thth .+ int.u[fail_indx, 6] .* dωdr_grd[:, 3] .* g_pp) .* dωdr_grd;
            
        end
    end
    function cond(u, lnt, integrator)
        r_s0 = 2.0 * Mass_NS * GNew / c_km^2
        AA = sqrt.(1.0 .- r_s0 ./ u[:, 1])
        test = (-u[:, 7] ./ AA .- GJ_Model_ωp_vecSPH(u, exp.(lnt), θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr)) ./ (- u[:, 7] ./ AA) .+ bounce_threshold # trigger when closer to reflection....
        return minimum(test)
        
    end
    cb = ContinuousCallback(cond, floor_aff!, interp_points=20, repeat_nudge = 1//100, rootfind=DiffEqBase.RightRootFind)
    

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Stop integration when a given number of level crossings is achieved
    # Multiple callbacks -> CallbackSet
    if make_tree

      cut_short = false
      
      # Store crossings to be used later
      xc = []; yc = []; zc = []
      kxc = []; kyc = []; kzc = []
      tc = []; Δωc = []

      # Cut after given amount of crossings
      function condition(u, lnt, integrator)
        return (log.(GJ_Model_ωp_vecSPH(u, exp.(lnt), θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr))
                .- log.(Mass_a))[1]
      end
      function affect!(i)

          if i.opts.userdata[:callback_count] == 0
            # If i.u has not changed, it is not a new crossings...
            s = 1.0001
            pos = [sin(i.u[2])*cos(i.u[3]) sin(i.u[2])*sin(i.u[3]) cos(i.u[2])]
            pos .*= i.u[1]
            if ( all(abs.(pos[1:3]) .< abs.(x0[1:3]).*s) &&
                 all(abs.(pos[1:3]) .> abs.(x0[1:3])./s) )
              return
            end
          end

          # Compute position in cartesian coordinates
          xpos = i.u[1]*sin(i.u[2])*cos(i.u[3]) 
          ypos = i.u[1]*sin(i.u[2])*sin(i.u[3])
          zpos = i.u[1]*cos(i.u[2])
          # Conversions close to the surface is unlikely
          # Slightly better to include this in "condition"!
          if sqrt(xpos^2 + ypos^2 + zpos^2) < rNS*1.01
            return
          end
          push!( xc, xpos )
          push!( yc, ypos )
          push!( zc, zpos )
          push!( tc, exp(i.t) ) # proper time
          push!( Δωc, i.u[7]/erg )

          # Compute proper velocity
          r_s = 2.0 * Mass_NS * GNew / c_km^2
          ω = 1.0 - r_s / i.u[1]
          v_pl = [i.u[4]*sqrt(ω)  i.u[5]/i.u[1]  i.u[6]/(i.u[1]*sin(i.u[2]))]
          v_pl .*= erg[1]*ω
          v_tmp = sin(i.u[2])*v_pl[1] + cos(i.u[2])*v_pl[2]
          push!( kxc, cos(i.u[3])*v_tmp   - sin(i.u[3])*v_pl[3] )
          push!( kyc, sin(i.u[3])*v_tmp   + cos(i.u[3])*v_pl[3] )
          push!( kzc, cos(i.u[2])*v_pl[1] - sin(i.u[2])*v_pl[2] ) 

          # Check if we want to stop ODE
          i.opts.userdata[:callback_count] +=1
          if i.opts.userdata[:callback_count] >= i.opts.userdata[:max_count]
              cut_short = true
              terminate!(i)
          end
      end
      # Cut if inside a neutron star (and a photon). 
      condition_r(u,lnt,integrator) = u[1] < (rNS*1.01)
      affect_r!(integrator) = terminate!(integrator)
     
      cb_s = ContinuousCallback(condition, affect!)
      cb_r = DiscreteCallback(condition_r, affect_r!)
      if is_axion
        cbset = CallbackSet(cb_s) # cb->reflection, cb_->NS, not for axion
      else
        cbset = CallbackSet(cb, cb_s, cb_r)
      end

      prob = ODEProblem(rhs, u0, tspan, [ω, Mvars], callback=cbset, userdata=Dict(:callback_count=>0, :max_count=>max_crossings))
      # prob = ODEProblem(rhs, u0, tspan, [ω, Mvars])
    else
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Define the ODEproblem
      prob = ODEProblem(rhs, u0, tspan, [ω, Mvars], callback=cb)
      # prob = ODEProblem(rhs, u0, tspan, [ω, Mvars])
    end


    # Solve the ODEproblem
    sol = solve(prob, Vern6(), saveat=saveat, reltol=1e-5, abstol=ode_err,
            max_iters=1e5, force_dtmin=true,
            dtmin=1e-13, dtmax=1e-2)


    for i in 1:length(sol.u)
        sol.u[i][:,4:6] .*= erg
    end
   
    # Axion can be inside NS
    r = [sol.u[i][1] for i in 1:length(sol.u)]
    Mass_NS = Mass_NS*ones(length(r), length(sol.u))
    for i in 1:length(sol.u)
      Mass_NS[r .< rNS, i] .*= r[r .< rNS].^3/rNS^3
    end
    
    # Define the Schwarzschild radii (in km)
    r_s = 2.0 .* ones(length(sol.u[1][:,1]), length(sol.u)) .* Mass_NS .* GNew ./ c_km^2
    ω = [(1.0 .- r_s[i] ./ sol.u[i][:,1]) for i in 1:length(sol.u)]



    # Switch back to proper velocity
    v_pl = [[sol.u[i][:,4] .* sqrt.(ω[i])  sol.u[i][:,5] ./ sol.u[i][:,1] sol.u[i][:,6] ./ (sol.u[i][:,1] .* sin.(sol.u[i][:,2])) ] .* ω[i] for i in 1:length(sol.u)]
    
    # Switch back to Cartesian coordinates
    x = [[sol.u[i][:,1] .* sin.(sol.u[i][:,2]) .* cos.(sol.u[i][:,3])  sol.u[i][:,1] .* sin.(sol.u[i][:,2]) .* sin.(sol.u[i][:,3])  sol.u[i][:,1] .* cos.(sol.u[i][:,2])] for i in 1:length(sol.u)]
    
    v = [[cos.(sol.u[i][:,3]) .* (sin.(sol.u[i][:,2]) .* v_pl[i][:,1] .+ cos.(sol.u[i][:,2]) .* v_pl[i][:,2]) .- sin.(sol.u[i][:,2]) .* sin.(sol.u[i][:,3]) .* v_pl[i][:,3] ./ sin.(sol.u[i][:,2])  sin.(sol.u[i][:,3]) .* (sin.(sol.u[i][:,2]) .* v_pl[i][:,1] .+ cos.(sol.u[i][:,2]) .* v_pl[i][:,2]) .+ sin.(sol.u[i][:,2]) .* cos.(sol.u[i][:,3]) .* v_pl[i][:,3] ./ sin.(sol.u[i][:,2])   cos.(sol.u[i][:,2]) .* v_pl[i][:,1] .-  sin.(sol.u[i][:,2]) .* v_pl[i][:,2] ] for i in 1:length(sol.u)]
    
    # print(k0,"\t",v[1,:], "\n")
    
    # Define the return values so that x_reshaped, v_reshaped (called with propagateAxion()[1] and propagateAxion()[2] respectively) are st by 3 by nsteps arrays (3 coordinates at different timesteps for different axion trajectories)
    
    # second one is index up, first is index down
    # dxdtau = [[sol.u[i][:,4]  sol.u[i][:,5]  sol.u[i][:,6] ] for i in 1:length(sol.u)]
    # dxdtau = [[sol.u[i][:,4] .* ω[i] sol.u[i][:,5] ./ sol.u[i][:,1].^2 sol.u[i][:,6] ./ (sol.u[i][:,1] .*  sin.(sol.u[i][:,2])).^2 ] for i in 1:length(sol.u)]
    
    
    x_reshaped = cat([x[:, 1:3] for x in x]..., dims=3)
    v_reshaped = cat([v[:, 1:3] for v in v]..., dims=3)
    # dxdtau = cat([dxdtau[:, 1:3] for dxdtau in dxdtau]..., dims=3)
    sphere_c = cat([sol.u[i][:, 1:3] for i in 1:length(sol.u)]..., dims=3)
    # ω_reshaped = cat([ω[:, 1] ./ ω[:, 1]  for ω in ω]..., dims=2)
    # ω_reshaped = cat([1.0 for ω in ω]..., dims=2)
    dt = cat([Array(u)[:, 7] for u in sol.u]..., dims = 2);
    

    fail_indx = ones(length(sphere_c[:, 1, end]))
    fail_indx[sphere_c[:, 1, end] .<= rNS] .= 0.0
    
    # dxdtau = dxdtau[sphere_c[:, 1, end] .> rNS, :, :]
    # ω_reshaped = ω_reshaped[sphere_c[:, 1, end] .> rNS]
    
    # dt = [sol.u[i][:, 7] for i in 1:length(sol.u)]
    # Also return the list of (proper) times at which the solution is saved for pinpointing the seeding time
    times = sol.t
    
    sol = nothing;
    v_pl = nothing;
    u0 = nothing;
    w0_pl = nothing;
    v0_pl = nothing;
    x0_pl = nothing;
    dr_dt = nothing;
    GC.gc();
   
    if make_tree
      return x_reshaped,v_reshaped,dt,fail_indx,cut_short,xc,yc,zc,kxc,kyc,kzc,tc,Δωc
    else
      return x_reshaped,v_reshaped,dt,fail_indx
    end
end


function g_schwartz(x0, Mass_NS; rNS=10.0)
    # (1 - r_s / r)
    # notation (-,+,+,+), upper g^mu^nu
    
    r = x0[:,1]
    
    # Reduced NS mass is done elsewhere...
    # Mass_NS = Mass_NS_in .* ones(length(r))
    # Mass_NS[r .<= rNS] .= Mass_NS_in .* r[r .<= rNS].^3 ./ rNS.^3


    # rs = 2 * GNew .* Mass_NS ./ c_km.^2 .* ones(length(x0[:,1]))
    # suppress GR inside NS
    # rs[r .<= rNS] .= 0.0
    rs = ones(eltype(r), size(r)) .* 2 * GNew .* Mass_NS ./ c_km.^2
    rs[r .<= rNS] .*= (r[r .<= rNS] ./ rNS).^3

    sin_theta = sin.(x0[:,2])
    g_tt = -1.0 ./ (1.0 .- rs ./ r);
    g_rr = (1.0 .- rs ./ r);
    g_thth = 1.0 ./ r.^2; # 1/km^2
    g_pp = 1.0 ./ (r.^2 .* sin_theta.^2); # 1/km^2

  
    return g_tt, g_rr, g_thth, g_pp
    
end


function hamiltonian(x, k,  time0, erg, θm, ωPul, B0, rNS, Mass_NS; iso=true, melrose=false, zeroIn=false, bndry_lyr=false)
    omP = GJ_Model_ωp_vecSPH(x, time0, θm, ωPul, B0, rNS, zeroIn=zeroIn, bndry_lyr=bndry_lyr);
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x, Mass_NS);
    ksqr = g_tt .* erg.^2 .+ g_rr .* k[:, 1].^2 .+ g_thth .* k[:, 2].^2 .+ g_pp .* k[:, 3].^2
    
    if iso
        Ham = 0.5 .* (ksqr .+ omP.^2)
    else
        if !melrose
            ctheta = Ctheta_B_sphere(x, k, [θm, ωPul, B0, rNS, time0, Mass_NS])

            Ham = (ksqr .- omP.^2 .* (1.0 .- ctheta.^2) ./ (omP.^2 .* ctheta.^2 .- erg.^2 ./ g_rr)  .* erg.^2 ./ g_rr) # original form
        else
            kpar = K_par(x, k, [θm, ωPul, B0, rNS, time0, Mass_NS])
            Ham = (ksqr .+ omP.^2 .* (erg.^2 ./ g_rr .- kpar.^2) ./  (erg.^2 ./ g_rr)  );
        end
    end
    
    return Ham
end

function omega_function(x, k,  time0, erg, θm, ωPul, B0, rNS, Mass_NS; iso=true, melrose=false, flat=false, zeroIn=false, bndry_lyr=false)
    # if r < rNS, need to not run...
    x[x[:,1] .< rNS, 1] .= rNS;

    omP = GJ_Model_ωp_vecSPH(x, time0, θm, ωPul, B0, rNS, zeroIn=zeroIn, bndry_lyr=bndry_lyr);
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x, Mass_NS);
    ksqr = 0.0;
    try
        ksqr = g_rr .* k[:, 1].^2 .+ g_thth .* k[:, 2].^2 .+ g_pp .* k[:, 3].^2
    catch
        ksqr = g_rr .* k[1].^2 .+ g_thth .* k[2].^2 .+ g_pp .* k[3].^2
    end
    
    if iso
        Ham = (ksqr .+ omP.^2)
    else
        
        kpar = K_par(x, k, [θm, ωPul, B0, rNS, time0, Mass_NS])
        Ham = (ksqr .+ omP.^2 .+ sqrt.(ksqr.^2 .+ 2 .* ksqr .* omP.^2 .- 4 .* kpar.^2 .* omP.^2 .+ omP.^4)) ./ sqrt.(2)
        
    end
    
    return sqrt.(Ham)
end

function test_on_shell(x, v_loc, vIfty_mag, time0, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=true, melrose=false, printStuff=false, bndry_lyr=false)
    # pass cartesian form
    # Define the Schwarzschild radius of the NS (in km)
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2

    # Switch to polar coordinates
    rr = sqrt.(sum(x.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x[:,3] ./ rr) atan.(x[:,2], x[:,1])]
    
    AA = (1.0 .- r_s0 ./ rr)
    AA[rr .< rNS] .= (1.0 .- r_s0 ./ rNS)
    
    gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ./ c_km).^2 )
    erg_inf = Mass_a .* sqrt.(1 .+ (vIfty_mag ./ c_km .* gammaA).^2)
    erg_loc = erg_inf ./ sqrt.(AA)

    v0 = transpose(v_loc) .* (erg_loc ./ sqrt.(erg_loc.^2 .+ Mass_a.^2))
    
    omP = GJ_Model_ωp_vecSPH(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=false, bndry_lyr=bndry_lyr);
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x .* v0, dims=2) ./ rr[:, 1]
    v0_pl = [dr_dt (x[:,3] .* dr_dt .- rr .* v0[:, 3]) ./ (rr .* sin.(x0_pl[:,2])) (-x[:,2] .* v0[:, 1] .+ x[:,1] .* v0[:, 2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    # Switch to celerity in polar coordinates
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);

    NrmSq = (-erg_inf.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp )
    w0_pl .*= sqrt.(NrmSq)
    val = hamiltonian(x0_pl, w0_pl, time0, erg_inf, θm, ωPul, B0, rNS, Mass_NS; iso=iso, melrose=melrose, zeroIn=false, bndry_lyr=bndry_lyr) ./ erg_inf.^2
    
    tVals = erg_loc .> omP
    val_final = val[tVals[:]]
    min_value = minimum(abs.(val))
    return val_final, tVals[:], min_value
    
end

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
function hamiltonian_axion(x, k,  time0, erg, θm, ωPul, B0, rNS,
    Mass_NS, mass_axion; iso=true, melrose=false)
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x, Mass_NS)
    ksqr = g_tt .* erg.^2 .+ g_rr .* k[:, 1].^2 .+ g_thth .* k[:, 2].^2 .+
            g_pp .* k[:, 3].^2
    #print(ksqr, "\n") 
    return 0.5 .* ksqr
    
end
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

function k_norm_Cart(x0, khat, time0, erg, θm, ωPul, B0, rNS, Mass_NS, Mass_a; melrose=false, flat=false, isotropic=false, ax_fix=true, is_photon=true)
    
    
    # Switch to polar coordinates
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]

    
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* khat, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* khat[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* khat[:,1] .+ x0[:,1] .* khat[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    
    # Switch to celerity in polar coordinates
    AA = (1.0 .- r_s0 ./ rr)
    
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA  # lower index defined, [eV, eV * km, eV * km]
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    
    if is_photon
        if ax_fix
            NrmSq = (-erg.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp )
        else
            omP = GJ_Model_ωp_vecSPH(x0_pl, time0, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr);
            kpar = K_par(x0_pl, w0_pl, [θm, ωPul, B0, rNS, time0, Mass_NS]; flat=flat)
            NrmSq = (-erg.^2 .* g_tt .- omP.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp .- omP.^2 ./ (-erg.^2 .* g_tt) .* kpar.^2 )
        end
    else
        NrmSq = (-erg.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp )
    end
          
    return sqrt.(NrmSq) .* khat
    
end




function dwdt_vec(x0, k0, tarr, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    nphotons = size(x0)[1]
    delW = zeros(nphotons);
    
    for k in 1:nphotons
        t0 = tarr .+ t_start[k];
        for i in 2:length(tarr)
            dωdt = derivative(t -> ω(transpose(x0[k, :, i]), transpose(k0[k, :, i]), t, θm, ωPul, B0, rNS, gammaF), t0[i]);
            delW[k] += dωdt[1] .* sqrt.(sum((x0[k, :, i] .- x0[k, :, i-1]) .^2)) / c_km
        end
    end
    return delW
end

function solve_vel_CS(θ, ϕ, r, NS_vel; guess=[0.1 0.1 0.1], errV=1e-24, Mass_NS=1)
    ff = sum(NS_vel.^2); # unitless
    
    
    GMr = GNew .* Mass_NS ./ r ./ (c_km .^ 2); # unitless
    rhat = [sin.(θ) .* cos.(ϕ) sin.(θ) .* sin.(ϕ) cos.(θ)]

    function f!(F, x)
        vx, vy, vz = x
        
        denom = ff .+ GMr .- sqrt.(ff) .* sum(x .* rhat);

        F[1] = (ff .* vx .+ sqrt.(ff) .* GMr .* rhat[1] .- sqrt.(ff) .* vx .* sum(x .* rhat)) ./ (NS_vel[1] .* denom) .- 1.0
        F[2] = (ff .* vy .+ sqrt.(ff) .* GMr .* rhat[2] .- sqrt.(ff) .* vy .* sum(x .* rhat)) ./ (NS_vel[2] .* denom) .- 1.0
        F[3] = (ff .* vz .+ sqrt.(ff) .* GMr .* rhat[3] .- sqrt.(ff) .* vz .* sum(x .* rhat)) ./ (NS_vel[3] .* denom) .- 1.0
        # print(F[1], "\t",F[2], "\t", F[3],"\n")
        # print(θ, "\t", ϕ,"\t", r, "\n")
    end

    soln = nlsolve(f!, guess, autodiff = :forward, ftol=errV, iterations=10000)

    FF = zeros(3)
    f!(FF, soln.zero)
    accur = sqrt.(sum(FF.^2))
    # print("accuracy... ", FF,"\n")
    return soln.zero, accur
end

function g_det(x0, t, θm, ωPul, B0, rNS, Mass_NS; flat=false, bndry_lyr=false)
    # returns determinant of sqrt(-g)
    if flat
        return ones(length(x0[:, 1]))
    end
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0, Mass_NS; rNS=rNS);
    
    r = x0[:,1]
    dωp = grad(GJ_Model_ωp_vecSPH(seed(x0), t, θm, ωPul, B0, rNS, zeroIn=false, bndry_lyr=bndry_lyr));
    # dωt =  derivative(tI -> GJ_Model_ωp_vecSPH(x0, tI, θm, ωPul, B0, rNS, zeroIn=false, bndry_lyr=bndry_lyr), t[1]);
    
    # dr_t = dωp[:, 1].^(-1) .* dωt ./ c_km # unitless
    dr_th = dωp[:, 1].^(-1) .* dωp[:, 2] # km
    dr_p = dωp[:, 1].^(-1) .* dωp[:, 3] # km
    A = g_rr
    
    sqrt_det_g = r .* sqrt.(sin.(x0[:,2]).^2 .* (A .* r.^2 .+ dr_th.^2) .+ dr_p.^2)
    sqrt_det_g_noGR = r .* sqrt.(sin.(x0[:,2]).^2 .* (r.^2 .+ dr_th.^2) .+ dr_p.^2)
    return sqrt_det_g ./ sqrt_det_g_noGR # km^2
end

function jacobian_fv(x_in, vel_loc)

    rmag = sqrt.(sum(x_in.^2));
    ϕ = atan.(x_in[2], x_in[1])
    θ = acos.(x_in[3] ./ rmag)
    
    dvXi_dV = grad(v_infinity(θ, ϕ, rmag, seed(transpose(vel_loc)), v_comp=1));
    dvYi_dV = grad(v_infinity(θ, ϕ, rmag, seed(transpose(vel_loc)), v_comp=2));
    dvZi_dV = grad(v_infinity(θ, ϕ, rmag, seed(transpose(vel_loc)), v_comp=3));
    
    JJ = det([dvXi_dV; dvYi_dV; dvZi_dV])

    return abs.(JJ).^(-1)
end

function v_infinity(θ, ϕ, r, vel_loc; v_comp=1, Mass_NS=1)
    vx, vy, vz = vel_loc
    vel_loc_mag = sqrt.(sum(vel_loc.^2))
    GMr = GNew .* Mass_NS ./ r ./ (c_km .^ 2); # unitless

    v_inf = sqrt.(vel_loc_mag.^2 .- (2 .* GMr)); # unitless
    rhat = [sin.(θ) .* cos.(ϕ) sin.(θ) .* sin.(ϕ) cos.(θ)]
    r_dot_v = sum(vel_loc .* rhat)
    
    denom = v_inf.^2 .+ GMr .- v_inf .* r_dot_v;
    
    if v_comp == 1
        v_inf_comp = (v_inf.^2 .* vx .+ v_inf .* GMr .* rhat[1] .- v_inf .* vx .* r_dot_v) ./ denom
    elseif v_comp == 2
        v_inf_comp = (v_inf.^2 .* vy .+ v_inf .* GMr .* rhat[2] .- v_inf .* vy .* r_dot_v) ./ denom
    else
        v_inf_comp = (v_inf.^2 .* vz .+ v_inf .* GMr .* rhat[3] .- v_inf .* vz .* r_dot_v) ./ denom
    end
    return v_inf_comp
end

function cyclotronF(x0, t0, θm, ωPul, B0, rNS)
    Bvec, ωp = GJ_Model_scalar(x0, t0, θm, ωPul, B0, rNS)
    omegaC = sqrt.(sum(Bvec.^2, dims=2)) * 0.3 / 5.11e5 * (1.95e-20 * 1e18) # eV
    return omegaC
end

function cyclotronF_vec(x0, t0, θm, ωPul, B0, rNS)
    Bvec, ωp = GJ_Model_vec(x0, t0, θm, ωPul, B0, rNS)
    omegaC = sqrt.(sum(Bvec.^2, dims=2)) * 0.3 / 5.11e5 * (1.95e-20 * 1e18) # eV
    return omegaC
end

function tau_cyc(x0, k0, tarr, Mvars, Mass_a)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    nphotons = size(x0)[1]
    cxing_indx = zeros(Int16, nphotons)
    tau = zeros(nphotons)
    xpoints = zeros(nphotons, 3)
    kpoints = zeros(nphotons, 3)
    tpoints = zeros(nphotons)
    # cyclFOut = zeros(nphotons)
    
    for k in 1:nphotons
        t0 = tarr .+ t_start[k];
        cyclF = zeros(length(t0));
        for i in 1:length(tarr)
            cyclF[i] = cyclotronF(x0[k, :, i], t0[i], θm, ωPul, B0, rNS)[1];
        end
        cxing_st = get_crossings(log.(cyclF) .- log.(Mass_a));
        if length(cxing_st.i1) == 0
            tpoints[k] = t0[1]
            xpoints[k, :] = x0[k, :, 1]
            kpoints[k, :] = [0 0 0]
            
        else
            
            tpoints[k] = t0[cxing_st.i1[1]] .* cxing_st.weight[1] .+ (1.0 - cxing_st.weight[1]) .* t0[cxing_st.i2[1]];
            xpoints[k, :] = (x0[k, :, cxing_st.i1[1]]  .* cxing_st.weight[1] .+  (1.0 - cxing_st.weight[1]) .* x0[k, :, cxing_st.i2[1]])
            kpoints[k, :] = (k0[k, :, cxing_st.i1[1]]  .* cxing_st.weight[1] .+  (1.0 - cxing_st.weight[1]) .* k0[k, :, cxing_st.i2[1]])
        end
        # cyclFOut[k] = cyclotronF(xpoints[k, :], tpoints[k], θm, ωPul, B0, rNS)[1]
        
    end
    
    ωp = GJ_Model_ωp_vec(xpoints, tpoints, θm, ωPul, B0, rNS)
    dOc_grd = grad(cyclotronF_vec(seed(xpoints), tpoints, θm, ωPul, B0, rNS))
    kmag = sqrt.(sum(kpoints .^ 2, dims=2))
    dOc_dl = abs.(sum(kpoints .* dOc_grd, dims=2))
    dOc_dl[kmag .> 0] ./= kmag[kmag .> 0]
    tau = π * ωp .^2 ./ dOc_dl ./ (c_km .* hbar);
    
    if sum(kmag .= 0) > 0
        tau[kmag .== 0] .= 0.0
        
    end
    
    
    return tau
end

# goldreich julian model
function GJ_Model_vec(x, t, θm, ω, B0, rNS; bndry_lyr=false)
    # For GJ model, return \vec{B} and \omega_p [eV]
    # Assume \vec{x} is in spherical coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]
    
    
    r = sqrt.(sum(x .* x, dims=2))
    
    ϕ = atan.(view(x, :, 2), view(x, :, 1))
    θ = acos.(view(x, :, 3)./ r)
    
    ψ = ϕ .- ω.*t
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    
    Bx = Br .* sin.(θ) .* cos.(ϕ) .+ Btheta .* cos.(θ) .* cos.(ϕ) .- Bphi .* sin.(ϕ)
    By = Br .* sin.(θ) .* sin.(ϕ) .+ Btheta .* cos.(θ) .* sin.(ϕ) .+ Bphi .* cos.(ϕ)
    Bz = Br .* cos.(θ) .- Btheta .* sin.(θ)

    nelec = abs.((2.0 .* ω .* Bz) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
    ωp = sqrt.(4 .* π .* nelec ./ 137 ./ 5.0e5);

    # format: [e-, e+] last two -- plasma mass and gamma factor
    
    if bndry_lyr
        nelec_pole = abs.((2.0 .* ω .* B0) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
        pole_val = sqrt.(4 .* π .* nelec ./ 137 ./ 5.0e5);
        ωp[r .>= rNS] .+= pole_val[r .>= rNS] .* exp.( - (r[r .>= rNS] .- rNS) ./ 0.5)
    end
    
    return [Bx By Bz], ωp
end

# computing dωp/dr along axion traj in conversion prob
function dwdr_abs_vec(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    dωdr_grd = grad(GJ_Model_ωp_vec(seed(x0), t_start, θm, ωPul, B0, rNS))
    dωdr_proj = abs.(sum(k0 .* dωdr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    return dωdr_proj
end

function surfNorm(x0, k0, Mvars; return_cos=true)
    # coming in cartesian, so change.
    
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS = Mvars2
    
    
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* k0, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* k0[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* k0[:,1] .+ x0[:,1] .* k0[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    # Switch to celerity in polar coordinates
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2
    AA = (1.0 .- r_s0 ./ rr)
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS)
    
    dωdr_grd = grad(GJ_Model_ωp_vecSPH(seed(x0_pl), t_start, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr))
    snorm = dωdr_grd ./ sqrt.(g_rr .* dωdr_grd[:, 1].^2  .+ g_thth .* dωdr_grd[:, 2].^2 .+ g_pp .* dωdr_grd[:, 3].^2)
    knorm = sqrt.(g_rr .* w0_pl[:,1].^2 .+ g_thth .* w0_pl[:,2].^2 .+ g_pp .* w0_pl[:,3].^2)
    ctheta = (g_rr .* w0_pl[:,1] .* snorm[:, 1] .+ g_thth .* w0_pl[:,2] .* snorm[:, 2] .+ g_pp .* w0_pl[:,3] .* snorm[:, 3]) ./ knorm

    
    # classical...
    # dωdr_grd_2 = grad(GJ_Model_ωp_vec(seed(x0), t_start, θm, ωPul, B0, rNS))
    # snorm_2 = dωdr_grd_2 ./ sqrt.(sum(dωdr_grd_2 .^ 2, dims=2))
    # ctheta = (sum(k0 .* snorm_2, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    # print(ctheta, "\n", cthetaG, "\n\n")
    
    
    if return_cos
        return ctheta
    else
        return ctheta, snorm
    end
end


function d2wdr2_abs_vec(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    dωdr2_grd = grad(dwdr_abs_vec(seed(x0), k0, Mvars))
    dωdr2_proj = abs.(sum(k0 .* dωdr2_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
  
    dwdr = dwdr_abs_vec(x0, k0, Mvars)
    θ = theta_B(x0, k0, Mvars2)
    
    d0dr_grd = grad(theta_B(seed(x0), k0, Mvars2))
    d0dr_proj = abs.(sum(k0 .* d0dr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    
    return (2 ./ tan.(θ) .* d0dr_proj .* dwdr .-  dωdr2_proj) ./ sin.(θ) .^ 2
end

function theta_B(x0, k0, Mvars2)
    θm, ωPul, B0, rNS, gammaF, t_start = Mvars2
    Bvec, ωpL = GJ_Model_vec(x0, t_start, θm, ωPul, B0, rNS)
    return acos.(sum(k0 .* Bvec, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2) .* sum(Bvec.^2, dims=2)) )
end

function Ctheta_B_sphere(x0, k0, Mvars)
    θm, ωPul, B0, rNS, t_start, Mass_NS = Mvars
    Br, Btheta, Bphi = Dipole_SPH(x0, t_start, θm, ωPul, B0, rNS)
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0, Mass_NS)
    #if length(x0[x0[:,1] .<= rNS, 1]) > 0
    #    print(x0[x0[:,1] .<= rNS, 1],"\n\n")
    #end
    Br .*= sqrt.(g_rr)
    Btheta .*= sqrt.(g_thth)
    Bphi .*= sqrt.(g_pp)
    
    Bnorm = sqrt.(Br.^2 ./ g_rr .+ Btheta.^2 ./ g_thth .+ Bphi.^2 ./ g_pp)
    knorm = sqrt.(g_rr .* k0[:,1].^2 .+ g_thth .* k0[:,2].^2 .+ g_pp .* k0[:,3].^2)
    return (k0[:,1] .* Br .+ k0[:,2] .* Btheta .+ k0[:,3] .* Bphi) ./ knorm ./ Bnorm
end

function spatial_dot(vec1, vec2, ntrajs, x0_pl, Mass_NS)
    # assumes both vectors v_mu
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    out_v = zeros(ntrajs)
    for i in 1:ntrajs
        out_v[i] = (g_rr[i] .* vec1[i, 1] .* vec2[i, 1] .+ g_thth[i] .* vec1[i, 2] .* vec2[i, 2] .+ g_pp[i] .* vec1[i, 3] .* vec2[i, 3])
    end
    return out_v
end

function k_sphere(x0, k0, θm, ωPul, B0, rNS, time0, Mass_NS, flat; zeroIn=true, bndry_lyr=false)
    if flat
        Mass_NS = 0.0;
    end
    
    # Define the Schwarzschild radius of the NS (in km)
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2

    # Switch to polar coordinates
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    
    omP = GJ_Model_ωp_vecSPH(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=zeroIn, bndry_lyr=bndry_lyr);
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    
    dr_dt = sum(x0 .* k0, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* k0[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* k0[:,1] .+ x0[:,1] .* k0[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    
    # Switch to celerity in polar coordinates
    AA = (1.0 .- r_s0 ./ rr)
    
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    return w0_pl
end

function angle_vg_sNorm(x0, k0, thetaB, Mvars; return_cos=true)
    # coming in cartesian, so change.
        
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS, flat, isotropic, erg, bndry_lyr = Mvars

    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* k0, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* k0[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* k0[:,1] .+ x0[:,1] .* k0[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    # Switch to celerity in polar coordinates
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2
    AA = (1.0 .- r_s0 ./ rr)
    vg = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2]  ./ AA # lower index defined, [eV, eV * km, eV * km]
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS)
    dωdr_grd = grad(GJ_Model_ωp_vecSPH(seed(x0_pl), t_start, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr))
    snorm = dωdr_grd ./ sqrt.(g_rr .* dωdr_grd[:, 1].^2  .+ g_thth .* dωdr_grd[:, 2].^2 .+ g_pp .* dωdr_grd[:, 3].^2)
    vgNorm = sqrt.(g_rr .* vg[:,1].^2 .+ g_thth .* vg[:,2].^2 .+ g_pp .* vg[:,3].^2)
    
    omP = GJ_Model_ωp_vecSPH(x0_pl, t_start, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr)
    
    ctheta = (g_rr .* vg[:,1] .* snorm[:, 1] .+ g_thth .* vg[:,2] .* snorm[:, 2] .+ g_pp .* vg[:,3] .* snorm[:, 3]) ./ vgNorm
    
    if return_cos
        return ctheta
    else
        return ctheta, snorm
    end
end

function K_par(x0_pl, k_sphere, Mvars; flat=false)
    θm, ωPul, B0, rNS, t_start, Mass_NS = Mvars
    ntrajs = length(x0_pl[:, 1])
    Bsphere = GJ_Model_Sphereical(x0_pl, t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true)
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS)
    if length(size(x0_pl)) > 1
        Bmag = sqrt.((g_rr .* Bsphere[:, 1].^2 .+ g_thth .* Bsphere[:, 2].^2 .+ g_pp .* Bsphere[:, 3].^2))
        k_par = (g_rr .* k_sphere[:,1] .* Bsphere[:, 1] .+ g_thth .* k_sphere[:,2] .* Bsphere[:, 2] .+ g_pp .* k_sphere[:,3] .* Bsphere[:, 3])  ./ Bmag
    else
        Bmag = sqrt.((g_rr .* Bsphere[1].^2 .+ g_thth .* Bsphere[2].^2 .+ g_pp .* Bsphere[3].^2))
        k_par = (g_rr .* k_sphere[1] .* Bsphere[1] .+ g_thth .* k_sphere[2] .* Bsphere[2] .+ g_pp .* k_sphere[3] .* Bsphere[3])  ./ Bmag
    end
    return k_par
end

function dθdr_proj(x0, k0, Mvars)
    d0dr_grd = grad(theta_B(seed(x0), k0, Mvars))
    return abs.(sum(k0 .* d0dr_grd, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
end

# just return net plasma freq
function GJ_Model_ωp_vec(x, t, θm, ω, B0, rNS)
    # For GJ model, return \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]

    r = sqrt.(sum(x .* x, dims=2))
    
    ϕ = atan.(view(x, :, 2), view(x, :, 1))
    θ = acos.(view(x, :, 3)./ r)
    ψ = ϕ .- ω.*t
    
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    
    Bx = Br .* sin.(θ) .* cos.(ϕ) .+ Btheta .* cos.(θ) .* cos.(ϕ) .- Bphi .* sin.(ϕ)
    By = Br .* sin.(θ) .* sin.(ϕ) .+ Btheta .* cos.(θ) .* sin.(ϕ) .+ Bphi .* cos.(ϕ)
    Bz = Br .* cos.(θ) .- Btheta .* sin.(θ)
    
    nelec = abs.((2.0 .* ω .* Bz) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
    ωp = sqrt.(4 .* π .* nelec ./ 137 ./ 5.0e5);

    return ωp
end

function Dipole_SPH(x, t, θm, ω, B0, rNS)
    r = view(x, :, 1)
    
    ϕ = view(x, :, 3)
    θ = view(x, :, 2)
    # print(r, "\n")
    ψ = ϕ .- ω.*t
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    return Br, Btheta, Bphi
end

function GJ_Model_ωp_vecSPH(x, t, θm, ω, B0, rNS; zeroIn=true, bndry_lyr=false)
    # For GJ model, return \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]

    r = view(x, :, 1)
    
    ϕ = view(x, :, 3)
    θ = view(x, :, 2)
    # print(r, "\n")
    ψ = ϕ .- ω.*t
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    
    Bx = Br .* sin.(θ) .* cos.(ϕ) .+ Btheta .* cos.(θ) .* cos.(ϕ) .- Bphi .* sin.(ϕ)
    By = Br .* sin.(θ) .* sin.(ϕ) .+ Btheta .* cos.(θ) .* sin.(ϕ) .+ Bphi .* cos.(ϕ)
    Bz = Br .* cos.(θ) .- Btheta .* sin.(θ)
    
    nelec = abs.((2.0 .* ω .* Bz) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
    ωp = sqrt.(4 .* π .* nelec ./ 137 ./ 5.0e5);
    
    if bndry_lyr
        nelec_pole = abs.((2.0 .* ω .* B0) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
        pole_val = sqrt.(4 .* π .* nelec ./ 137 ./ 5.0e5);
        ωp .+= pole_val .* exp.( - (r .- rNS) ./ 0.5)
    end

    if zeroIn
        ωp[r .<= rNS] .= 0.0;
    end

    return ωp
end

function GJ_Model_ωp_scalar(x, t, θm, ω, B0, rNS)
    # For GJ model, return \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]

    r = sqrt.(sum(x .* x))
    ϕ = atan.(x[2], x[1])
    θ = acos.( x[3]./ r)
    ψ = ϕ .- ω.*t
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    
    Bx = Br .* sin.(θ) .* cos.(ϕ) .+ Btheta .* cos.(θ) .* cos.(ϕ) .- Bphi .* sin.(ϕ)
    By = Br .* sin.(θ) .* sin.(ϕ) .+ Btheta .* cos.(θ) .* sin.(ϕ) .+ Bphi .* cos.(ϕ)
    Bz = Br .* cos.(θ) .- Btheta .* sin.(θ)
    
    nelec = abs.((2.0 .* ω .* Bz) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
    ωp = sqrt.(4 .* π .* nelec ./ 137 ./ 5.0e5);
    

    return ωp
end

function GJ_Model_scalar(x, t, θm, ω, B0, rNS)
    # For GJ model, return \vec{B} and \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]

    r = sqrt.(sum(x .* x))
    ϕ = atan.(x[2], x[1])
    θ = acos.( x[3]./ r)
    ψ = ϕ .- ω.*t
    
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    
    Bx = Br .* sin.(θ) .* cos.(ϕ) .+ Btheta .* cos.(θ) .* cos.(ϕ) .- Bphi .* sin.(ϕ)
    By = Br .* sin.(θ) .* sin.(ϕ) .+ Btheta .* cos.(θ) .* sin.(ϕ) .+ Bphi .* cos.(ϕ)
    Bz = Br .* cos.(θ) .- Btheta .* sin.(θ)
    nelec = abs.((2.0 .* ω .* Bz) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
    ωp = sqrt.(4 .* π .* nelec ./ 137 ./ 5.0e5);

    # format: [e-, e+] last two -- plasma mass and gamma factor
    return [Bx By Bz], ωp
end

# roughly compute conversion surface, so we know where to efficiently sample
function Find_Conversion_Surface(Ax_mass, t_in, θm, ω, B0, rNS, gammaL, relativ)

    rayT = RayTracerGR;
    if θm < (π ./ 2.0)
        θmEV = θm ./ 2.0
    else
        θmEV = (θm .+ π) ./ 2.0
    end
    # estimate max dist
    om_test = GJ_Model_ωp_scalar(rNS .* [sin.(θmEV) 0.0 cos.(θmEV)], t_in, θm, ω, B0, rNS);
    rc_guess = rNS .* (om_test ./ Ax_mass) .^ (2.0 ./ 3.0);
    
    return rc_guess .* 1.01 # add a bit just for buffer
end


###
# ~~~ Energy as function of phase space parameters
###
function ωFree(x, k, t, θm, ωPul, B0, rNS, gammaF)
    # assume simple case where ωp proportional to r^{-3/2}, no time dependence, no magnetic field
    return sqrt.(sum(k .* k, dims = 2) .+ 1e-60 .* sqrt.(sum(x .* x, dims=2)) ./ (rNS.^ 2) )
end

function ωFixedp(x, k, t, θm, ωPul, B0, rNS, gammaF)
    # assume simple case where ωp proportional to r^{-3/2}, no time dependence, no magnetic field

    r = sqrt(sum(x .* x), dims = 2)
    ωp = 1e-6 * (rNS / r)^(3/2)
    k2 = sum(k.*k, dims = 2)

    return sqrt.(k2 .+ ωp^2)
end

function ωSimple(x, k, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, but no magnetic field

    ωpL = GJ_Model_ωp_vec(x, t, θm, ωPul, B0, rNS)
    return sqrt.(sum(k.*k, dims=2) .+ ωpL .^2)
end

function ωSimple_SPHERE(x, k, t, θm, ωPul, B0, rNS, gammaF, Mass_NS)
    #  GJ charge density, but no magnetic field
    ωpL = GJ_Model_ωp_vecSPH(x, t, θm, ωPul, B0, rNS)
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x, Mass_NS);
    return sqrt.((k[:, 1] .* g_rr).^2 .+ (k[:, 2] .* g_thth).^2 .+ (k[:, 3] .* g_pp).^2  .+ ωpL .^2) ./ (-g_tt)
end


function ωNR_e(x, k, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, Magnetic field, non-relativstic e only
    Bvec, ωpL = GJ_Model_vec(x, t, θm, ωPul, B0, rNS)

    kmag = sqrt.(sum(k .* k, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    ωp = sqrt.(sum(ωpL .* ωpL, dims=2))

    cθ = sum(k .* Bvec, dims=2) ./ (kmag .* Bmag)
    
    cθTerm = (1.0 .- 2.0 .* cθ.^2)
    # print(cθTerm)
    # abs not necessary, but can make calculation easier
    return sqrt.(abs.(0.5 .* (kmag .^2 + ωp .^2 + sqrt.(abs.(kmag .^4 + ωp .^4 + 2.0 .* cθTerm .*kmag .^2 .* ωp .^2 )))))

end

function GJ_Model_Sphereical(x, t, θm, ω, B0, rNS; Mass_NS=1.0, flat=false, sphericalX=false)
    # For GJ model, return \vec{B} and \omega_p [eV]
    # Assume \vec{x} is in spherical coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]
    if flat
        Mass_NS = 0.0
    end
    if !sphericalX
        
        r = sqrt.(sum(x .^2 , dims=2))
        ϕ = atan.(view(x, :, 2), view(x, :, 1))
        θ = acos.(view(x, :, 3)./ r)
    else
        r = view(x, :, 1)
        θ = view(x, :, 2)
        ϕ = view(x, :, 3)
    end
    
    ψ = ϕ .- ω.*t
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    
    x0_pl = [r θ ϕ]
  
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS)
    return [Br ./ sqrt.(g_rr) Btheta ./ sqrt.(g_thth)  Bphi ./ sqrt.(g_pp)] # lower?

end


function dwp_ds(xIn, ksphere, Mvars)
    # xIn cartesian, ksphere [spherical]
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS, flat, isotropic, ωErg, bndry_lyr = Mvars
    
    rr = sqrt.(sum(xIn.^2, dims=2))
    x0_pl = [rr acos.(xIn[:,3] ./ rr) atan.(xIn[:,2], xIn[:,1])]
    
    ntrajs = length(t_start)
    omP = GJ_Model_ωp_vecSPH(x0_pl, t_start, θm, ωPul, B0, rNS, zeroIn=true, bndry_lyr=bndry_lyr)
    grad_omP = grad(GJ_Model_ωp_vecSPH(seed(x0_pl), t_start, θm, ωPul, B0, rNS, zeroIn=true, bndry_lyr=bndry_lyr));
    
    
    
    Bsphere = GJ_Model_Sphereical(xIn, t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat)
    
    kmag = sqrt.(spatial_dot(ksphere, ksphere, ntrajs, x0_pl, Mass_NS));
    dz_op = spatial_dot(ksphere ./ kmag, grad_omP, ntrajs, x0_pl, Mass_NS)
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    kB_norm = spatial_dot(Bsphere, ksphere ./ kmag, ntrajs, x0_pl, Mass_NS)
    
    v_ortho = -(Bsphere .- kB_norm .* ksphere ./ kmag)
    v_ortho ./= sqrt.(spatial_dot(v_ortho, v_ortho, ntrajs, x0_pl, Mass_NS))
    
    dy_op = spatial_dot(v_ortho, grad_omP, ntrajs, x0_pl, Mass_NS)
    Bmag = sqrt.(spatial_dot(Bsphere, Bsphere, ntrajs, x0_pl, Mass_NS));
    ctheta_B = spatial_dot(Bsphere , ksphere, ntrajs, x0_pl, Mass_NS) ./ (kmag .* Bmag)
    stheta_B = sin.(acos.(ctheta_B))
    if isotropic
        ctheta_B .*= 0.0
        stheta_B ./= stheta_B
    end
    
    xi = stheta_B .^2 ./ (1.0 .- ctheta_B.^2 .* omP.^2 ./ ωErg.^2)
    w_prime = dz_op .+ omP.^2 ./ ωErg.^2 .* xi ./ (stheta_B ./ ctheta_B) .* dy_op
    
    # group velocity based on millar
    snorm = grad_omP ./ sqrt.(g_rr .* grad_omP[:, 1].^2  .+ g_thth .* grad_omP[:, 2].^2 .+ g_pp .* grad_omP[:, 3].^2)
    vec2 = ksphere ./ kmag .+ omP.^2 ./ ωErg.^2 .* xi ./ (stheta_B ./ ctheta_B) .* v_ortho
    vec2Norm = sqrt.(spatial_dot(vec2, vec2, ntrajs, x0_pl, Mass_NS));
    angleVal = acos.(spatial_dot(vec2 ./ vec2Norm, snorm, ntrajs, x0_pl, Mass_NS))
    kdotN = spatial_dot(ksphere, snorm, ntrajs, x0_pl, Mass_NS)
    
    # this is group velocity based on dwdk
    ωErg_inf = ωErg .* sqrt.(g_rr)
    
    test = grad(omega_function(x0_pl, seed(ksphere), t_start, -ωErg_inf, θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=true, bndry_lyr=bndry_lyr)) #
    test[:, 1] ./= g_rr
    test[:, 2] ./= g_thth
    test[:, 3] ./= g_pp
    vgN = sqrt.(spatial_dot(test, test, ntrajs, x0_pl, Mass_NS));
    dwdk_snorm = acos.(spatial_dot(test ./ vgN, snorm, ntrajs, x0_pl, Mass_NS))
    
    if !isotropic
        return abs.(w_prime), angleVal, kdotN, dwdk_snorm, test ./ vgN, vgN
    else
       return abs.(dz_op), angleVal, kdotN, dwdk_snorm, test ./ vgN, vgN
    end
end



function dk_ds(x0, k0, Mvars)
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start, ωErg = Mvars2
    
    dkdr_grd = grad(kNR_e(seed(x0), k0, ωErg, t_start, θm, ωPul, B0, rNS, gammaF))
    
    Bvec, ωpL = GJ_Model_vec(x0, t_start, θm, ωPul, B0, rNS)
    
    kmag = sqrt.(sum(k0 .* k0, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    cθ = sum(k0 .* Bvec, dims=2) ./ (kmag .* Bmag)
    khat = k0 ./ kmag
    Bhat = Bvec ./ Bmag
    
    dkdr_proj_s = zeros(length(k0[:, 1]));
    for i in 1:length(k0[:,1])
        uvec = [k0[i,2] .* Bvec[i,3] - Bvec[i,2] .* k0[i,3],  k0[i,3] .* Bvec[i,1] - Bvec[i,3] .* k0[i,1], k0[i,1] .* Bvec[i,2] - Bvec[i,1] .* k0[i,2]] ./ Bmag[i] ./ kmag[i]
        uhat = uvec ./ sqrt.(sum(uvec .^ 2));
        R = [uhat[1].^2 uhat[1] .* uhat[2] .+ uhat[3] uhat[1] .* uhat[3] .- uhat[2]; uhat[1] .* uhat[2] .- uhat[3] uhat[2].^2 uhat[2] .* uhat[3] .+ uhat[1]; uhat[1].*uhat[3] .+ uhat[2] uhat[2].*uhat[3] .- uhat[1] uhat[3].^2];
        # shat = R * Bhat[i, :];
        yhat = R * khat[i, :];
        
        # dkdr_proj_s[i] = abs.(sum(shat .* dkdr_grd[i, :]));
        dkdz = (sum(khat[i, :] .* dkdr_grd[i, :]));
        dkdy = (sum(yhat .* dkdr_grd[i, :]));
        dkdr_proj_s[i] = (dkdz .+ dkdy .* sin.(acos.(cθ[i])).^2 .* ωpL[i].^2 ./ (ωErg[i].^2 .- ωpL[i].^2 .* cθ[i].^2) ./ tan.(acos.(cθ[i])))
        
    end
    
    return abs.(dkdr_proj_s)
end


function kNR_e(x, k, ω, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, Magnetic field, non-relativstic e only
    Bvec, ωpL = GJ_Model_vec(x, t, θm, ωPul, B0, rNS)

    kmag = sqrt.(sum(k .* k, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    ωp = sqrt.(sum(ωpL .* ωpL, dims=2))

    
    # abs not necessary, but can make calculation easier (numerical error can creap at lvl of 1e-16 for ctheta)
    return sqrt.(abs.(ω.^2 .-  ωp.^2))
end

function ωGam(x, k, t, θm, ωPul, B0, rNS, gammaF)
    #  GJ charge density, Magnetic field, thermal single species plasma (assume thermal is first species!)

    Bvec, ωpL = GJ_Model_vec(x, t, θm, ωPul, B0, rNS)
    gam = gammaF[1]

    kmag = sqrt.(sum(k .* k, dims=2))
    Bmag = sqrt.(sum(Bvec .* Bvec, dims=2))
    ωp = sqrt.(sum(ωpL .* ωpL, dims=2))

    cθ = sum(k .* Bvec, dims=2) ./ (kmag .* Bmag)
    sqr_fct = sqrt.(kmag.^4 .* (gam .^2 .- cθ.^2 .* (gam.^2 .- 1)).^2 .-
        2 .* kmag .^2 .* gam .* (cθ.^2 .+ (cθ.^2 .- 1) .* gam.^2) .* ωp.^2 .+ gam.^2 .* ωp.^4)
    ω_final = sqrt.((kmag.^2 .*(gam.^2 .+ cθ.^2 .*(gam.^2 .- 1)) .+ gam.*ωp.^2 .+ sqr_fct) ./ (2 .* gam.^2))

    return ω_final
end

function find_samples(maxR, ntimes_ax, θm, ωPul, B0, rNS, Mass_a, Mass_NS; n_max=8, batchsize=2, thick_surface=false, iso=false, melrose=false, bndry_lyr=false)

    tt_ax = LinRange(-2*maxR, 2*maxR, ntimes_ax); # Not a real physical time -- just used to get trajectory crossing
    cxing = nothing
    
    # randomly sample angles θ, ϕ, hit conv surf
    θi = acos.(1.0 .- 2.0 .* rand(batchsize));
    ϕi = rand(batchsize) .* 2π;
    vvec_all = [sin.(θi) .* cos.(ϕi) sin.(θi) .* sin.(ϕi) cos.(θi)];
    
    # local velocity
    θi_loc = acos.(1.0 .- 2.0 .* rand(batchsize));
    ϕi_loc = rand(batchsize) .* 2π;
    vvec_loc = [sin.(θi_loc) .* cos.(ϕi_loc) sin.(θi_loc) .* sin.(ϕi_loc) cos.(θi_loc)];
    
    # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
    ϕRND = rand(batchsize) .* 2π;
    
    rRND = sqrt.(rand(batchsize)) .* maxR; # standard flat sampling
    # rRND = rand(batchsize) .* maxR; # New 1/r sampling
    x1 = rRND .* cos.(ϕRND);
    x2 = rRND .* sin.(ϕRND);
    # rotate using Inv[EurlerMatrix(ϕi, θi, 0)] on vector (x1, x2, 0)
    x0_all= [x1 .* cos.(-ϕi) .* cos.(-θi) .+ x2 .* sin.(-ϕi) x2 .* cos.(-ϕi) .- x1 .* sin.(-ϕi) .* cos.(-θi) x1 .* sin.(-θi)];
    x_axion = [transpose(x0_all[i,:]) .+ transpose(vvec_all[i,:]) .* tt_ax[:] for i in 1:batchsize];
    
    vIfty = (220.0 .+ rand(batchsize, 3) .* 1.0e-5) ./ sqrt.(3);
    # vIfty = 220.0 .* erfinv.( 2.0 .* rand(batchsize, 3) .- 1.0);
    
    vIfty_mag = sqrt.(sum(vIfty.^2, dims=2));
    gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ./ c_km).^2 )
    erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag ./ c_km .* gammaA).^2)
    Mass_NS = 1.0

    if !thick_surface
        cxing_st = [get_crossings(log.(GJ_Model_ωp_vec(x_axion[i], 0.0, θm, ωPul, B0, rNS)) .- log.(Mass_a)) for i in 1:batchsize];
        cxing = [apply(cxing_st[i], tt_ax) for i in 1:batchsize];
    else
        
        for i in 1:batchsize
            valF, truth_vals, minV = test_on_shell(x_axion[i], vvec_loc[i,:], vIfty_mag[i], 0.0, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=iso, melrose=melrose, bndry_lyr=bndry_lyr)
            # print("min V: \t", minV, "\n")
            cxing_st = [get_crossings(valF, keep_all=true) for i in 1:batchsize];
            tt_axNew = tt_ax[truth_vals]
            
            if isnothing(cxing)
                cxing = [apply(cxing_st[i], tt_axNew)]
            else
                cxing = [cxing; [apply(cxing_st[i], tt_axNew)]]
            end
        end
        
        # cxing_stW = [get_crossings(log.(GJ_Model_ωp_vec(x_axion[i], 0.0, θm, ωPul, B0, rNS)) .- log.(Mass_a)) for i in 1:batchsize];
        # cxingW = [apply(cxing_stW[i], tt_ax) for i in 1:batchsize];
        # print(cxingW, "\n", cxing, "\n\n")
        
    end
    
    
    randInx = [rand(1:n_max) for i in 1:batchsize];
    
    # see if keep any crossings
    indx_cx = [if length(cxing[i]) .>= randInx[i] i else -1 end for i in 1:batchsize];

    
    # remove those which dont
    randInx = randInx[indx_cx .> 0];
    indx_cx_cut = indx_cx[indx_cx .> 0];

    cxing_short = [cxing[indx_cx_cut][i][randInx[i]] for i in 1:length(indx_cx_cut)];
    weights = [length(cxing[indx_cx_cut][i]) for i in 1:length(indx_cx_cut)];
    
    
    numX = length(cxing_short);
    R_sample = vcat([rRND[indx_cx_cut][i] for i in 1:numX]...);
    erg_inf_ini = vcat([erg_inf_ini[indx_cx_cut][i] for i in 1:numX]...);

    
    vvec_loc = vvec_loc[indx_cx_cut, :];
    vIfty_mag = vcat([vIfty_mag[indx_cx_cut][i] for i in 1:numX]...);
    
    cxing = nothing
    if numX != 0
        
        xpos = [transpose(x0_all[indx_cx_cut[i], :]) .+ transpose(vvec_all[indx_cx_cut[i], :]) .* cxing_short[i] for i in 1:numX];
        vvec_full = [transpose(vvec_all[indx_cx_cut[i],:]) .* ones(1, 3) for i in 1:numX];
        
        
        t_new_arr = LinRange(- abs.(tt_ax[5] - tt_ax[1]), abs.(tt_ax[5] - tt_ax[1]), 1000);
        xpos_proj = [xpos[i] .+ vvec_full[i] .* t_new_arr[:] for i in 1:numX];

        
        if !thick_surface
            cxing_st = [get_crossings(log.(GJ_Model_ωp_vec(xpos_proj[i], 0.0, θm, ωPul, B0, rNS)) .- log.(Mass_a)) for i in 1:numX];
            cxing = [apply(cxing_st[i], t_new_arr) for i in 1:numX];
        else
            for i in 1:numX
                valF, truth_vals, minV = test_on_shell(xpos_proj[i], vvec_loc[i,:], vIfty_mag[i], 0.0, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=iso, melrose=melrose, bndry_lyr=bndry_lyr)
                # print("min V: \t", minV, "\n")
                cxing_st = [get_crossings(valF, keep_all=true) for i in 1:batchsize];
                tt_axNew = t_new_arr[truth_vals]
                if isnothing(cxing)
                    cxing = [apply(cxing_st[i], tt_axNew)]
                else
                    cxing = [cxing; [apply(cxing_st[i], tt_axNew)]]
                end
            end
        
        # cxing_st = [get_crossings(test_on_shell(xpos_proj[i], vvec_loc[i, :], vIfty_mag[i], 0.0,  θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=iso, melrose=melrose, bndry_lyr=bndry_lyr)) for i in 1:numX];
        end
    
        
        
        indx_cx = [if length(cxing[i]) .> 0 i else -1 end for i in 1:numX];
        indx_cx_cut = indx_cx[indx_cx .> 0];
        R_sample = R_sample[indx_cx_cut];
        erg_inf_ini = erg_inf_ini[indx_cx_cut];
        vvec_loc = vvec_loc[indx_cx_cut, :];
        vIfty_mag = vIfty_mag[indx_cx_cut];
        
        numX = length(indx_cx_cut);
        if numX == 0
            return 0.0, 0.0, 0, 0.0, 0.0, 0.0
        end



        randInx = [rand(1:length(cxing[indx_cx_cut][i])) for i in 1:numX];
        cxing = [cxing[indx_cx_cut][i][randInx[i]] for i in 1:numX];
        vvec_flat = reduce(vcat, vvec_full);
       
        xpos = [xpos[indx_cx_cut[i],:] .+ vvec_full[indx_cx_cut[i],:] .* cxing[i] for i in 1:numX];
        vvec_full = [vvec_full[indx_cx_cut[i],:] for i in 1:numX];

        try
            xpos_flat = reduce(vcat, xpos);
        catch
            print("why is this a rare fail? \t", xpos, "\n")
        end
   
        try
            xpos_flat = reduce(vcat, xpos_flat);
            vvec_flat = reduce(vcat, vvec_full);
        catch
            print("for some reason reduce fail... ", vvec_full, "\t", xpos_flat, "\n")
            vvec_flat = vvec_full;
        end

       
        rmag = sqrt.(sum(xpos_flat .^ 2, dims=2));
        indx_r_cut = rmag .> rNS; #
        # print(xpos_flat, "\t", vvec_flat,"\t", R_sample, "\t", indx_r_cut, "\n")
        if sum(indx_r_cut) - length(xpos_flat[:,1 ]) < 0
            xpos_flat = xpos_flat[indx_r_cut[:], :]
            vvec_flat = vvec_flat[indx_r_cut[:], :]
            R_sample = R_sample[indx_r_cut[:]]
            erg_inf_ini = erg_inf_ini[indx_r_cut[:]];
            vvec_loc = vvec_loc[indx_r_cut[:], :];
            vIfty_mag = vIfty_mag[indx_r_cut[:]];
        
            numX = length(xpos_flat);
            rmag = sqrt.(sum(xpos_flat .^ 2, dims=2));
        end
        
        ntrajs = length(R_sample)
        if ntrajs == 0
            return 0.0, 0.0, 0, 0.0, 0.0, 0.0
        end
        
        
        # Renormalize loc velocity and solve asymptotic
        
        vmag_loc = sqrt.(vIfty_mag.^2 .+ 2 .* GNew .* Mass_NS ./ rmag) ./ c_km
       
        v0 = vvec_loc .* vmag_loc
        
        vIfty = zeros(ntrajs, 3)
        ϕ = atan.(xpos_flat[:, 2], xpos_flat[:, 1])
        θ = acos.(xpos_flat[:, 3] ./ rmag)
       
        for i in 1:ntrajs
            for j in 1:3
                vIfty[i, j] = v_infinity(θ[i], ϕ[i], rmag[i], transpose(v0[i, :]); v_comp=j, Mass_NS=Mass_NS);
            end
        end
        
        ωpL = GJ_Model_ωp_vec(xpos_flat, zeros(ntrajs), θm, ωPul, B0, rNS)
        if ntrajs > 1
            vtot = sqrt.(sum(v0.^2, dims=2))
        else
            vtot = sqrt.(sum(v0.^2))
        end
        gamF = 1 ./ sqrt.(1.0 .- vtot.^2)
        erg_ax = Mass_a .* sqrt.(1.0 .+ (gamF .* vtot).^2) ;
        

      
        # make sure not in forbidden region....
        fails = ωpL .> erg_ax;
        n_fails = sum(fails);
        
        
        
        cxing = nothing;
        if n_fails > 0
            try
                vvec_flat = reduce(vcat, vvec_flat);
            catch
                vvec_flat = vvec_flat;
            end
            print("fails... \n")
      
            if !thick_surface
                ωpLi2 = [if fails[i] == 1 erg_ax .- GJ_Model_ωp_vec(transpose(xpos_flat[i,:]) .+ transpose(vvec_flat[i,:]) .* t_new_arr[:], [0.0], θm, ωPul, B0, rNS) else -1 end for i in 1:ntrajs];
                ωpLi2 = transpose(reduce(vcat, ωpLi2));
                
                t_new = [if length(ωpLi2[i]) .> 1 t_new_arr[findall(x->x==ωpLi2[i][ωpLi2[i] .> 0][argmin(ωpLi2[i][ωpLi2[i] .> 0])], ωpLi2[i])][1] else -1e6 end for i in 1:length(ωpLi2)];
                t_new = t_new[t_new .> -1e6];
                xpos_flat[fails[:],:] .+= vvec_flat[fails[:], :] .* t_new;
                
            else
                xpos_proj = [transpose(xpos_flat[i,:]) .+ transpose(vvec_flat[i,:]) .* t_new_arr[:] for i in 1:ntrajs];
                ## FIXING
                for i in 1:numX
                    valF, truth_vals, minV = test_on_shell(xpos_proj[i], vvec_loc[i,:], vIfty_mag[i], 0.0, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=iso, melrose=melrose, bndry_lyr=bndry_lyr)
                    cxing_st = [get_crossings(valF, keep_all=true) for i in 1:batchsize];
                    tt_axNew = t_new_arr[truth_vals]
                    if isnothing(cxing)
                        cxing = [apply(cxing_st[i], tt_axNew)]
                    else
                        cxing = [cxing; [apply(cxing_st[i], tt_axNew)]]
                    end
                end
                
                xpos_flat = [xpos_flat[i,:] .+ vvec_flat[i,:] .* cxing[i] for i in 1:ntrajs];
                try
                    xpos_flat = reduce(vcat, xpos_flat);
                catch
                    xpos_flat = xpos_flat;
                end
            end
            
        end
        
        return xpos_flat, R_sample, ntrajs, weights, v0, vIfty
    else
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0
    end
    
end



end


function dist_diff(xfin)
    b = zeros(size(xfin[:,1,:]))
    b[:, 1:end-1] = abs.((sqrt.(sum(xfin[:, :, 2:end] .^ 2, dims=2)) .- sqrt.(sum(xfin[:, :, 1:end-1] .^ 2, dims=2)) )) ./ c_km ./ hbar # 1 / eV
    b[end] = b[end-2]
    return b
end

function ConvL_weights(xfin, kfin, v_sur, tt, conL, Mvars)
    # xfin and kfin from runner
    # tt from time list, conL [km], and vel at surface (unitless)
    func_use, MagnetoVars, Mass_a = Mvars
    RT = RayTracerGR
    
    dR = dist_diff(xfin)
    ntimes = length(tt)
    nph = length(xfin[:,1,1])
    phaseS = zeros(nph, ntimes)
    for i in 1:ntimes
        Bvec, ωpL = RT.GJ_Model_vec(xfin[:,:,i], tt[i], MagnetoVars[1], MagnetoVars[2], MagnetoVars[3], MagnetoVars[4])
        thetaB_hold = sum(kfin[:,:,i] .* Bvec, dims=2) ./ sqrt.(sum(Bvec .^ 2, dims=2) .* sum(kfin[:,:,i] .^ 2, dims=2))
        if sum(thetaB_hold .> 1) > 0
            thetaB_hold[thetaB_hold .> 1] .= 1.0;
        end
        thetaB = acos.(thetaB_hold)
        thetaZ_hold = sqrt.( sum(kfin[:,:,i] .* kfin[:,:,1], dims=2) .^2 ./ sum(kfin[:,:,1] .^ 2, dims=2) ./ sum(kfin[:,:,i] .^ 2, dims=2))
        if sum(thetaZ_hold .> 1) > 0
            thetaZ_hold[thetaZ_hold .> 1] .= 1.0;
        end
        thetaZ = acos.(thetaZ_hold)
        ωfull = func_use(xfin[:, :, i], kfin[:, :, i], tt[i], MagnetoVars[1], MagnetoVars[2], MagnetoVars[3], MagnetoVars[4], MagnetoVars[5])
        phaseS[:, i] = dR[:, i] .* (Mass_a .* v_sur .- cos.(thetaZ) .* sqrt.( abs.((ωfull .^2 .-  ωpL .^ 2 ) ./ (1 .- cos.(thetaB) .^2 .* ωpL .^2 ./ ωfull .^2)))  )
        
    end
    δphase = cumsum(phaseS, dims=2)
    weights = zeros(nph)
    for i in 1:nph
        if abs.(δphase[i,1]) .> (π ./ 2)
            weights[i] = 0.0
        else
            cx_list = RT.get_crossings(abs.(δphase[i,:]) .- π ./ 2 )
            convL_real = sum(dR[i, 1:cx_list.i2[1]]) .* hbar .* c_km

            weights[i] = convL_real[1] ./ conL[i]
        end
        # print(weights[i], "\n")
    end
    return weights
end
