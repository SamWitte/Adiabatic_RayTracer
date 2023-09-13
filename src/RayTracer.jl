__precompile__()


include("Constants.jl")
import .Constants: c_km, hbar, GNew

module RayTracerGR
import ..Constants: c_km, hbar, GNew

using ForwardDiff: gradient, derivative, Dual, Partials, hessian
using OrdinaryDiffEq
using DiffEqBase
using SpecialFunctions
using LinearAlgebra: cross, det
using NLsolve


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
        
        
        θm, ωPul, B0, rNS, gammaF, time0, Mass_NS, Mass_a, erg, flat, isotropic, melrose, bndry_lyr = Mvars;
        if flat
            Mass_NS = 0.0;
        end
        time = time0 .+  t;
        
        g_tt, g_rr, g_thth, g_pp = g_schwartz(view(u, :, 1:3), Mass_NS);
        
        du[:, 4:6] .= -grad(hamiltonian(seed(view(u, :, 1:3)), view(u, :, 4:6) .* erg , time[1], -view(u, :, 7), θm, ωPul, B0, rNS, Mass_NS, Mass_a, iso=isotropic, melrose=melrose)) .* c_km .* t .* (g_rr ./ -view(u, :, 7)) ./ erg;
        du[:, 1:3] .= grad(hamiltonian(view(u, :, 1:3), seed(view(u, :, 4:6)  .* erg ), time[1], -view(u, :, 7), θm, ωPul, B0, rNS, Mass_NS, Mass_a, iso=isotropic, melrose=melrose)) .* c_km .* t .* (g_rr ./ -view(u, :, 7));
        du[u[:,1] .<= rNS .* 1.01, :] .= 0.0;
        
        du[:,7 ] .= derivative(tI -> hamiltonian(view(u, :, 1:3), view(u, :, 4:6)  .* erg , tI, -view(u, :, 7), θm, ωPul, B0, rNS, Mass_NS, Mass_a, iso=isotropic, melrose=melrose, bndry_lyr=bndry_lyr), time[1])[:] .* t .* (g_rr[:] ./ -view(u, :, 7));
    
    end
end

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# compute axion trajectories
function func_axion!(du, u, Mvars, lnt)
    @inbounds begin
        t = exp.(lnt);
        
        
        θm, ωPul, B0, rNS, gammaF, time0, Mass_NS, erg, flat, isotropic, melrose, mass_axion = Mvars;
        # Mass_NS_in = Mass_NS .* ones(length(u[:,1]))
        # Mass_NS_in[u[:,1] .<= rNS] .= Mass_NS_in .* u[:,1][u[:,1] .<= rNS].^3 ./ rNS.^3
        if flat
            Mass_NS = 0.0;
        end
        time = time0 .+  t;
       
        
        g_tt, g_rr, g_thth, g_pp = g_schwartz(view(u, :, 1:3), Mass_NS);


        du[:, 4:6] .= -grad(hamiltonian_axion(seed(view(u, :, 1:3)),
                       view(u, :, 4:6) .* erg , time[1], erg, θm, ωPul, B0, rNS,
                       Mass_NS, mass_axion, iso=isotropic, melrose=melrose)) .*
                           c_km .* t .* (g_rr ./ erg) ./ erg;
        du[:, 1:3] .= grad(hamiltonian_axion(view(u, :, 1:3),
              seed(view(u, :, 4:6)  .* erg ), time[1], erg, θm, ωPul,
              B0, rNS, Mass_NS, mass_axion, iso=isotropic, melrose=melrose)) .*
                    c_km .* t .* (g_rr ./ erg);
        du[:, 7] .= 0.0
        
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
  times
end
# Constructor
node(x0=0.,y0=0.,z0=0.,kx0=0.,ky0=0.,kz0=0.,t0=0.,Δω0=-1.0,
     species0="axion",prob0=0.,
     weight0=0.,parent_weight0=0.,prob_conv=0.,prob_conv0=0.) = node(
      x0,y0,z0,kx0,ky0,kz0,t0,Δω0,species0,
      prob0,weight0,parent_weight0,prob_conv,prob_conv0,
      [],[],[],[],[],[],[],[],[],false,[],[],[],[])
#node(x=0.,y=0.,z=0.,kx=0.,ky=0.,kz=0.,t=0.,species="axion",prob=0.,weight=0.,
#     parent_weight=0.,xc=[],yc=[],zc=[],kxc=[],kyc=[],kzc=[],tc=[],Pc=[],
#    is_final=false,traj=[],mom=[]) = node(x,y,z,kx,ky,kz,t,species,prob,weight,
#     parent_weight,xc,yc,zc,kxc,kyc,kzc,tc,Pc,is_final,traj,mom)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# propogate photon module
function propagate(x0::Matrix, k0::Matrix,  nsteps, Mvars, NumerP, rhs=func!,
    make_tree=false, is_axion=false, Mass_a=1e-6, max_crossings=3, Δω=-1.0)
    ln_tstart, ln_tend, ode_err = NumerP
    
    tspan = (ln_tstart, ln_tend)
    saveat = (tspan[2] .- tspan[1]) ./ (nsteps-1)
   
    
    if is_axion 
      θm,ωPul,B0,rNS,gammaF,time0,Mass_NS,erg,flat,isotropic,melrose,Mass_a,bndry_lyr=Mvars;
      k0 = k_norm_Cart(x0, k0, time0, erg, θm, ωPul, B0, rNS, Mass_NS, Mass_a, melrose=melrose, flat=flat, isotropic=isotropic, is_photon=false, bndry_lyr=bndry_lyr)
    else
      θm,ωPul,B0,rNS,gammaF,time0,Mass_NS,Mass_a,erg,flat,isotropic,melrose,bndry_lyr=Mvars;
      
      k0 = k_norm_Cart(x0, k0, time0, erg, θm, ωPul, B0, rNS, Mass_NS, Mass_a, melrose=melrose, flat=flat, isotropic=isotropic, is_photon=true, bndry_lyr=bndry_lyr, ax_fix=true)
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
    
    omP = GJ_Model_ωp_vecSPH(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=true, bndry_lyr=bndry_lyr, Mass_a=Mass_a);
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* k0, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* k0[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* k0[:,1] .+ x0[:,1] .* k0[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    
    # Switch to celerity in polar coordinates
    AA = (1.0 .- r_s0 ./ rr)
    
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    w0_pl .*= 1.0 ./ erg # switch to dt and renormalize for order 1 vals
    
    # Define initial conditions so that u0[1] returns a list of x positions (again, 1 entry for each axion trajectory) etc.
    # Δω: relative energy change (is negative)
    u0 = ([x0_pl w0_pl erg .* Δω])
    # u0 = ([x0_pl w0_pl -erg])
    # u0 = ([x0_pl w0_pl])
    
#    if !is_axion
#        test = hamiltonian(x0_pl, w0_pl .* erg, time0, erg, θm, ωPul, B0, rNS, Mass_NS; iso=isotropic, melrose=melrose, zeroIn=false) ./ erg.^2
#        print("test \t", test, "\t", omP, "\n")
#    end
    

    function out_domain(u, Mvars, lnt)
        r_s0 = 2.0 * Mass_NS * GNew / c_km^2

        # If the photon is inside the NS, it will be killed by a callback
        # function soon. We therefore simply use abs
        AA = sqrt.(abs.(1.0 .- r_s0 ./ u[:, 1]))
        testCond = (-u[:,7] ./ AA .- GJ_Model_ωp_vecSPH(u, exp.(lnt), θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a)) ./ abs.(u[:,7] )

        if (sum(testCond .< 0) .> 0.0)||(sum(u[:,1] .<= rNS) .> 0)
            return true
        else
            return false
        end
    end
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Stop integration when a given number of level crossings is achieved
    # Multiple callbacks -> CallbackSet
    cut_short = false
    # Store crossings to be used later
    xc = []; yc = []; zc = []
    kxc = []; kyc = []; kzc = []
    tc = []; Δωc = []
    callback_count = 0

    if make_tree
      
      # Cut after given amount of crossings
      function condition(u, lnt, integrator)
        thick_surface = true # TODO: Make as input parameter in propagate
        if !thick_surface
          return (GJ_Model_ωp_vecSPH(u, exp.(lnt), θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a)
                .- Mass_a)[1]
        else

          
          x0_pl = [u[1] u[2] u[3]]
          w0_pl = [u[4] u[5] u[6]]
          #v0_pl = v0_pl ./ u[4]
          rr = u[1]
          erg_inf = u[7]
          t0 = exp.(lnt)

          r_s0 = 2.0 * Mass_NS * GNew / c_km^2
          
          AA = (1.0 .- r_s0 ./ rr)
          if rr .< rNS
                AA = 1.0
            end
          
          #w0_pl = [v0_pl[:, 1] ./ sqrt.(AA)   v0_pl[:, 2] ./ rr .* rr.^2  v0_pl[:, 3] ./ (rr .* sin.(x0_pl[:, 2])) .* (rr .* sin.(x0_pl[:, 2])).^2 ] ./ AA 
          #w0_pl .*= 1.0 ./ erg           

          g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);

          NrmSq = (-erg_inf.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp )
          w0_pl .*= sqrt.(NrmSq)
          
          omP = GJ_Model_ωp_vecSPH(u, exp.(lnt), θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a)
          
          if isotropic
              kpar = 0.0
          else
              kpar = K_par(x0_pl, w0_pl, [θm, ωPul, B0, rNS, t0, Mass_NS, Mass_a])
          end
          ksqr = g_tt .* erg_inf.^2 .+ g_rr .* w0_pl[:, 1].^2 .+ g_thth .* w0_pl[:, 2].^2 .+ g_pp .* w0_pl[:, 3].^2
          Ham = 0.5 .* (ksqr .+ omP.^2 .* (erg_inf.^2 ./ g_rr .- kpar.^2) ./  (erg_inf.^2 ./ g_rr)  ) ./ (erg_inf.^2);
          
          #print("  : ", u[1:3], " " , w0_pl, " ", exp.(lnt), " ", θm, " ", ωPul, " ", B0, " ", rNS, " ")  
          #print(Ham[1], "\n")  
          return Ham[1]
        end
      end
      
      callback_count = 0
      function affect!(i)

          if callback_count == 0
            # If i.u has not changed, it is not a new crossings...
            # There is now a way to do this with a input into the
            # condition.
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
          push!( Δωc, (i.u[7]/erg[1]) )
          
          #print("Test \t ", i.u, "\t", GJ_Model_ωp_vecSPH(i.u, exp.(i.t), θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a), "\t",bndry_lyr,  " ",  (log.(GJ_Model_ωp_vecSPH(i.u, exp.(i.t), θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a))
           #     .- log.(Mass_a))[1], "\n")
         
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
          callback_count += 1
          if callback_count >= max_crossings
              cut_short = true
              terminate!(i)
          end
      end
      # Cut if inside a neutron star (and a photon). 
      condition_r(u,lnt,integrator) = u[1] < (rNS*1.01)
      affect_r!(integrator) = terminate!(integrator)
      
       
     
      cb_s = ContinuousCallback(condition, affect!,
                  rootfind=true, interp_points=50)
      cb_r = DiscreteCallback(condition_r, affect_r!)
    
      if is_axion
        cbset = CallbackSet(cb_s) # cb->reflection, cb_->NS, not for axion
        prob = ODEProblem(rhs, u0, tspan, Mvars, callback=cbset)
      else
        cbset = CallbackSet(cb_s, cb_r)

        prob = ODEProblem(rhs, u0, tspan, Mvars, callback=cbset)

      end

      # prob = ODEProblem(rhs, u0, tspan, Mvars, callback=cbset)
      # prob = ODEProblem(rhs, u0, tspan, Mvars)
    else
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Define the ODEproblem
      prob = ODEProblem(rhs, u0, tspan, Mvars)
    end
    #print(u0, " ", Mvars, " ", tspan, "\n")
    # Solve the ODEproblem



    sol = solve(prob, Vern6(), saveat=saveat, reltol=1e-7, abstol=ode_err,
                dtmin=1e-13, force_dtmin=true, maxiters=1e5)

    if (sol.retcode != :Success)&&(sol.retcode != :Terminated)
        print("problem? \n", x0, "\n", k0, "\n ", omP, "\t", tspan, "\n\n")
        print(u0, "\t", size(u0), "\t", typeof(u0), "\n")
        print(sol.u, "\n\n")
        print(sol.retcode, "\n")
    end

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
    fail_indx[sphere_c[:, 1, end] .<= (rNS .* 1.01)] .= 0.0
    
    # dxdtau = dxdtau[sphere_c[:, 1, end] .> rNS, :, :]
    # ω_reshaped = ω_reshaped[sphere_c[:, 1, end] .> rNS]
    
    # dt = [sol.u[i][:, 7] for i in 1:length(sol.u)]
    # Also return the list of (proper) times at which the solution is saved for pinpointing the seeding time
    times = sol.t
    
   
    if make_tree
      return x_reshaped,v_reshaped,dt,fail_indx,cut_short,xc,yc,zc,kxc,kyc,kzc,tc,Δωc,times
    else
      return x_reshaped,v_reshaped,dt,fail_indx
    end
end


function g_schwartz(x0, Mass_NS; rNS=10.0)
    # (1 - r_s / r)

    # notation (-,+,+,+), upper g^mu^nu
    
    if length(size(x0)) > 1
        r = x0[:,1]
        rs = ones(eltype(r), size(r)) .* 2 * GNew .* Mass_NS ./ c_km.^2
        rs[r .<= rNS] .*= (r[r .<= rNS] ./ rNS).^3
        sin_theta = sin.(x0[:,2])
    else
        rs = 2 * GNew .* Mass_NS ./ c_km.^2
        r = x0[1]
        if r <= rNS
            rs .*= (r ./ rNS).^3
        end
        sin_theta = sin.(x0[2])
    end
    
    # r = x0[:,1]
    
    # Reduced NS mass is done elsewhere...
    # Mass_NS = Mass_NS_in .* ones(length(r))
    # Mass_NS[r .<= rNS] .= Mass_NS_in .* r[r .<= rNS].^3 ./ rNS.^3


    # rs = 2 * GNew .* Mass_NS ./ c_km.^2 .* ones(length(x0[:,1]))
    # suppress GR inside NS
    # rs[r .<= rNS] .= 0.0
    # rs = ones(eltype(r), size(r)) .* 2 * GNew .* Mass_NS ./ c_km.^2
    # Mass_NS is already re-adjusted in func_axion!...
    # rs[r .<= rNS] .*= (r[r .<= rNS] ./ rNS).^3

    # sin_theta = sin.(x0[:,2])

    g_tt = -1.0 ./ (1.0 .- rs ./ r);
    g_rr = (1.0 .- rs ./ r);
    g_thth = 1.0 ./ r.^2; # 1/km^2
    g_pp = 1.0 ./ (r.^2 .* sin_theta.^2); # 1/km^2
    
    # rs = 2 * GNew .* Mass_NS ./ c_km.^2 .* ones(length(x0[:,1]))
    g_tt[r .<= rNS] = -4 ./ (3 .* sqrt.(1 .- rs[r .<= rNS] / rNS) .- sqrt.(1 .- r[r .<= rNS].^2 .* rs[r .<= rNS] ./ rNS.^3) ).^2
    g_rr[r .<= rNS] = (1 .- r[r .<= rNS].^2 .* rs[r .<= rNS]./rNS.^3)

    return g_tt, g_rr, g_thth, g_pp
    
end

function Cristoffel(x0_pl, time0, θm, ωPul, B0, rNS, Mass_NS; flat=false)
    if flat
        MassNS = 0.0
    else
        MassNS = Mass_NS
    end
    r = x0_pl[1]
    theta = x0_pl[2]
    
    GM = GNew .* Mass_NS ./ c_km.^2
    G_rrr = - GM ./ (r .* (r .- 2 .* GM))
    G_rtt = - (r - 2 .* GM)
    G_rpp = - (r - 2 .* GM) .* sin.(theta).^2
    G_trt = 1.0 ./ r
    G_tpp = -sin.(theta) .* cos.(theta)
    G_prp = 1.0 ./ r
    G_ptp = cos.(theta) ./ sin.(theta)
    
    G_ttr = 1.0 ./ r
    G_ppr = 1.0 ./ r
    G_ppt = cos.(theta) ./ sin.(theta)
    
    return G_rrr, G_rtt, G_rpp, G_trt, G_tpp, G_prp, G_ptp, G_ttr, G_ppr, G_ppt
    # return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
end


function hamiltonian(x, k,  time0, erg, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=true, melrose=false, zeroIn=false, bndry_lyr=-1)
    x[x[:,1] .< rNS, 1] .= rNS;
    omP = GJ_Model_ωp_vecSPH(x, time0, θm, ωPul, B0, rNS, zeroIn=zeroIn, bndry_lyr=bndry_lyr, Mass_a=Mass_a);
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x, Mass_NS);
    ksqr = 0.0;
    try
        ksqr = g_tt .* erg.^2 .+ g_rr .* k[:, 1].^2 .+ g_thth .* k[:, 2].^2 .+ g_pp .* k[:, 3].^2
    catch
        ksqr = g_tt .* erg.^2 .+ g_rr .* k[1].^2 .+ g_thth .* k[2].^2 .+ g_pp .* k[3].^2
    end
    
    if iso
        Ham = 0.5 .* (ksqr .+ omP.^2)
        # print(time0, "\t", Ham, "\t", omP, "\n\n")
    else
        if !melrose
            ctheta = Ctheta_B_sphere(x, k, [θm, ωPul, B0, rNS, time0, Mass_NS])
            Ham = 0.5 .* (ksqr .- omP.^2 .* (1.0 .- ctheta.^2) ./ (omP.^2 .* ctheta.^2 .- erg.^2 ./ g_rr)  .* erg.^2 ./ g_rr) # original form
        else
            kpar = K_par(x, k, [θm, ωPul, B0, rNS, time0, Mass_NS, Mass_a])
            Ham = 0.5 .* (ksqr .+ omP.^2 .* (erg.^2 ./ g_rr .- kpar.^2) ./  (erg.^2 ./ g_rr)  );
        end
    end
    
    return Ham
end

function omega_function(x, k,  time0, erg, θm, ωPul, B0, rNS, Mass_NS, Mass_a; kmag=nothing, cthetaB=nothing, iso=true, melrose=false, flat=false, zeroIn=false, bndry_lyr=-1)
    # if r < rNS, need to not run...
    x[x[:,1] .< rNS, 1] .= rNS;

    omP = GJ_Model_ωp_vecSPH(x, time0, θm, ωPul, B0, rNS, zeroIn=zeroIn, bndry_lyr=bndry_lyr, Mass_a=Mass_a);
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x, Mass_NS);
    ksqr = 0.0;
    
    if isnothing(kmag)
        try
            ksqr = g_rr .* k[:, 1].^2 .+ g_thth .* k[:, 2].^2 .+ g_pp .* k[:, 3].^2
        catch
            ksqr = g_rr .* k[1].^2 .+ g_thth .* k[2].^2 .+ g_pp .* k[3].^2
        end
        
    else
        ksqr = kmag.^2
    end

    
    if iso
        Ham = (ksqr .+ omP.^2)
    else
        
        kpar = K_par(x, k, [θm, ωPul, B0, rNS, time0, Mass_NS, Mass_a])
        Ham = (ksqr .+ omP.^2 .+ sqrt.(ksqr.^2 .+ 2 .* ksqr .* omP.^2 .- 4 .* kpar.^2 .* omP.^2 .+ omP.^4)) ./ sqrt.(2)
        
    end
    
    return sqrt.(Ham)
end

function test_on_shell(x, v_loc, vIfty_mag, time0, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=true, melrose=false, printStuff=false, bndry_lyr=-1)
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
    
    omP = GJ_Model_ωp_vecSPH(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=false, bndry_lyr=bndry_lyr, Mass_a=Mass_a);
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x .* v0, dims=2) ./ rr[:, 1]
    v0_pl = [dr_dt (x[:,3] .* dr_dt .- rr .* v0[:, 3]) ./ (rr .* sin.(x0_pl[:,2])) (-x[:,2] .* v0[:, 1] .+ x[:,1] .* v0[:, 2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    # Switch to celerity in polar coordinates
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);

    NrmSq = (-erg_inf.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp )
    w0_pl .*= sqrt.(NrmSq)
    val = hamiltonian(x0_pl, w0_pl, time0, erg_inf, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=iso, melrose=melrose, zeroIn=false, bndry_lyr=bndry_lyr) ./ erg_inf.^2
    
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
 
    return 0.5 .* ksqr
    
end
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

function k_norm_Cart(x0, khat, time0, erg, θm, ωPul, B0, rNS, Mass_NS, Mass_a; melrose=false, flat=false, isotropic=false, ax_fix=false, is_photon=true, bndry_lyr=-1)
    
    
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
            omP = GJ_Model_ωp_vecSPH(x0_pl, time0, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a);
            
            if !isotropic
                kpar = K_par(x0_pl, w0_pl, [θm, ωPul, B0, rNS, time0, Mass_NS, Mass_a]; flat=flat)
            else
                kpar = 0.0
            end
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

function g_det(x0, t, θm, ωPul, B0, rNS, Mass_NS, Mass_a; flat=false, bndry_lyr=-1)
    # returns determinant of sqrt(-g)
    if flat
        return ones(length(x0[:, 1]))
    end
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0, Mass_NS; rNS=rNS);
    
    r = x0[:,1]
    dωp = grad(GJ_Model_ωp_vecSPH(seed(x0), t, θm, ωPul, B0, rNS, zeroIn=false, bndry_lyr=bndry_lyr, Mass_a=Mass_a));
    # dωt =  derivative(tI -> GJ_Model_ωp_vecSPH(x0, tI, θm, ωPul, B0, rNS, zeroIn=false, bndry_lyr=bndry_lyr, Mass_a=Mass_a), t[1]);
    
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
    Bvec, ωp = GJ_Model_scalar(x0, t0, θm, ωPul, B0, rNS, Mass_a)
    omegaC = sqrt.(sum(Bvec.^2, dims=2)) * 0.3 / 5.11e5 * (1.95e-20 * 1e18) # eV
    return omegaC
end

function cyclotronF_vec(x0, t0, θm, ωPul, B0, rNS, Mass_a)
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
function GJ_Model_vec(x, t, θm, ω, B0, rNS; bndry_lyr=-1, Mass_a=1e-5)
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
    if sum(r .>= rNS) > 0
        if bndry_lyr > 0.0
            nelec_pole = abs.((2.0 .* ω .* B0) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
            pole_val = sqrt.(4 .* π .* nelec_pole ./ 137 ./ 5.0e5);
            rmax = rNS .* (pole_val ./ Mass_a).^(2.0 ./ 3.0)
            ωp += pole_val .* (rNS ./ r).^(3.0 ./ 2.0) .* exp.(- (r .- rmax .* bndry_lyr) ./ (0.1 .* rmax))
        end
    end
    
    return [Bx By Bz], ωp
end



function surfNorm(x0, k0, Mvars; return_cos=true)
    # coming in cartesian, so change.
    
    ω, Mvars2 = Mvars
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS, Mass_a = Mvars2
    
    
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
    
    dωdr_grd = grad(GJ_Model_ωp_vecSPH(seed(x0_pl), t_start, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a))
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

function k_sphere(x0, k0, θm, ωPul, B0, rNS, time0, Mass_NS, Mass_a, flat; zeroIn=true, bndry_lyr=-1)
    if flat
        Mass_NS = 0.0;
    end
    
    # Define the Schwarzschild radius of the NS (in km)
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2

    # Switch to polar coordinates
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    
    omP = GJ_Model_ωp_vecSPH(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=zeroIn, bndry_lyr=bndry_lyr, Mass_a=Mass_a);
    
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
        
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS, Mass_a, flat, isotropic, erg, bndry_lyr = Mvars

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
    dωdr_grd = grad(GJ_Model_ωp_vecSPH(seed(x0_pl), t_start, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a))
    snorm = dωdr_grd ./ sqrt.(g_rr .* dωdr_grd[:, 1].^2  .+ g_thth .* dωdr_grd[:, 2].^2 .+ g_pp .* dωdr_grd[:, 3].^2)
    vgNorm = sqrt.(g_rr .* vg[:,1].^2 .+ g_thth .* vg[:,2].^2 .+ g_pp .* vg[:,3].^2)
    
    omP = GJ_Model_ωp_vecSPH(x0_pl, t_start, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a)
    
    ctheta = (g_rr .* vg[:,1] .* snorm[:, 1] .+ g_thth .* vg[:,2] .* snorm[:, 2] .+ g_pp .* vg[:,3] .* snorm[:, 3]) ./ vgNorm
    
    if return_cos
        return ctheta
    else
        return ctheta, snorm
    end
end

function K_par(x0_pl, k_sphere, Mvars; flat=false)
    θm, ωPul, B0, rNS, t_start, Mass_NS, Mass_a = Mvars
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
function GJ_Model_ωp_vec(x, t, θm, ω, B0, rNS; bndry_lyr=-1, Mass_a=1e-5)
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

    if sum(r .>= rNS) > 0
        if bndry_lyr > 0
            nelec_pole = abs.((2.0 .* ω .* B0) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
            pole_val = sqrt.(4 .* π .* nelec_pole ./ 137 ./ 5.0e5);
            rmax = rNS .* (pole_val ./ Mass_a).^(2.0 ./ 3.0)
            
            ωp[r .>= rNS] .+= pole_val .* (rNS ./ r[r .>= rNS]).^(3.0 ./ 2.0) .* exp.(- (r[r .>= rNS] .- rmax .* bndry_lyr) ./ (0.1 .* rmax))
          
        end
    end
    
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

function GJ_Model_ωp_vecSPH(x, t, θm, ω, B0, rNS; zeroIn=true, bndry_lyr=-1, Mass_a=1e-5)
    # For GJ model, return \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]

#    xC = spherical_to_cartesian(x)
#    for i in 1:length(x[:,1])
#        xC[i, :] .= rotate_y(θR) * xC[i, :]
#    end
#    xNew = cartesian_to_spherical(xC)
    if length(size(x)) > 1
        r = view(x, :, 1)
        θ = view(x, :, 2)
        ϕ = view(x, :, 3)
    else
        r = x[1]
        θ = x[2]
        ϕ = x[3]
    end

    
    ψ = ϕ .- ω.*t
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    
    # Bx = Br .* sin.(θ) .* cos.(ϕ) .+ Btheta .* cos.(θ) .* cos.(ϕ) .- Bphi .* sin.(ϕ)
    # By = Br .* sin.(θ) .* sin.(ϕ) .+ Btheta .* cos.(θ) .* sin.(ϕ) .+ Bphi .* cos.(ϕ)
    Bz = Br .* cos.(θ) .- Btheta .* sin.(θ)
    
    nelec = abs.((2.0 .* ω .* Bz) ./ sqrt.(4 .* π ./ 137) .* 1.95e-2 .* hbar) ; # eV^3
    ωp = sqrt.(4 .* π .* nelec ./ 137 ./ 5.0e5);
    if sum(r .>= rNS) > 0
        if bndry_lyr > 0
            nelec_pole = abs.((2.0 .* ω .* B0) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
            pole_val = sqrt.(4 .* π .* nelec_pole ./ 137 ./ 5.0e5);
            rmax = rNS .* (pole_val ./ Mass_a).^(2.0 ./ 3.0)
            
            ωp[r .>= rNS] .+= pole_val .* (rNS ./ r[r .>= rNS]).^(3.0 ./ 2.0) .* exp.(- (r[r .>= rNS] .- rmax .* bndry_lyr) ./ (0.1 .* rmax))
        end
    end

    if zeroIn
        ωp[r .<= rNS] .= 0.0;
    end

    return ωp
end

function GJ_Model_ωp_scalar(x, t, θm, ω, B0, rNS; bndry_lyr=-1, Mass_a=1e-5)
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
    
    if bndry_lyr > 0
        nelec_pole = abs.((2.0 .* ω .* B0) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
        pole_val = sqrt.(4 .* π .* nelec_pole ./ 137 ./ 5.0e5);
        if r .>= rNS
            rmax = rNS .* (pole_val ./ Mass_a).^(2.0 ./ 3.0)
            
            ωp += pole_val .* (rNS ./ r).^(3.0 ./ 2.0) .* exp.(- (r .- rmax .* bndry_lyr) ./ (0.1 .* rmax))
            
        end

    
    end

    return ωp
end

function GJ_Model_scalar(x, t, θm, ω, B0, rNS; bndry_lyr=-1, Mass_a=1e-5)
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
    
    if bndry_lyr > 0.0
        nelec_pole = abs.((2.0 .* ω .* B0) ./ sqrt.(4 .* π ./ 137) .* (1.95e-2) .* hbar) ; # eV^3
        pole_val = sqrt.(4 .* π .* nelec_pole ./ 137 ./ 5.0e5);
        if r .>= rNS
            rmax = rNS .* (pole_val ./ Mass_a).^(2.0 ./ 3.0)
            ωp .+= pole_val .* (rNS ./ r).^(3.0 ./ 2.0) .* exp.(- (r .- rmax .* bndry_lyr) ./ (0.1 .* rmax))
        
        end
        
    end

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




function GJ_Model_Sphereical(x, t, θm, ω, B0, rNS; Mass_NS=1.0, flat=false, sphericalX=false, return_comp=-1)
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
    if return_comp == -1
        return [Br ./ sqrt.(g_rr) Btheta ./ sqrt.(g_thth)  Bphi ./ sqrt.(g_pp)] # lower
    elseif return_comp == 0
        return sqrt.(Br.^2 .+ Btheta.^2 .+ Bphi.^2) * 1.95e-2
    elseif return_comp == 1
        return Br ./ sqrt.(g_rr) .* g_rr * 1.95e-2 # this is for d_mu B^i
    elseif return_comp == 2
        return Btheta ./ sqrt.(g_thth) .* g_thth * 1.95e-2
    elseif return_comp == 3
        return Bphi ./ sqrt.(g_pp) .* g_pp * 1.95e-2
    end

end

function k_gamma(x0_pl, ksphere, time0, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a; melrose=false, flat=false, isotropic=false, bndry_lyr=false)
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    ntrajs = length(erg_inf_ini)
    Bsphere = GJ_Model_Sphereical(x0_pl, time0, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat)
    omP = GJ_Model_ωp_vecSPH(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=true, bndry_lyr=bndry_lyr, Mass_a=Mass_a)
    kmag = sqrt.(g_rr .* ksphere[:, 1].^2  .+ g_thth .* ksphere[:, 2].^2 .+ g_pp .* ksphere[:, 3].^2)
    Bmag = sqrt.(g_rr .* Bsphere[:, 1].^2  .+ g_thth .* Bsphere[:, 2].^2 .+ g_pp .* Bsphere[:, 3].^2)
    ctheta_B = (g_rr .* Bsphere[:, 1] .* ksphere[:, 1]  .+ g_thth .* Bsphere[:, 2] .* ksphere[:, 2] .+ g_pp .* Bsphere[:, 3] .* ksphere[:, 3])./ (kmag .* Bmag)
    if isotropic
        ctheta_B .*= 0.0
    end
    erg_loc = erg_inf_ini ./ g_rr
    return erg_loc .* sqrt.(erg_loc.^2 .- omP.^2) ./  sqrt.(erg_loc.^2 .- omP.^2 .* ctheta_B.^2)
    
end

function dwp_ds(xIn, ksphere, Mvars)
    # xIn cartesian, ksphere [spherical]
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS, Mass_a, flat, isotropic, ωErg, bndry_lyr = Mvars
    
    rr = sqrt.(sum(xIn.^2, dims=2))
    x0_pl = [rr acos.(xIn[:,3] ./ rr) atan.(xIn[:,2], xIn[:,1])]
    
    ntrajs = length(t_start)
    # general info we need for all
    omP = GJ_Model_ωp_vecSPH(x0_pl, t_start, θm, ωPul, B0, rNS, zeroIn=true, bndry_lyr=bndry_lyr, Mass_a=Mass_a)
    erg_inf_ini = sqrt.(1.0 .- 2.0 .* GNew .* Mass_NS ./ rr ./ c_km.^2) .* ωErg
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    Bsphere = GJ_Model_Sphereical(xIn, t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat)
    kmag = sqrt.(spatial_dot(ksphere, ksphere, ntrajs, x0_pl, Mass_NS));
    kB_norm = spatial_dot(Bsphere, ksphere ./ kmag, ntrajs, x0_pl, Mass_NS)
    v_ortho = -(Bsphere .- kB_norm .* ksphere ./ kmag)
    v_ortho ./= sqrt.(spatial_dot(v_ortho, v_ortho, ntrajs, x0_pl, Mass_NS))
    Bmag = sqrt.(spatial_dot(Bsphere, Bsphere, ntrajs, x0_pl, Mass_NS));
    ctheta_B = spatial_dot(Bsphere , ksphere, ntrajs, x0_pl, Mass_NS) ./ (kmag .* Bmag)
    stheta_B = sin.(acos.(ctheta_B))
    if isotropic
        ctheta_B .*= 0.0
        stheta_B ./= stheta_B
    end
    xi = stheta_B .^2 ./ (1.0 .- ctheta_B.^2 .* omP.^2 ./ ωErg.^2)
    
    # omega_p computation
    grad_omP = grad(GJ_Model_ωp_vecSPH(seed(x0_pl), t_start, θm, ωPul, B0, rNS, zeroIn=true, bndry_lyr=bndry_lyr, Mass_a=Mass_a));
    grad_omP_norm = grad_omP ./ sqrt.(g_rr .* grad_omP[:, 1].^2  .+ g_thth .* grad_omP[:, 2].^2 .+ g_pp .* grad_omP[:, 3].^2)
    dz_omP = spatial_dot(ksphere ./ kmag, grad_omP, ntrajs, x0_pl, Mass_NS)
    dy_omP = spatial_dot(v_ortho, grad_omP, ntrajs, x0_pl, Mass_NS)
    w_prime = dz_omP .+ omP.^2 ./ ωErg.^2 .* xi ./ (stheta_B ./ ctheta_B) .* dy_omP
    
    # k gamma computation
    grad_kgamma = grad(k_gamma(seed(x0_pl), ksphere, t_start, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a; bndry_lyr=bndry_lyr, melrose=true, flat=flat, isotropic=isotropic))
    gradK_norm = grad_kgamma ./ sqrt.(g_rr .* grad_kgamma[:, 1].^2  .+ g_thth .* grad_kgamma[:, 2].^2 .+ g_pp .* grad_kgamma[:, 3].^2)
    dz_k = spatial_dot(ksphere ./ kmag, grad_kgamma, ntrajs, x0_pl, Mass_NS)
    dy_k = spatial_dot(v_ortho, grad_kgamma, ntrajs, x0_pl, Mass_NS)
    k_prime = dz_k .+ omP.^2 ./ ωErg.^2 .* xi ./ (stheta_B ./ ctheta_B) .* dy_k
    cos_k = abs.(spatial_dot(ksphere ./ kmag, gradK_norm, ntrajs, x0_pl, Mass_NS))
    
    
    # energy computation
    grad_omega = grad(omega_function(seed(x0_pl), ksphere, t_start, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a, iso=isotropic, melrose=true)) # based on energy gradient
    gradE_norm = grad_omega ./ sqrt.(g_rr .* grad_omega[:, 1].^2  .+ g_thth .* grad_omega[:, 2].^2 .+ g_pp .* grad_omega[:, 3].^2)
    dz_w = spatial_dot(ksphere ./ kmag, grad_omega, ntrajs, x0_pl, Mass_NS)
    dy_w = spatial_dot(v_ortho, grad_omega, ntrajs, x0_pl, Mass_NS)
    erg_prime = dz_w .+ omP.^2 ./ ωErg.^2 .* xi ./ (stheta_B ./ ctheta_B) .* dy_w
    cos_w = abs.(spatial_dot(ksphere ./ kmag, gradE_norm, ntrajs, x0_pl, Mass_NS))
   
    
    # group velocity
    v_group = grad(omega_function(x0_pl, seed(ksphere), t_start, -erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a, iso=isotropic, melrose=true))
    
    v_group[:, 1] ./= g_rr
    v_group[:, 2] ./= g_thth
    v_group[:, 3] ./= g_pp
    vgNorm = sqrt.(spatial_dot(v_group, v_group, ntrajs, x0_pl, Mass_NS));
    
    
    slength = sqrt.(1.0 .+ (omP.^2 ./ ωErg.^2 .* stheta_B.^2 ./ (1.0 .- omP.^2 ./ ωErg.^2 .* ctheta_B.^2) .* (ctheta_B ./stheta_B)  ).^2)
    if isotropic
        slength ./= slength
    end
    newGuess = (slength ./ vgNorm) .* spatial_dot(ksphere ./ kmag, grad_omega, ntrajs, x0_pl, Mass_NS)
    
    omp_vg = abs.(spatial_dot(v_group ./ vgNorm, grad_omP_norm, ntrajs, x0_pl, Mass_NS))
    dk_vg = abs.(spatial_dot(v_group ./ vgNorm, gradK_norm, ntrajs, x0_pl, Mass_NS)) # this is group velocity photon to surface normal
    k_vg = abs.(spatial_dot(v_group ./ vgNorm, ksphere ./ kmag, ntrajs, x0_pl, Mass_NS))
    dE_vg = abs.(spatial_dot(v_group ./ vgNorm, gradE_norm, ntrajs, x0_pl, Mass_NS)) # this is group velocity photon to surface normal
    
    
    
    ###
    return abs.(w_prime), abs.(k_prime), abs.(newGuess), cos_w, vgNorm, dk_vg, dE_vg, k_vg
   
end

function conversion_prob(Ax_g, x0_pl, Mvars, local_vars; one_D=false)
    
    # xIn cartesian, ksphere [spherical]
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS, Mass_a, flat, isotropic, ωErg, bndry_lyr = Mvars
    ωp, Bsphere, Bmag, ksphere, kmag, vgNorm, ctheta_B, stheta_B = local_vars
    vloc = sqrt.(ωErg.^2 .- Mass_a.^2) ./ ωErg
    

    
    ntrajs = length(t_start)
    rr = x0_pl[:, 1]
    
    erg_inf_ini = sqrt.(1.0 .- 2.0 .* GNew .* Mass_NS ./ rr ./ c_km.^2) .* ωErg
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    
    if isotropic
        dmu_E = grad(omega_function(seed(x0_pl), ksphere, t_start, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a, kmag=kmag, cthetaB=ctheta_B, iso=isotropic, flat=flat, melrose=true)) # based on energy gradient
        dmu_E_2 = dmu_E
    else
        G_rrr, G_rtt, G_rpp, G_trt, G_tpp, G_prp, G_ptp, G_ttr, G_ppr, G_ppt = Cristoffel(x0_pl, t_start, θm, ωPul, B0, rNS, Mass_NS; flat=flat)
        rs = 2 * GNew .* Mass_NS ./ c_km.^2
        
        dmu_omP = grad(GJ_Model_ωp_vecSPH(seed(x0_pl), t_start, θm, ωPul, B0, rNS, zeroIn=true, bndry_lyr=bndry_lyr, Mass_a=Mass_a))

        dmu_B = grad(GJ_Model_Sphereical(seed(x0_pl), t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true, return_comp=0))
        # dmu_B = grad(func_BField(seed(x0_pl), t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true, return_comp=0))
  
        term1 = ksphere[1] .* grad(GJ_Model_Sphereical(seed(x0_pl), t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true, return_comp=1)) .+ ksphere[2] .* grad(GJ_Model_Sphereical(seed(x0_pl), t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true, return_comp=2)) .+ ksphere[3] .* grad(GJ_Model_Sphereical(seed(x0_pl), t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true, return_comp=3))
        
        term2_r = ksphere[1] .* (g_rr .* Bsphere[1] .* 1.95e-2) .* G_rrr .+ ksphere[2] .* G_trt .* (Bsphere[2] .* g_thth .* 1.95e-2) .+ ksphere[3] .* G_prp .* (Bsphere[3] .* g_pp .* 1.95e-2)
        term2_t = ksphere[1] .* (g_thth .* Bsphere[2] .* 1.95e-2) .* G_rtt .+ ksphere[3] .* G_ptp .* (Bsphere[3] .* g_pp .* 1.95e-2) .+ ksphere[2] .* (g_rr .* Bsphere[1] .* 1.95e-2) .* G_ttr
        term2_p =  ksphere[1] .* (g_pp .* Bsphere[3] .* 1.95e-2) .* G_rpp .+ ksphere[2] .* G_tpp .* (Bsphere[3] .* g_pp .* 1.95e-2) .+ ksphere[3] .* G_ppr .* (Bsphere[1] .* g_rr .* 1.95e-2) .+ ksphere[3] .* G_ppt .* (Bsphere[2] .* g_thth .* 1.95e-2)
        
        dmu_ctheta = (term1 .+ [term2_r term2_t term2_p]) ./ (kmag .* Bmag) .- ctheta_B .* dmu_B ./ Bmag
        
        v_group = grad(omega_function(x0_pl, seed(ksphere), t_start, -erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a, flat=flat, iso=isotropic, melrose=true)) #
        term2_r = G_rrr .* ksphere[1] .* (g_rr .* v_group[1]) .+ G_trt .* ksphere[2] .* (g_thth .* v_group[2]) .+ G_prp .* ksphere[3] .* (g_pp .* v_group[3])
        term2_t = G_rtt .* ksphere[1] .* (g_thth .* v_group[2]) .+ G_ptp .* ksphere[3] .* (g_pp .* v_group[3]) .+ G_ttr .* ksphere[2] .* (g_rr .* v_group[1])
        term2_p = G_rpp .* ksphere[1] .* (g_pp .* v_group[3]) .+ G_tpp .* ksphere[2] .* (g_pp .* v_group[3]) .+ G_ppr .* ksphere[3] .* (g_rr .* v_group[1]) .+ G_ppt .* ksphere[3] .* (g_thth .* v_group[2])
        
        term2 = [term2_r term2_t term2_p]
       
        
        preF = ωp ./ abs.(ωErg.^5 .+ ctheta_B.^2 .* ωErg .* (ωp.^4 .- 2 .* ωp.^2 .* ωErg.^2))
        dmu_E = preF .* (ωErg.^4 .* stheta_B.^2 .* dmu_omP .- ωErg.^2 .* ctheta_B .* ωp .* (ωErg.^2 .- ωp.^2) .* dmu_ctheta)
        
        dmu_E_2 = dmu_E .+ term2
        vhat_gradE = spatial_dot(-2.0 .* ksphere ./ kmag, dmu_E, ntrajs, x0_pl, Mass_NS)
        # vhat_gradE_2 = spatial_dot(-2.0 .* ksphere ./ kmag, dmu_E_2, ntrajs, x0_pl, Mass_NS)
    end
    
    gradE_norm = dmu_E ./ sqrt.(g_rr .* dmu_E[:, 1].^2  .+ g_thth .* dmu_E[:, 2].^2 .+ g_pp .* dmu_E[:, 3].^2)
    gradE_norm_2 = dmu_E_2 ./ sqrt.(g_rr .* dmu_E_2[:, 1].^2  .+ g_thth .* dmu_E_2[:, 2].^2 .+ g_pp .* dmu_E_2[:, 3].^2)
    cos_w = abs.(spatial_dot(ksphere ./ kmag, gradE_norm, ntrajs, x0_pl, Mass_NS))
    cos_w_2 = abs.(spatial_dot(ksphere ./ kmag, gradE_norm_2, ntrajs, x0_pl, Mass_NS))
    vhat_gradE = spatial_dot(ksphere ./ kmag, dmu_E, ntrajs, x0_pl, Mass_NS)
    grad_Emag = spatial_dot(dmu_E, dmu_E, ntrajs, x0_pl, Mass_NS)
    grad_Emag_2 = spatial_dot(dmu_E_2, dmu_E_2, ntrajs, x0_pl, Mass_NS)
    
    if one_D
        Prob = π ./ 2.0 .* (Ax_g .* 1e-9 .* Bmag).^2  ./ (vloc .* (abs.(vhat_gradE) .* c_km .* hbar)) #
    else
        prefactor = ωErg.^4 .* stheta_B.^2 ./ (ctheta_B.^2 .* ωp.^2 .* (ωp.^2 .- 2 .* ωErg .^2) .+ ωErg.^4)
        Prob = π ./ 2.0 .* prefactor .* (Ax_g .* 1e-9 .* Bmag).^2 ./ (abs.(vhat_gradE) .* vloc .* c_km .* hbar)  # jamies paper
        # Prob = π ./ 2.0 .* prefactor .* (Ax_g .* 1e-9 .* Bmag).^2 ./ (abs.(vhat_gradE_2) .* vloc .* c_km .* hbar)  # jamies paper
    end
    # print(x0_pl, "\t", ksphere, "\t", vhat_gradE, "\t", Prob, "\n")
    return Prob, abs.(vhat_gradE), abs.(cos_w), sqrt.(grad_Emag), abs.(cos_w_2), sqrt.(grad_Emag_2)
end






function find_samples_new(maxR, θm, ωPul, B0, rNS, Mass_a, Mass_NS; n_max=6, batchsize=2, thick_surface=false, iso=false, melrose=false, pre_randomized=nothing, t0=0.0, flat=false, rand_cut=true, bndry_lyr=-1)
    
    if isnothing(pre_randomized)
        # ~~~collecting random samples
        
        # randomly sample angles θ, ϕ, hit conv surf
        θi = acos.(1.0 .- 2.0 .* rand());
        ϕi = rand() .* 2π;
        
        # local velocity
        θi_loc = acos.(1.0 .- 2.0 .* rand());
        ϕi_loc = rand() .* 2π;
        
        # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
        ϕRND = rand() .* 2π;
        
        # radius on disk
        rRND = sqrt.(rand()) .* maxR; # standard flat sampling
        # rRND = rand() .* maxR; # 1/r sampling
        
        # ~~~ Done collecting random samples
    else

        θi = acos.(1.0 .- 2.0 .* pre_randomized[1]);
        ϕi = pre_randomized[2] .* 2π;
        
        # local velocity
        θi_loc = acos.(1.0 .- 2.0 .* pre_randomized[3]);
        ϕi_loc = pre_randomized[4] .* 2π;
        
        # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
        ϕRND = pre_randomized[5] .* 2π;
        
        # radius on disk
        # rRND = sqrt.(pre_randomized[6]) .* maxR; # standard flat sampling
        rRND = pre_randomized[6] .* maxR; # 1/r sampling
        
        
    end
    
    # disk direction
    vvec_all = [sin.(θi) .* cos.(ϕi) sin.(θi) .* sin.(ϕi) cos.(θi)];
    
    # vel direction
    vvec_loc = [sin.(θi_loc) .* cos.(ϕi_loc) sin.(θi_loc) .* sin.(ϕi_loc) cos.(θi_loc)];
    
    x1 = rRND .* cos.(ϕRND);
    x2 = rRND .* sin.(ϕRND);
    # rotate using Inv[EurlerMatrix(ϕi, θi, 0)] on vector (x1, x2, 0)
    x0_all= [x1 .* cos.(-ϕi) .* cos.(-θi) .+ x2 .* sin.(-ϕi) x2 .* cos.(-ϕi) .- x1 .* sin.(-ϕi) .* cos.(-θi) x1 .* sin.(-θi)];
    
    vIfty = (220.0 .+ rand(batchsize, 3) .* 1.0e-5) ./ sqrt.(3);
    # vIfty = 220.0 .* erfinv.( 2.0 .* rand(batchsize, 3) .- 1.0);
    # vIfty = erfinv.(2 .* rand(batchsize, 3) .- 1.0) .* vmean_ax .+ v_NS # km /s
    
    vIfty_mag = sqrt.(sum(vIfty.^2, dims=2));
    
    gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ./ c_km).^2 )
    erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag ./ c_km .* gammaA).^2)
    # Mass_NS = 1.0
    
    # print(x0_all, "\t", vvec_all, "\n")
    x0_all .+= vvec_all .* (-maxR .* 1.1)
    
    xc = []; yc = []; zc = []
    
    
    function condition(u, t, integrator)
        if !thick_surface
            # print(u, "\t", func_Plasma(u, t0, θm, ωPul, B0, rNS; sphericalX=false, zeroIn=false), "\n")
            return (log.(GJ_Model_ωp_vec(u, t0, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a)) .- log.(Mass_a))[1]
            # return (func_Plasma(u, t0, θm, ωPul, B0, rNS; sphericalX=false) .- Mass_a)[1]
        else
            
            r_s0 = 2.0 * Mass_NS * GNew / c_km^2
            rr = sqrt.(sum(u.^2))
            x0_pl = [rr acos.(u[3] ./ rr) atan.(u[2], u[1])]
            AA = (1.0 .- r_s0 ./ rr)
            if rr .< rNS
                AA = 1.0
            end
            
  
            dr_dt = sum(u .* vvec_loc) ./ rr
            v0_pl = [dr_dt (u[3] .* dr_dt .- rr .* vvec_loc[3]) ./ (rr .* sin.(x0_pl[2])) (-u[2] .* vvec_loc[1] .+ u[1] .* vvec_loc[2]) ./ (rr .* sin.(x0_pl[2])) ];
            # Switch to celerity in polar coordinates
            w0_pl = [v0_pl[1] ./ sqrt.(AA)   v0_pl[2] ./ rr .* rr.^2  v0_pl[3] ./ (rr .* sin.(x0_pl[2])) .* (rr .* sin.(x0_pl[2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
            g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);

            NrmSq = (-erg_inf_ini.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[1].^2 .* g_rr .+  w0_pl[2].^2 .* g_thth .+ w0_pl[3].^2 .* g_pp )
            w0_pl .*= sqrt.(NrmSq)
            
            omP = GJ_Model_ωp_vec(u, t0, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a)
            if iso
                kpar = 0.0
            else
                kpar = K_par(x0_pl, w0_pl, [θm, ωPul, B0, rNS, t0, Mass_NS, Mass_a])
            end
            ksqr = g_tt .* erg_inf_ini.^2 .+ g_rr .* w0_pl[1].^2 .+ g_thth .* w0_pl[2].^2 .+ g_pp .* w0_pl[3].^2
            Ham = 0.5 .* (ksqr .+ omP.^2 .* (erg_inf_ini.^2 ./ g_rr .- kpar.^2) ./  (erg_inf_ini.^2 ./ g_rr)  ) ./ (erg_inf_ini.^2);
            
            return Ham[1]
        end
    end
    
    function affect!(int)
        rr = sqrt.(sum(int.u.^2))
        x0_pl = [rr acos.(int.u[3] ./ rr) atan.(int.u[2], int.u[1])]
        omP = GJ_Model_ωp_vec(int.u, t0, θm, ωPul, B0, rNS, bndry_lyr=bndry_lyr, Mass_a=Mass_a)[1]
        g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
        ergL = erg_inf_ini ./ sqrt.(g_rr)
        
        if (rr .> rNS) && (ergL[1] .> omP)
            push!( xc, int.u[1] )
            push!( yc, int.u[2]  )
            push!( zc, int.u[3]  )
        end
    end
    
    cb_s = ContinuousCallback(condition, affect!, interp_points=20, abstol=1e-6)
    
    function func_line!(du, u, Mvars, t)
        @inbounds begin
            kk, v_loc = Mvars
            du .= kk
        end
    end

    
    # solve differential equation with callback
    Mvars = [vvec_all, vvec_loc]
    prob = ODEProblem(func_line!, x0_all, (0, 2.2*maxR), Mvars, callback=cb_s)
    
    sol = solve(prob, Euler(), abstol=1e-4, reltol=1e-3, dt=0.5)

    
    
    
    if length(xc) == 0
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0
    end
    
    if rand_cut
        randInx = rand(1:n_max)
        weights = length(xc)
        indx = 0
        
        
        if weights .>= randInx
            indx = randInx
        else
            return 0.0, 0.0, 0, 0.0, 0.0, 0.0
        end
        
        xpos_flat = [xc[indx] yc[indx] zc[indx]]
        rmag = sqrt.(sum(xpos_flat.^2))

            
        vmag_loc = sqrt.(vIfty_mag.^2 .+ 2 .* GNew .* Mass_NS ./ rmag) ./ c_km
        
        return xpos_flat, rRND, 1, weights, vvec_loc .* vmag_loc[1], vIfty ./ c_km
    else
        xpos_flat = [xc yc zc]
        weights = ones(length(xc))
        rmag = sqrt.(sum(xpos_flat.^2))
        num_c = length(xc)
        
        vmag_loc = sqrt.(vIfty_mag.^2 .+ 2 .* GNew .* Mass_NS ./ rmag) ./ c_km
        
        return xpos_flat, rRND .* ones(num_c), num_c, weights, vvec_loc .* vmag_loc, vIfty ./ c_km .* ones(num_c)
        
    end
    
end


function find_samples(maxR, ntimes_ax, θm, ωPul, B0, rNS, Mass_a, Mass_NS; n_max=8, batchsize=2, thick_surface=false, iso=false, melrose=false, pre_randomized=nothing, t0=0.0, flat=false, bndry_lyr=-1)
    
    if isnothing(pre_randomized)
        # ~~~collecting random samples
        
        # randomly sample angles θ, ϕ, hit conv surf
        θi = acos.(1.0 .- 2.0 .* rand());
        ϕi = rand() .* 2π;
        
        # local velocity
        θi_loc = acos.(1.0 .- 2.0 .* rand());
        ϕi_loc = rand() .* 2π;
        
        # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
        ϕRND = rand() .* 2π;
        
        # radius on disk
        # rRND = sqrt.(rand()) .* maxR; # standard flat sampling
        rRND = rand() .* maxR; # 1/r sampling
        
        # ~~~ Done collecting random samples
    else

        θi = acos.(1.0 .- 2.0 .* pre_randomized[1]);
        ϕi = pre_randomized[2] .* 2π;
        
        # local velocity
        θi_loc = acos.(1.0 .- 2.0 .* pre_randomized[3]);
        ϕi_loc = pre_randomized[4] .* 2π;
        
        # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
        ϕRND = pre_randomized[5] .* 2π;
        
        # radius on disk
        # rRND = sqrt.(pre_randomized[6]) .* maxR; # standard flat sampling
        rRND = rand() .* pre_randomized[6]; # 1/r sampling
    end
    
    # disk direction
    vvec_all = [sin.(θi) .* cos.(ϕi) sin.(θi) .* sin.(ϕi) cos.(θi)];
    
    # vel direction
    vvec_loc = [sin.(θi_loc) .* cos.(ϕi_loc) sin.(θi_loc) .* sin.(ϕi_loc) cos.(θi_loc)];
    
    x1 = rRND .* cos.(ϕRND);
    x2 = rRND .* sin.(ϕRND);
    # rotate using Inv[EurlerMatrix(ϕi, θi, 0)] on vector (x1, x2, 0)
    x0_all= [x1 .* cos.(-ϕi) .* cos.(-θi) .+ x2 .* sin.(-ϕi) x2 .* cos.(-ϕi) .- x1 .* sin.(-ϕi) .* cos.(-θi) x1 .* sin.(-θi)];
    
    vIfty = (220.0 .+ rand(batchsize, 3) .* 1.0e-5) ./ sqrt.(3);
    # vIfty = 220.0 .* erfinv.( 2.0 .* rand(batchsize, 3) .- 1.0);
    
    vIfty_mag = sqrt.(sum(vIfty.^2, dims=2));
    
    gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ./ c_km).^2 )
    erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag ./ c_km .* gammaA).^2)
    Mass_NS = 1.0
    
    x0_all .+= vvec_all .* (-maxR)
    
    xc = []; yc = []; zc = []
    
    # print("\n\n")
    function condition(u, t, integrator)
        if !thick_surface
            return (log.(GJ_Model_ωp_vec(u, t0, θm, ωPul, B0, rNS; bndry_lyr=bndry_lyr, Mass_a=Mass_a)) .- log.(Mass_a))[1]
        else
            
            r_s0 = 2.0 * Mass_NS * GNew / c_km^2
            rr = sqrt.(sum(u.^2))
            x0_pl = [rr acos.(u[3] ./ rr) atan.(u[2], u[1])]
            AA = (1.0 .- r_s0 ./ rr)
            if rr .< rNS
                AA = 1.0
            end
            
  
            dr_dt = sum(u .* vvec_loc) ./ rr
            v0_pl = [dr_dt (u[3] .* dr_dt .- rr .* vvec_loc[3]) ./ (rr .* sin.(x0_pl[2])) (-u[2] .* vvec_loc[1] .+ u[1] .* vvec_loc[2]) ./ (rr .* sin.(x0_pl[2])) ];
            # Switch to celerity in polar coordinates
            w0_pl = [v0_pl[1] ./ sqrt.(AA)   v0_pl[2] ./ rr .* rr.^2  v0_pl[3] ./ (rr .* sin.(x0_pl[2])) .* (rr .* sin.(x0_pl[2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
            g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);

            NrmSq = (-erg_inf_ini.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[1].^2 .* g_rr .+  w0_pl[2].^2 .* g_thth .+ w0_pl[3].^2 .* g_pp )
            w0_pl .*= sqrt.(NrmSq)
            
            omP = GJ_Model_ωp_vec(u, t0, θm, ωPul, B0, rNS; bndry_lyr=bndry_lyr, Mass_a=Mass_a)
            kpar = K_par(x0_pl, w0_pl, [θm, ωPul, B0, rNS, t0, Mass_NS, Mass_a])
            ksqr = g_tt .* erg_inf_ini.^2 .+ g_rr .* w0_pl[1].^2 .+ g_thth .* w0_pl[2].^2 .+ g_pp .* w0_pl[3].^2
            Ham = 0.5 .* (ksqr .+ omP.^2 .* (erg_inf_ini.^2 ./ g_rr .- kpar.^2) ./  (erg_inf_ini.^2 ./ g_rr)  ) ./ (erg_inf_ini.^2);
            
            return Ham[1]
        end
    end
    
    function affect!(int)
        rr = sqrt.(sum(int.u.^2))
        x0_pl = [rr acos.(int.u[3] ./ rr) atan.(int.u[2], int.u[1])]
        omP = GJ_Model_ωp_vec(int.u, t0, θm, ωPul, B0, rNS; bndry_lyr=bndry_lyr, Mass_a=Mass_a)[1]
        g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
        ergL = erg_inf_ini ./ sqrt.(g_rr)
        
        if (rr .> rNS) && (ergL[1] .> omP)
            push!( xc, int.u[1] )
            push!( yc, int.u[2]  )
            push!( zc, int.u[3]  )
        end
    end
    
    cb_s = ContinuousCallback(condition, affect!, interp_points=20, abstol=1e-4)
    
    function func_line!(du, u, Mvars, t)
        @inbounds begin
            kk, v_loc = Mvars
            du .= kk
        end
    end

    
    # solve differential equation with callback
    Mvars = [vvec_all, vvec_loc]
    
    prob = ODEProblem(func_line!, x0_all, (0.0, 2.0*maxR), Mvars, callback=cb_s)
    sol = solve(prob, Tsit5(), abstol=1e-3, reltol=5e-2)

    
    randInx = rand(1:n_max)
    weights = length(xc)
    indx = 0
    
    if weights .>= randInx
        indx = randInx
    else
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0
    end
    
    xpos_flat = [xc[indx] yc[indx] zc[indx]]
    rmag = sqrt.(sum(xpos_flat.^2))

        
    vmag_loc = sqrt.(vIfty_mag.^2 .+ 2 .* GNew .* Mass_NS ./ rmag) ./ c_km
    return xpos_flat, rRND, 1, weights, vvec_loc .* vmag_loc, vIfty ./ c_km
    
end


end


function dist_diff(xfin)
    b = zeros(size(xfin[:,1,:]))
    b[:, 1:end-1] = abs.((sqrt.(sum(xfin[:, :, 2:end] .^ 2, dims=2)) .- sqrt.(sum(xfin[:, :, 1:end-1] .^ 2, dims=2)) )) ./ c_km ./ hbar # 1 / eV
    b[end] = b[end-2]
    return b
end

