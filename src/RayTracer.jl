__precompile__()


include("Constants.jl")
import .Constants: c_km, hbar, GNew

module RayTracerGR
import ..Constants: c_km, hbar, GNew

using ForwardDiff: gradient, derivative, Dual, Partials, hessian
using OrdinaryDiffEq
using LSODA
# using CuArrays

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
function get_crossings(A)
    # Matrix with 2 for upward and -2 for downward crossings
    sign_A = sign.(A)

    cross = sign_A[2:end] - sign_A[1:end-1]

    # Index just before crossing
    i1 = Array(findall(x -> x .!= 0., cross))

    # Index just behind crossing
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
        θm, ωPul, B0, rNS, gammaF, time0, Mass_NS, erg, flat, isotropic, melrose = Mvars2;
        if flat
            Mass_NS = 0.0;
        end
        time = time0 .+  t;
        
        g_tt, g_rr, g_thth, g_pp = g_schwartz(view(u, :, 1:3), Mass_NS);
        
        du[:, 4:6] .= -grad(hamiltonian(seed(view(u, :, 1:3)), view(u, :, 4:6) .* erg , time[1], erg, θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=melrose)) .* c_km .* t .* (g_rr ./ erg) ./ erg;
        du[:, 1:3] .= grad(hamiltonian(view(u, :, 1:3), seed(view(u, :, 4:6)  .* erg ), time[1], erg, θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=melrose)) .* c_km .* t .* (g_rr ./ erg);
        du[u[:,1] .<= rNS, :] .= 0.0;
        
        du[:,7 ] .= derivative(tI -> hamiltonian(view(u, :, 1:3), view(u, :, 4:6)  .* erg , tI, erg, θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=melrose), time[1])[:] .* c_km .* t .* (g_rr[:] ./ erg[:]);
        
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
        
        g_tt, g_rr, g_thth, g_pp = g_schwartz(view(u, :, 1:3), Mass_NS);
        
        du[:, 4:6] .= -grad(hamiltonian_axion(seed(view(u, :, 1:3)), view(u, :, 4:6) .* erg , time[1], erg, θm, ωPul, B0, rNS, Mass_NS, mass_axion, iso=isotropic, melrose=melrose)) .* c_km .* t .* (g_rr ./ erg) ./ erg;
        du[:, 1:3] .= grad(hamiltonian_axion(view(u, :, 1:3), seed(view(u, :, 4:6)  .* erg ), time[1], erg, θm, ωPul, B0, rNS, Mass_NS, mass_axion, iso=isotropic, melrose=melrose)) .* c_km .* t .* (g_rr ./ erg);
        du[u[:,1] .<= rNS, :] .= 0.0;
        
        du[:,7 ] .= derivative(tI -> hamiltonian_axion(view(u, :, 1:3), view(u, :, 4:6)  .* erg , tI, erg, θm, ωPul, B0, rNS, Mass_NS, mass_axion, iso=isotropic, melrose=melrose), time[1])[:] .* c_km .* t .* (g_rr[:] ./ erg[:]);
        
    end
end

# Struct for conversion points in trajectory
mutable struct node
  x    # Conversion position
  y
  z
  kx
  ky
  kz
  species # Axion or photon?
  #prob    # Conversion probability
  weight
  parent_weight
end
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# propogate photon module
function propagate(ω, x0::Matrix, k0::Matrix,  nsteps, Mvars, NumerP, rhs=func!)
    ln_tstart, ln_tend, ode_err = NumerP
    
    tspan = (ln_tstart, ln_tend)
    saveat = (tspan[2] .- tspan[1]) ./ (nsteps-1)
    
    θm, ωPul, B0, rNS, gammaF, time0, Mass_NS, erg, flat, isotropic, melrose = Mvars;
    
    if flat
        Mass_NS = 0.0;
    end
    
    
    # Define the Schwarzschild radius of the NS (in km)
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2

    # Switch to polar coordinates
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* k0, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* k0[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* k0[:,1] .+ x0[:,1] .* k0[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    
    # Switch to celerity in polar coordinates
    AA = (1.0 .- r_s0 ./ rr)
    
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    
    w0_pl ./= erg
    
    # Define initial conditions so that u0[1] returns a list of x positions (again, 1 entry for each axion trajectory) etc.
    
    
    u0 = ([x0_pl w0_pl zeros(length(rr))])
    # u0 = ([x0_pl w0_pl])
   

    function floor_aff!(int)
    
        r_s0 = 2.0 * Mass_NS * GNew / c_km^2
        AA = sqrt.(1.0 .- r_s0 ./ int.u[:, 1])
        
        test = (erg ./ AA .- GJ_Model_ωp_vecSPH(int.u[:, 1:3], exp.(int.t), θm, ωPul, B0, rNS) ) ./ erg # if negative we have problem
        fail_indx = [if test[i] .< 1e-3 i else -1 end for i in 1:length(int.u[:,1])]; # define when to bounce
        fail_indx = fail_indx[ fail_indx .> 0];
       
        if int.dt > 1e-10
            set_proposed_dt!(int,(int.t-int.tprev)/100)
        end
        
        if length(fail_indx) .> 0
            g_tt, g_rr, g_thth, g_pp = g_schwartz(int.u[fail_indx, 1:3], Mass_NS);
                        
            dωdr_grd = -grad(GJ_Model_ωp_vecSPH(seed(int.u[fail_indx, 1:3]), exp.(int.t), θm, ωPul, B0, rNS)); # [eV / km, eV, eV]
            dωdr_grd ./= sqrt.(dωdr_grd[:, 1].^2 .* g_rr .+ dωdr_grd[:, 2].^2 .* g_thth  .+ dωdr_grd[:, 3].^2 .* g_pp) # eV / km, net: [ , km , km]

            int.u[fail_indx, 4:6] .= int.u[fail_indx, 4:6] .- 2.0 .* (int.u[fail_indx, 4] .* dωdr_grd[:, 1] .* g_rr .+ int.u[fail_indx, 5] .* dωdr_grd[:, 2] .* g_thth .+ int.u[fail_indx, 6] .* dωdr_grd[:, 3] .* g_pp) .* dωdr_grd;
            print((int.u[fail_indx, 4] .* dωdr_grd[:, 1] .* g_rr .+ int.u[fail_indx, 5] .* dωdr_grd[:, 2] .* g_thth .+ int.u[fail_indx, 6] .* dωdr_grd[:, 3] .* g_pp), "\n")
        end
    end
    function cond(u, lnt, integrator)
        r_s0 = 2.0 * Mass_NS * GNew / c_km^2
        AA = sqrt.(1.0 .- r_s0 ./ u[:, 1])
        
        test = (erg ./ AA .- GJ_Model_ωp_vecSPH(u, exp.(lnt), θm, ωPul, B0, rNS)) ./ (erg ./ AA) .+ 1e-3 # trigger when closer to reflection....
        
        return minimum(test)
        
    end
    cb = ContinuousCallback(cond, floor_aff!, repeat_nudge=1//100, rootfind=DiffEqBase.RightRootFind)

    # Define the ODEproblem
    #!!!
    #prob = ODEProblem(func!, u0, tspan, [ω, Mvars], reltol=1e-5, abstol=ode_err, max_iters=1e5, callback=cb, dtmin=1e-13, dtmax=1e-2, force_dtmin=true)
    prob = ODEProblem(rhs, u0, tspan, [ω, Mvars], reltol=1e-5, abstol=ode_err, max_iters=1e5, callback=cb, dtmin=1e-13, dtmax=1e-2, force_dtmin=true)
    #!!!

    # Solve the ODEproblem
    sol = solve(prob, Vern6(), saveat=saveat)
    # sol = solve(prob, lsoda(), saveat=saveat)
    
    for i in 1:length(sol.u)
        sol.u[i][:,4:6] .*= erg
    end
   
    
    # Define the Schwarzschild radii (in km)
    r_s = 2.0 .* ones(length(sol.u[1][:,1]), length(sol.u)) .* Mass_NS .* GNew ./ c_km^2
    
    # print(sum(sol.u[end][:,1] .< 1e4), "\t", sol.u[end][:,1] , "\n")
    # Calculate the total particle energies (unitless); this is later used to find the resonance and should be constant along the trajectory
    for i in 1:length(sol.u)
        sol.u[i][sol.u[i][:,1] .<= r_s[:,i], 1] .= 2.0 .* Mass_NS .* GNew ./ c_km^2 .+ 1e-10
    end
    ω = [(1.0 .- r_s[:,i] ./ sol.u[i][:,1]) for i in 1:length(sol.u)]



    # Switch back to proper velocity
    v_pl = [[sol.u[i][:,4] .* sqrt.(ω[i])  sol.u[i][:,5] ./ sol.u[i][:,1] sol.u[i][:,6] ./ (sol.u[i][:,1] .* sin.(sol.u[i][:,2])) ] .* ω[i] for i in 1:length(sol.u)]
    
    # Switch back to Cartesian coordinates
    x = [[sol.u[i][:,1] .* sin.(sol.u[i][:,2]) .* cos.(sol.u[i][:,3])  sol.u[i][:,1] .* sin.(sol.u[i][:,2]) .* sin.(sol.u[i][:,3])  sol.u[i][:,1] .* cos.(sol.u[i][:,2])] for i in 1:length(sol.u)]

    
    v = [[cos.(sol.u[i][:,3]) .* (sin.(sol.u[i][:,2]) .* v_pl[i][:,1] .+ cos.(sol.u[i][:,2]) .* v_pl[i][:,2]) .- sin.(sol.u[i][:,2]) .* sin.(sol.u[i][:,3]) .* v_pl[i][:,3] ./ sin.(sol.u[i][:,2]) sin.(sol.u[i][:,3]) .* (sin.(sol.u[i][:,2]) .* v_pl[i][:,1] .+ cos.(sol.u[i][:,2]) .* v_pl[i][:,2]) .+  sin.(sol.u[i][:,2]) .* cos.(sol.u[i][:,3]) .* v_pl[i][:,3] ./ sin.(sol.u[i][:,2]) cos.(sol.u[i][:,2]) .* v_pl[i][:,1] .-  sin.(sol.u[i][:,2]) .* v_pl[i][:,2] ] for i in 1:length(sol.u)]
    
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
    
    
    return x_reshaped, v_reshaped, dt, fail_indx
    
end


function g_schwartz(x0, Mass_NS; rNS=10.0)
    # (1 - r_s / r)
    # notation (-,+,+,+), upper g^mu^nu
    rs = 2 * GNew .* Mass_NS ./ c_km.^2 .* ones(length(x0[:,1]))
    r = x0[:,1]
    
    # turn off GR inside NS
    rs[r .<= rNS] .= 0.0
    
    sin_theta = sin.(x0[:,2])
    g_tt = -1.0 ./ (1.0 .- rs ./ r);
    g_rr = (1.0 .- rs ./ r);
    g_thth = 1.0 ./ r.^2; # 1/km^2
    g_pp = 1.0 ./ (r.^2 .* sin_theta.^2); # 1/km^2

  
    return g_tt, g_rr, g_thth, g_pp
    
end


function hamiltonian(x, k,  time0, erg, θm, ωPul, B0, rNS, Mass_NS; iso=true, melrose=false)
    omP = GJ_Model_ωp_vecSPH(x, time0, θm, ωPul, B0, rNS);
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

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
function hamiltonian_axion(x, k,  time0, erg, θm, ωPul, B0, rNS, Mass_NS, mass_axion; iso=true, melrose=false)
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x, Mass_NS);
    ksqr = g_tt .* erg.^2 .+ g_rr .* k[:, 1].^2 .+ g_thth .* k[:, 2].^2 .+ g_pp .* k[:, 3].^2
    
    # Why factor 1/2 ??
    Ham = 0.5 .* ksqr .+ mass_axion

    return Ham
end
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

function k_norm_Cart(x0, khat,  time0, erg, θm, ωPul, B0, rNS, Mass_NS; melrose=false)
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2

    # Switch to polar coordinates
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* khat, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* khat[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* khat[:,1] .+ x0[:,1] .* khat[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    
    # Switch to celerity in polar coordinates
    AA = (1.0 .- r_s0 ./ rr)
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    
    omP = GJ_Model_ωp_vecSPH(x0_pl, time0, θm, ωPul, B0, rNS);
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    
    # ksqr = g_tt .* erg.^2 .+ g_rr .* w0_pl[:, 1].^2 .+ g_thth .* w0_pl[:, 2].^2 .+ g_pp .* w0_pl[:, 3].^2
    
    ctheta = Ctheta_B_sphere(x0_pl, w0_pl, [θm, ωPul, B0, rNS, time0, Mass_NS])

    if !melrose
        norm = sqrt.(abs.( (omP.^2 .* (1.0 .- ctheta.^2) ./ (omP.^2 .* ctheta.^2 .- erg.^2 ./ g_rr)  .* erg.^2 ./ g_rr .- g_tt .* erg.^2 ) ./ (g_rr .* w0_pl[:, 1].^2 .+ g_thth .* w0_pl[:, 2].^2 .+ g_pp .* w0_pl[:, 3].^2)))
    else
        k_no_norm2 = (g_rr .* w0_pl[:, 1].^2 .+ g_thth .* w0_pl[:, 2].^2 .+ g_pp .* w0_pl[:, 3].^2)
        norm = sqrt.((g_tt .* erg.^2 .+ omP.^2) ./ (k_no_norm2 .* (omP.^2 .* ctheta.^2 ./ (erg.^2 ./ g_rr) .- 1.0)))
    end
    
    
    return norm .* khat
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
function GJ_Model_vec(x, t, θm, ω, B0, rNS)
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
    

    # classical...
    dωdr_grd_2 = grad(GJ_Model_ωp_vec(seed(x0), t_start, θm, ωPul, B0, rNS))
    snorm_2 = dωdr_grd_2 ./ sqrt.(sum(dωdr_grd_2 .^ 2, dims=2))
    ctheta = (sum(k0 .* snorm_2, dims=2) ./ sqrt.(sum(k0 .^ 2, dims=2)))
    
    
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

function K_par(x0, k0, Mvars)
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
    return (k0[:,1] .* Br .+ k0[:,2] .* Btheta .+ k0[:,3] .* Bphi) ./ Bnorm
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

function GJ_Model_ωp_vecSPH(x, t, θm, ω, B0, rNS)
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
    

    ωp[r .<= rNS] .= 0.0;

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

function find_samples(maxR, ntimes_ax, θm, ωPul, B0, rNS, Mass_a, Mass_NS)
    batchsize = 2;


    tt_ax = LinRange(-2*maxR, 2*maxR, ntimes_ax); # Not a real physical time -- just used to get trajectory crossing
    
    # randomly sample angles θ, ϕ
    θi = acos.(1.0 .- 2.0 .* rand(batchsize));
    ϕi = rand(batchsize) .* 2π;
    
    vvec_all = [sin.(θi) .* cos.(ϕi) sin.(θi) .* sin.(ϕi) cos.(θi)];
    # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
    ϕRND = rand(batchsize) .* 2π;
    # rRND = sqrt.(rand(batchsize)) .* maxR; standard flat sampling
    rRND = rand(batchsize) .* maxR; # New 1/r sampling
    x1 = rRND .* cos.(ϕRND);
    x2 = rRND .* sin.(ϕRND);
    # rotate using Inv[EurlerMatrix(ϕi, θi, 0)] on vector (x1, x2, 0)
    x0_all= [x1 .* cos.(-ϕi) .* cos.(-θi) .+ x2 .* sin.(-ϕi) x2 .* cos.(-ϕi) .- x1 .* sin.(-ϕi) .* cos.(-θi) x1 .* sin.(-θi)];
    x_axion = [transpose(x0_all[i,:]) .+ transpose(vvec_all[i,:]) .* tt_ax[:] for i in 1:batchsize];

    cxing_st = [get_crossings(log.(GJ_Model_ωp_vec(x_axion[i], 0.0, θm, ωPul, B0, rNS)) .- log.(Mass_a)) for i in 1:batchsize];
    cxing = [apply(cxing_st[i], tt_ax) for i in 1:batchsize];
    # see if any crossings
    indx_cx = [if length(cxing[i]) .> 0 i else -1 end for i in 1:batchsize];
    # remove those which dont
    indx_cx_cut = indx_cx[indx_cx .> 0];
    # assign index for random point selection
    randInx = [rand(1:length(cxing[indx_cx_cut][i])) for i in 1:length(indx_cx_cut)];
    cxing_short = [cxing[indx_cx_cut][i][randInx[i]] for i in 1:length(indx_cx_cut)];
    weights = [length(cxing[indx_cx_cut][i]) for i in 1:length(indx_cx_cut)];


    numX = length(cxing_short);
    R_sample = vcat([rRND[indx_cx_cut][i] for i in 1:numX]...);

    # print(cxing, "\t", cxing_short, "\t", indx_cx_cut, "\t", randInx, "\n")
    
    if numX != 0
        # print(x0_all, "\t", cxing_short, "\t", indx_cx_cut, "\t", randInx,"\n")
        xpos = [transpose(x0_all[indx_cx_cut[i], :]) .+ transpose(vvec_all[indx_cx_cut[i], :]) .* cxing_short[i] for i in 1:numX];
        vvec_full = [transpose(vvec_all[indx_cx_cut[i],:]) .* ones(1, 3) for i in 1:numX];
        

        # print(x0_all, "\t", xpos, "\t", vvec_full, "\t", R_sample, "\n")
        t_new_arr = LinRange(- abs.(tt_ax[3] - tt_ax[1]), abs.(tt_ax[3] - tt_ax[1]), 100);
        xpos_proj = [xpos[i] .+ vvec_full[i] .* t_new_arr[:] for i in 1:numX];

        cxing_st = [get_crossings(log.(GJ_Model_ωp_vec(xpos_proj[i], 0.0, θm, ωPul, B0, rNS)) .- log.(Mass_a)) for i in 1:numX];
        cxing = [apply(cxing_st[i], t_new_arr) for i in 1:numX];
        indx_cx = [if length(cxing[i]) .> 0 i else -1 end for i in 1:numX];
        indx_cx_cut = indx_cx[indx_cx .> 0];
        R_sample = R_sample[indx_cx_cut];
        numX = length(indx_cx_cut);
        if numX == 0
            return 0.0, 0.0, 0, 0.0
        end



        randInx = [rand(1:length(cxing[indx_cx_cut][i])) for i in 1:numX];
        cxing = [cxing[indx_cx_cut][i][randInx[i]] for i in 1:numX];
        vvec_flat = reduce(vcat, vvec_full);
        # print(xpos, "\t", vvec_full, "\t", cxing, "\n")
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
        indx_r_cut = rmag .> (rNS + 0.5); # add .5 km buffer
        # print(xpos_flat, "\t", vvec_flat,"\t", R_sample, "\t", indx_r_cut, "\n")
        if sum(indx_r_cut) - length(xpos_flat[:,1 ]) < 0
            xpos_flat = xpos_flat[indx_r_cut[:], :]
            vvec_flat = vvec_flat[indx_r_cut[:], :]
            R_sample = R_sample[indx_r_cut[:]]
            numX = length(xpos_flat);
            rmag = sqrt.(sum(xpos_flat .^ 2, dims=2));
        end
        
        ntrajs = length(R_sample)
        if ntrajs == 0
            return 0.0, 0.0, 0, 0.0
        end
        
        # print("here...\t", xpos_flat, "\t", R_sample,"\n")
        ωpL = GJ_Model_ωp_vec(xpos_flat, zeros(ntrajs), θm, ωPul, B0, rNS)
        vmag = sqrt.(2 * 132698000000.0 .* Mass_NS ./ rmag) ; # km/s
        erg_ax = sqrt.( Mass_a^2 .+ (Mass_a .* vmag / 2.998e5) .^2 );
        
        # make sure not in forbidden region....
        fails = ωpL .> erg_ax;
        n_fails = sum(fails);
        if n_fails > 0
            
            ωpLi2 = [if fails[i] == 1 Mass_a .- GJ_Model_ωp_vec(transpose(xpos_flat[i,:]) .+ transpose(vvec_flat[i,:]) .* t_new_arr[:], [0.0], θm, ωPul, B0, rNS) else -1 end for i in 1:ntrajs];
            # ωpLi2 = [if fails[i] == 1 Mass_a .- GJ_Model_ωp_vec(xpos_flat[i,:] .+ vvec_flat[i,:] .* t_new_arr[:], [0.0], θm, ωPul, B0, rNS) else -1 end for i in 1:ntrajs];

            t_new = [if length(ωpLi2[i]) .> 1 t_new_arr[findall(x->x==ωpLi2[i][ωpLi2[i] .> 0][argmin(ωpLi2[i][ωpLi2[i] .> 0])], ωpLi2[i])][1] else -1e6 end for i in 1:length(ωpLi2)];
            t_new = t_new[t_new .> -1e6];
            xpos_flat[fails[:],:] .+= vvec_flat[fails[:], :] .* t_new;

        end
        # print(xpos_flat, "\t")
        return xpos_flat, R_sample, ntrajs, weights
    else
        return 0.0, 0.0, 0, 0.0
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
