module SDDESolarDynamo

"""
Include the following lines in the main script:
---
include("./SDDESolarDynamo.jl")
using .SDDESolarDynamo
---
This gives access to functions 'bfield' and 'sn'
"""

export bfield, sn

using StochasticDelayDiffEq
using SpecialFunctions 

Bmin = 1  # --- fixed parameter 
# myseed  # --- use this only when you want to produce the same dataset

# --- Nonlinear function 
ftilde(x, Bmin, Bmax) = x/4*(1+erf(x^2-Bmin^2))*(1-erf(x^2-Bmax^2))

#########################################################################
############ B-field WITHOUT Jupiter ############
#########################################################################
# --- Model : B field (random, no seed)
function bfield(θ, Tsim)
	#= du = f(u,h,p,t)dt + g(u,h,p,t) dW_t with t≥t0 
	u(t0) = u0 
	u(t) = h(t) for t < t0 =#
	#= Example: sol = Bfield([40,35,5],[1,10,0.1,10000]) 
	sol.t -> time points 
	sol[1,:] -> B filed 
	sol[2,:] -> db/dt =#
	τ, T, Nd, sigma, Bmax = θ
	lags = (T, )
	h(p, t) = [Bmax, 0.]
	#################################
	function f(du,u,h,p,t)     # Drift function
		τ, T, Nd, sigma, Bmax = p
		hist = h(p, t - lags[1])[1]  # --- with Jupiter
		du[1] = u[2]    # u = [B, dB/dt]; du = [dB/dt, d^2B/dt^2]
        du[2] = -u[1]/τ^2-2*u[2]/τ - Nd/τ^2*ftilde(hist, Bmin, Bmax)
	end
	#################################
	function g(du,u,h,p,t)     # Diffusion function
		τ, T, Nd, sigma, Bmax = p
		du[1] = 0
		du[2] = Bmax*sigma/(τ^(3/2))
	end
	#################################
	tspan = (0.0, Tsim)
	#= SDDEProblem(f,g[, u0], h, tspan[, p]; <keyword arguments>) =#
	prob = SDDEProblem(f, g, [Bmax, 0.], h, tspan, θ; constant_lags = lags)
	solve(prob, EM(), dt=0.1, saveat=1.0)
end

# --- Model : B field (with seed)
function bfield(θ, Tsim, s)
	#= du = f(u,h,p,t)dt + g(u,h,p,t) dW_t with t≥t0 
	u(t0) = u0 
	u(t) = h(t) for t < t0 =#
	#= Example: sol = Bfield([40,35,5],[1,10,0.1,10000]) 
	sol.t -> time points 
	sol[1,:] -> B filed 
	sol[2,:] -> db/dt =#
	τ, T, Nd, sigma, Bmax = θ
	lags = (T, )
	h(p, t) = [Bmax, 0.]
	#################################
	function f(du,u,h,p,t)     # Drift function
		τ, T, Nd, sigma, Bmax = p
		hist = h(p, t - lags[1])[1]  # --- with Jupiter
		# hist = h(p, t - (1+eps*cos(2*π*(1/11.86)*t + mod(phi, 2*π))) * lags[1])[1]  # --- with Jupiter
		# hist = h(p, t - lags[1])[1]  # --- without Jupiter
		du[1] = u[2]    # u = [B, dB/dt]; du = [dB/dt, d^2B/dt^2]
        du[2] = -u[1]/τ^2-2*u[2]/τ - Nd/τ^2*ftilde(hist, Bmin, Bmax)
	end
	#################################
	function g(du,u,h,p,t)     # Diffusion function
		τ, T, Nd, sigma, Bmax = p
		du[1] = 0
		du[2] = Bmax*sigma/(τ^(3/2))
	end
	#################################
	tspan = (0.0, Tsim)
	#= SDDEProblem(f,g[, u0], h, tspan[, p]; <keyword arguments>) =#
	prob = SDDEProblem(f, g, [Bmax, 0.], h, tspan, θ; constant_lags = lags)
	solve(prob, EM(), dt=0.1, saveat=1.0, seed = s) # --- use seed only to generate same dataset 
end

#########################################################################
#########################################################################

# --- SN from B field
function sn(θ, args...; Twarmup = 200, Tobs = 929, s=nothing, kwargs...)
    # Data-generating model
	Tsim = Twarmup + Tobs  # Total simulation steps
	if isa(s, Number)
		y = (bfield(θ, Tsim, s)[1,2:end]).^2
	else
		y = (bfield(θ, Tsim)[1,2:end]).^2
	end

    data = y[Twarmup + 1:end]  # Get rid of warmup points
    return data
end

end