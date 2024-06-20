module AffineInvMCMC

import RobustPmap
import Random

export runMCMCsample, flattenMCMCarray 


"""
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
**** FUNCTIONS FROM AffineInvaraintMCMC.jl PACKAGE ****
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
"""
function runMCMCsample(llhood::Function, numwalkers::Int, x0::Array, numsamples_perwalker::Integer, thinning::Integer, a::Number=2.; rng::Random.AbstractRNG=Random.GLOBAL_RNG)
	@assert length(size(x0)) == 2
	x = copy(x0)
	chain = Array{Float64}(undef, size(x0, 1), numwalkers, div(numsamples_perwalker, thinning))
	lastllhoodvals = RobustPmap.rpmap(llhood, map(i->x[:, i], 1:size(x, 2))) # Function to get the current L llhood values (L walkers, one value per walker) 
	llhoodvals = Array{Float64}(undef, numwalkers, div(numsamples_perwalker, thinning))
	llhoodvals[:, 1] = lastllhoodvals
	chain[:, :, 1] = x0
	batch1 = 1:div(numwalkers, 2)
	batch2 = div(numwalkers, 2) + 1:numwalkers
	divisions = [(batch1, batch2), (batch2, batch1)]
	tinit = time_ns()
	for i = 1:numsamples_perwalker
		#= if i % 100 == 0
			println("*** Step ", i, " of ", numsamples_perwalker, " *** Elapsed time: ", round( (time_ns()-tinit)/1.0e9/60, digits = 1), " minutes ")
			flush(stdout)
		end =#
		for ensembles in divisions
			active, inactive = ensembles
		
			#= Density g of the scaling variable Z: 
			g(z) = a/(2(a-1)*Sqrt[a z]) for 1/a <= z <= a; 0 elsewhere

			It is easy to verify that 
			Integrate[a/(2 (-1 + a) Sqrt[a z]), {z, 1/a, a}, Assumptions -> {a \[Element] Reals, a > 1}] = 1 as expected

			Cumulative distribution function:
			G(z) = (sqrt(a z) - 1) / (a-1) for 1/a <= z <= a; 0 elsewhere
			It is easy to verify that G(1/a) = 0 and G(a) = 1 as expected 
			Moreover, G'(z) = g(z)

			Now, sampling from g(z), with u ~ U[0,1]:
			u = G(z) -> G^(-1)(u) = z

			u = (sqrt(a z) - 1) / (a-1) -> z = ((a-1)u + 1)^2 / a =# 

			zs = map(u->((a - 1) * u + 1)^2 / a, rand(rng, length(active)))
			#= Apply stretch move to active walkers, each with a random inactive =#
			proposals = map(i->zs[i] * x[:, active[i]] + (1 - zs[i]) * x[:, rand(rng, inactive)], 1:length(active))  # stretch move
			newllhoods = RobustPmap.rpmap(llhood, proposals)
			for (j, walkernum) in enumerate(active)
				z = zs[j]
				newllhood = newllhoods[j]
				proposal = proposals[j]
				logratio = (size(x, 1) - 1) * log(z) + newllhood - lastllhoodvals[walkernum]
				if log(rand(rng)) < logratio
					lastllhoodvals[walkernum] = newllhood
					x[:, walkernum] = proposal
				end
				if i % thinning == 0
					chain[:, walkernum, div(i, thinning)] = x[:, walkernum]
					llhoodvals[walkernum, div(i, thinning)] = lastllhoodvals[walkernum]
				end
			end
		end
	end
	return chain, llhoodvals
end

"Flatten MCMC arrays"
function flattenMCMCarray(chain::Array, llhoodvals::Array)
	numdims, numwalkers, numsteps = size(chain)
	newchain = Array{Float64}(undef, numdims, numwalkers * numsteps)
	for j = 1:numsteps
		for i = 1:numwalkers
			newchain[:, i + (j - 1) * numwalkers] = chain[:, i, j]
		end
	end
	return newchain, llhoodvals[1:end]
end

end
