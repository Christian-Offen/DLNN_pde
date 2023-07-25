# Creation of training data for testing the "FitDensity.jl" machine learning architecture for Learning discrete Lagrangian densities


# Collect u,uup,uleft,udown,uright,uupleft,udownright from data
function CollectStencils(U)
       
    N = size(U)[1]-1 # time steps
    M = size(U)[2]   # spacial steps
    
    # periodic spatial indices
    modInd(i::Int) = mod(i-1,M)+1
    
    Collection = zeros((N-1)*M,7)
    
    for i=2:N
            for j=1:M
                k = j + (i-2)*M
                # u,uup,uleft,udown,uright,uupleft,udownright
                Collection[k,:] = [U[i,j] U[i+1,j] U[i,modInd(j-1)] U[i-1,j] U[i,modInd(j+1)] U[i+1,modInd(j-1)] U[i-1,modInd(j+1)]]
            end
        end

    return Collection
    
end


# Create Training Data (not sub-sampled)
function CreateTrainingData(Lagrangian; NoSamples=50,l = 1,T = .5, M = 10, N = 10, frequency_decay = [-2,4])

	## input arguments
	# l = 1. # length of spatial domain including (periodic) boundary
	# T = .5 # final time

	## discretisation parameters
	# M = 10 # spatial grid points 
	# N = 10 # time steps

	dx = l/M # periodic boundary conditions
	dt = T/N

	# sample periodic initial values
	no_freq=length(rfft(zeros(M)))
	rng = MersenneTwister(1234);
	random_vel=randn(rng, Float64,(NoSamples,M))
	random_freq=M*exp.(frequency_decay[1]*(0:no_freq-1).^frequency_decay[2]).*randn(rng, Float64,(no_freq,NoSamples));


	# pre-allocation 
	training_dataMatrix = zeros((N-1)*M*NoSamples,7) # collecting stencils
	training_dataU = zeros(N+1,M,NoSamples) # collecting PDE solution data over mesh

	for k in 1:NoSamples
	    
	    u0    = irfft(random_freq[:,k],M)
	    u0dot = random_vel[k,:]
	    
	    U = PDESolve(Lagrangian, u0, u0dot; interval_length = l, time_steps = N, t_final = T)
	    
	    training_dataMatrix[(k-1)*(N-1)*M+1:k*(N-1)*M,:] = CollectStencils(U)
	    
	    training_dataU[:,:,k] = U
	    
	end

	return training_dataMatrix, training_dataU

end
