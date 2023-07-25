# Creation of training data for testing the "FitDensity.jl" machine learning architecture for Learning discrete Lagrangian densities
# requires 9ptStencil.jl 

using FFTW

# Create Training Data (not sub-sampled)
function CreateTrainingData(Lagrangian, rng; NoSamples=50,l = 1,T = .5, M = 10, N = 10, frequency_decayPhi = [-2,4],frequency_decayP = [-2,4],frequency_scalingP=0.2)

	## input arguments
	# rng MersenneTwister
	# l = 1. # length of spatial domain including (periodic) boundary
	# T = .5 # final time

	## discretisation parameters
	# M = 10 # spatial grid points 
	# N = 10 # time steps

	dx = l/M # periodic boundary conditions
	dt = T/N

    # discretise Lagrangian using 9 point stencil
    _,_,Ldxd,firstStep = DiscretiseLDensity(Lagrangian)
    LdxdInstance = (U0,U1) -> Ldxd(dt,dx,U0,U1)
    
	# sample periodic initial values
	no_freq=length(rfft(zeros(M)))
	random_freqPhi=M*exp.(frequency_decayPhi[1]*(0:no_freq-1).^frequency_decayPhi[2]).*randn(rng, Float64,(no_freq,NoSamples));
    random_freqP=frequency_scalingP*M*exp.(frequency_decayP[1]*(0:no_freq-1).^frequency_decayP[2]).*randn(rng, Float64,(no_freq,NoSamples));
    
	# pre-allocation 
	training_dataMatrix = zeros((N-1)*M*NoSamples,9,2) # collecting stencils
	training_dataU = zeros(2,M,N+1,NoSamples) # collecting PDE solution data over mesh

    
    
	for k in 1:NoSamples
	    
	    
	    phi0 = irfft(random_freqPhi[:,k],M)
        p0 = irfft(random_freqP[:,k],M)
        u0 = transpose([phi0 p0])
	    u1 = firstStep(dx,dt,u0).zero
        
	    Utrj = DELSolve(LdxdInstance,u0,u1,N)
	    
	    training_dataMatrix[(k-1)*(N-1)*M+1:k*(N-1)*M,:,:] = CollectStencils(Utrj)
	    
	    training_dataU[:,:,:,k] = Utrj
	    
	end

	return training_dataMatrix, training_dataU

end


function TrainingDataBlockForm(training_dataU)
    
    dim,M,N1,NoSamples = size(training_dataU)
    N = N1-1
    
    return [training_dataU[:,:,i-1:i+1,k] for k=1:NoSamples for i = 2:N]
end 
