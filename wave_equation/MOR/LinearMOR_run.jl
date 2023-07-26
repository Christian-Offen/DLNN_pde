# load packages
using JSON
using LinearAlgebra
using Random
using Flux
using ForwardDiff
using Dates

# load tools
include("LdNetworkArchitecture.jl")
include("MLSetup.jl")

# load model
run_data=JSON.parsefile("2023-01-31_08-53-14run_data.json");

# set parameters of space-time domain
l = 1. # length of spatial domain including (periodic) boundary
T = .5 # final time

# discretisation parameters
M = length(run_data["training_data"][1]) # spatial grid points 
N = length(run_data["training_data"][1][1])-1 # time steps

dx = l/M # periodic boundary conditions
dt = T/N

# initial values (periodic)
XMesh   = 0:dx:(M-1)*dx
XMeshbd = 0:dx:M*dx # repeat boundary
TMesh = 0:dt:N*dt

converter = x-> hcat(x...)
training_data=converter.(run_data["training_data"]);

rng = MersenneTwister(345);

data_training_ROM=vcat(training_data...);

SVD_data=svd(data_training_ROM');
#plot(SVD_data.S,yscale=:log10)

MOR_dim = 2;

MOR_basis = SVD_data.U[:,1:MOR_dim];
data_red = MOR_basis'*data_training_ROM';

training_data_red=(x-> MOR_basis'*x').(training_data);

# absolute reconstruction error || A-MOR_basis*MOR_basis'*A ||
print("absolute reconstruction error (spectral/Frobenius): "*string(SVD_data.S[MOR_dim+1])*", "*string(sqrt(sum(SVD_data.S[MOR_dim+1:end].^2)))*"\n")   # spectral / Frobenius  (equates opnorm(reconst_error,2), norm(reconst_error,2))

# relative projection error
data_reconstr = MOR_basis*data_red
reconst_error=data_training_ROM' - data_reconstr
reconstr_error_average = sum( sqrt.(sum(reconst_error.^2,dims=1)./sum((data_training_ROM').^2,dims=1))) /size(data_training_ROM,1)
print("relative reconstruction error (average): "*string(reconstr_error_average)*"\n")

triple_collect = [training_data_red[k][:,j:j+2] for j=1:N-1 for k=1:length(training_data_red)]; # data triples

# batch training_data
train_loaderROM=Flux.DataLoader(transpose(triple_collect), batchsize=10, shuffle=true);

# form 3-layer neural network for discrete Lagrangian 

layersize=10
input_size=2*MOR_dim

LdArchitectureROM,paramsVecROM = InitLd(rng; inputWidth=input_size, NNwidth=layersize, sigma = softplus);

function odeDEL(Ld,q0,q1,q2)
    DELPre = q1-> Ld(q0,q1)+Ld(q1,q2)
    return ForwardDiff.gradient(DELPre,q1)
end

function odeDELTestROM(paramVec,dataTriple)
    return odeDEL((q0,q1)->LdArchitectureROM(paramVec,[q0;q1]),dataTriple[:,1],dataTriple[:,2],dataTriple[:,3])
end

# data as elements of Flux.DataLoader
function lossDataROM(paramVec,data)
    ls = 0.
    for dataTriple in data
        ls = ls + sum(odeDELTestROM(paramVec,dataTriple').^2)
    end
    return ls
end

iterStart = randn(rng,MOR_dim)
iterStart = iterStart ./ norm(iterStart)

function invIterM2Steps(M,steps)
    # smallest eigenvalue of M^T*M

    MChol = cholesky(transpose(M)*M,check=false)
    vNrm = iterStart
    lambda = 0.

    for j = 1:steps
        v = vNrm
        v = MChol\v
        vNrm = v ./ norm(v)
        lambda = 1/(transpose(vNrm)*v)
    end

    return lambda
end

stepsInvVecIter = 3;

function RegulariserLdode(Ld,qtriple)
    M = ForwardDiff.hessian(Ld,[qtriple[:,2];qtriple[:,3]])[MOR_dim+1:end,1:MOR_dim]
    lambda = invIterM2Steps(M,stepsInvVecIter) # square of smallest singular value (squared)
    return 1/lambda
    #return M
end


# data as elements of Flux.DataLoader
function lossRegulariserROM(paramVec,data)
    ls = 0.
    for dataTriple in data
        ls = ls + RegulariserLdode(x0->LdArchitectureROM(paramVec,x0),dataTriple')
    end
    return ls
end


function lossesROM(paramVec,data)
    return [lossDataROM(paramVec,data);lossRegulariserROM(paramVec,data)]
end

weights_loss_ROM = [1.,1e-8]

function lossROM(paramVec,data)
    return dot(weights_loss_ROM,lossesROM(paramVec,data))
end

lossesROM(paramsVecROM,train_loaderROM.data)

function lossROM_gradient(paramVec,data)
    loss_pre(paramVec) = lossROM(paramVec,data)
    return ForwardDiff.gradient(loss_pre,paramVec)
end

train_batchedROM!, save_paramsROM, save_TrainingDataROM = TrainingFunctions(lossesROM,lossROM_gradient,ADAM());

## Uncomment to train the model
train_batchedROM!(train_loaderROM,paramsVecROM,100000; print_every=1, save_every=10)


