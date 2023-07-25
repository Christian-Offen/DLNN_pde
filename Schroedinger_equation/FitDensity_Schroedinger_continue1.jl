using Flux
using Random
rng = MersenneTwister(4321);

include("9ptStencil.jl")
include("TrainingData.jl")
include("LdNetworkArchitecture.jl")
include("LossFunction.jl")
include("MLSetup.jl")

using ForwardDiff
using Dates	# for saving the run
using JSON	# for saving the run

println(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"))

NoSamples=80;	# number of solutions to PDE used for training

## time-spacial domain
l = 1. # length of spatial domain including (periodic) boundary
T = .12 # final time

# discretisation parameters
M = 8 # spatial grid points 
N = 12 # time steps
dim = 2 # real dimension of target space of solution to PDE
dx = l/M # periodic boundary conditions
dt = T/N
XMesh   = 0:dx:(M-1)*dx # periodic mesh
XMeshbd = 0:dx:M*dx # periodic mesh
TMesh = 0:dt:N*dt       


# Lagrangian
V(r) = r
alpha(u) = [u[2];-u[1]]
H(u,ux) = sum(ux.^2) + V(sum(u.^2))
L_ref(u,ut,ux) = dot(alpha(u),ut)-H(u,ux)
Ld_ref,Ldx_ref,Ldxd_ref,firstStep_ref = DiscretiseLDensity(L_ref)
function Ld_refInstance(u,uup,uright,uupright); return Ld_ref(dx,dt,u,uup,uright,uupright); end

println("Load training data")

_,training_dataU = CreateTrainingData(L_ref, rng; NoSamples=NoSamples,l = l,T = T, M = M, N = N);#

#include("plotting_tools.jl")
#plotU, contourU = InstantiatePlotFun(dt,dx)
#plotU(transpose(training_dataU[1,:,:,5]))

train_loader=Flux.DataLoader(TrainingDataBlockForm(training_dataU), batchsize=2, shuffle=true) # batch training data

println("training data: number of stencils: " * string(length(train_loader.data)*M*dim))
println("training data in bytes: " * string(Base.summarysize(train_loader)))
println("number of batches: " * string(length(train_loader)));

println("Set up neural network and training functions")

LdLearn, paramsVec = InitLd(rng; NNwidth=12, sigma=softplus);
println("trainable parameters: "*string(length(paramsVec)))

# continue optimisation of previous run
run_data_update=JSON.parsefile("9_2023-04-26_12-41-07run_param_data.json")
paramsVec = Float64.(run_data_update["learned_parameters"])

## loss function

lossesLd = InstantiateLdLoss(rng)

# WEIGHTS data loss/regulariser
weights_losses = [1.,1.];

function loss(paramsVec,DataBlock)
    LdLearnInstance=(u,uup,uright,uupright) -> LdLearn(paramsVec,u,uup,uright,uupright)
    loss_datareg = lossesLd(LdLearnInstance,DataBlock)
    return weights_losses[1]*loss_datareg[1] + weights_losses[2]*loss_datareg[2]
end

function loss_gradient(params,data) 
    #return gradient(params->loss(params,data),params)
    return ForwardDiff.gradient(params->loss(params,data),params)
end

# instatiate Ld for info about prior
function LdLearnInstance(u,uup,uright,uupright); return LdLearn(paramsVec,u,uup,uright,uupright) end;

# losses before training
println("Losses of prior")

println("Loss data consistency / regulariser of reference Ld "*string(lossesLd(Ld_refInstance,train_loader.data)))
println("Loss data consistency / regulariser "*string(lossesLd(LdLearnInstance,train_loader.data)))

println("weighted loss "*string(string(loss(paramsVec,train_loader.data))))


# setup ML problem
opt = ADAM() # select optimiser
train_batched!, save_params, save_TrainingData = TrainingFunctions(LdLearn,lossesLd,loss_gradient,opt)

save_TrainingData(training_dataU)

println("Set up complete.")
println(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")) 

## training

no_epochs = 5000000 # upper bound, controlled via time out
save_every = 9

println("Now training.")
train_batched!(train_loader,paramsVec,no_epochs; print_every=1, save_every=save_every)

println("Training complete.")
println(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"))

## save
save_params(paramsVec,no_epochs)


