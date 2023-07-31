using Dates
using JSON
using SplitApplyCombine
using Flux
using SliceMap  # Zygote compatible alais of mapslices
using ForwardDiff

#include("plotting_tools.jl")
include("TrainingData.jl")
include("LdNetworkArchitecture.jl")
include("LossFunction.jl")
include("MLSetup.jl")

dynamicalData = JSON.parsefile("training_data.json")

# read and process training data for presentation
converter  = x-> hcat(x...)
converter2 = x -> converter.(x)
training_data = converter2(dynamicalData["training_dataU"]);

#plotU, contourU, contourU! = InstantiatePlotFun(dynamicalData["dt"],dynamicalData["dx"])
#plotU(training_data[1])

# subsampled 
leapi = 2
leapj = 2

# stencil data of subsampled solution
stencils_training_subsampled=(U->CollectStencils(U,leapi,leapj)).(training_data)
stencils_training_subsampled=vcat(stencils_training_subsampled...)
println("count of stencils: "*string(size(stencils_training_subsampled,1)))

train_loader=Flux.DataLoader(transpose(stencils_training_subsampled), batchsize=10, shuffle=true);

rng = MersenneTwister(1234)
LdLearn, paramsVec = InitLd(rng; NNwidth=10, sigma = tanh)
println("number of parameters in neural network: "*string(length(paramsVec)))

lossesLd, lossLd = InitLosses([1.,0.1])

function loss(params,data)
    LdInstance(stencil) = LdLearn(params,stencil)
    return lossLd(LdInstance,data)
end

function loss_gradient(params,data)
    return ForwardDiff.gradient(params->loss(params,data),params)
end

# losses before training
println("Losses of prior")
println("Loss data consistency / regulariser "*string(lossesLd(stencil -> LdLearn(paramsVec,stencil),train_loader.data)))
println("weighted loss "*string(string(loss(paramsVec,train_loader.data))))

# setup ML problem
opt = ADAM() # select optimiser
train_batched!, save_params = TrainingFunctions(LdLearn,lossesLd,loss_gradient,opt)

println("Set up complete.")
println(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")) 

## training

no_epochs = 132000     # upper bound, controlled via time-out
save_every = 10
print_every = 10

println("Now training.")
train_batched!(train_loader,paramsVec,no_epochs; print_every=print_every, save_every=save_every)

println("Training complete.")
println(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"))

## save
save_params(paramsVec,no_epochs)


