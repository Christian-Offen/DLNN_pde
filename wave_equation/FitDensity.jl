using Flux
using FFTW
using Random
include("7ptStencilFun.jl")
include("TrainingData.jl")

using SliceMap  # Zygote compatible alais of mapslices
using ForwardDiff
using ReverseDiff
using Plots     
#using BenchmarkTools

using Dates	# for saving the run
using JSON	# for saving the run

println(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"))

NoSamples=80;	# number of solutions to PDE used for training

## time-spacial domain
l = 1. # length of spatial domain including (periodic) boundary
T = .5 # final time

# discretisation parameters
M = 20 # spatial grid points 
N = 20 # time steps
dx = l/M # periodic boundary conditions
dt = T/N
XMesh   = 0:dx:(M-1)*dx # periodic mesh
XMeshbd = 0:dx:M*dx # periodic mesh
TMesh = 0:dt:N*dt       

# Lagrangian 
Potential(u) = 1/2  * u^2
Lagrangian(u,ut,ux) = 1/2*ut^2-1/2*ux^2-Potential(u)
Ld_ref(u,udown,uleft) = Lagrangian(u,(u-udown)/dt,(u-uleft)/dx)
Ld_ref(x0) = Ld_ref(x0[1],x0[2],x0[3])

println("Compute training data")

training_dataMatrix,training_dataU = CreateTrainingData(Lagrangian; NoSamples=NoSamples,l = l,T = T, M = M, N = N);

# batch training_data
train_loader=Flux.DataLoader(transpose(training_dataMatrix), batchsize=10, shuffle=true);

# visualise training data
#sampleNo = 13
#UTrainingSample = training_dataU[:,:,sampleNo]

#UPlot = [UTrainingSample UTrainingSample[:,1]] # add repeated boundary for plotting
#plot(XMeshbd,TMesh,UPlot,st=:surface,xlabel="x",ylabel="t")

println("Set up neural network and training functions")

function LdLearn(paramVec,x0)
    
    
    A1 = reshape(paramVec[1:10*3],(10,3))
    A2 = reshape(paramVec[10*3+1: 10*3+10*10],(10,10))
    A3 = reshape(paramVec[10*3+10*10+1:10*3+10*10+1*10],(1,10))
    
    b1 = paramVec[10*3+10*10+1*10+1:10*3+10*10+1*10+10]
    b2 = paramVec[10*3+10*10+1*10+10+1:end]

    
    x1 = tanh.(A1*x0 + b1)
    x2 = tanh.(A2*x1 + b2)
    x3 = A3*x2
    
    return x3[1]
    
end

# initial parameters

A1 = rand(10,3)
A2 = rand(10,10)
A3 = rand(1,10)

b1 = rand(10)
b2 = rand(10)

paramsVec=[A1[:];A2[:];A3[:];b1[:];b2[:]];


function RegulariserSolvability(Ld,uupuuupleft)
    
    nabla_uuup = ForwardDiff.hessian(Ld,uupuuupleft)[1,2]^2
    
    return 1/nabla_uuup^2
    
end


function lossRegulariserSolvabilityPre(paramsVec,stencil)
    # u,uup,uleft,udown,uright,uupleft,udownright = stencil
    return RegulariserSolvability(x0->LdLearn(paramsVec,x0),[stencil[2],stencil[1],stencil[6]])  # uup, u, uupleft
    
end

lossRegulariserSolvability(paramsVec,data) = sum(mapcols(stencil->lossRegulariserSolvabilityPre(paramsVec,stencil),data).^2)/(size(data)[2])

lossRegulariserSolvabilityPre_gradient(paramsVec,stencil) = ForwardDiff.gradient(paramsV->lossRegulariserSolvabilityPre(paramsV,stencil),paramsVec)
lossRegulariserSolvability_gradient(paramsVec,data) = ForwardDiff.gradient(paramsV->lossRegulariserSolvability(paramsV,data),paramsVec)


# discrete Euler-Lagrange equation
function DiscreteEL(Ld,stencil)
    u,uup,uleft,udown,uright,uupleft,udownright = stencil   
    DELpre = u -> (Ld([u,udown,uleft])+Ld([uup,u,uupleft])+Ld([uright,udownright,u]))[1] 
    return ForwardDiff.derivative(DELpre, u) #[1]
    #return gradient(DELpre, u)[1]
end

# loss for consistency of data with Euler-Lagrange equation
lossDataPre(paramsVec,stencil) = DiscreteEL(x0->LdLearn(paramsVec,x0),stencil)
lossData(paramsVec,data) = sum(mapcols(stencil->lossDataPre(paramsVec,stencil),data).^2)/(size(data)[2])

lossData_gradient(paramsVec,data) = ForwardDiff.gradient(paramsVec->lossData(paramsVec,data),paramsVec)



weights_losses = [1.,0.1];

loss(params,data) = weights_losses[1]*lossData(params,data) + weights_losses[2]*lossRegulariserSolvability(params,data)
loss_gradient(params,data) = weights_losses[1].*lossData_gradient(params,data) .+ weights_losses[2].*lossRegulariserSolvability_gradient(params,data)

# select optimiser
opt = ADAM()


# losses before training
println("Computed losses of prior")
println("Loss data consistency "*string(lossData(paramsVec,train_loader.data)))
println("Loss regulariser "*string(string(lossRegulariserSolvability(paramsVec,train_loader.data))))
println("weighted loss "*string(string(loss(paramsVec,train_loader.data))))

# initialise training loss variable
loss_hist = zeros(2,0)

# define training functions

function train_single_batch(data)
    
    # compute gradient
    grads=loss_gradient(paramsVec,data)
    
    # update parameters
    Flux.update!(opt, paramsVec, grads)
    
end


function train_batched(Data)
    
    for data in Data
        train_single_batch(data)
    end
   
    current_loss = [lossData(paramsVec,Data.data);lossRegulariserSolvability(paramsVec,Data.data)]
    
    return current_loss
    
end

function train_batched(Data,epochs::Int)
    
    for j = 1:epochs
        
        current_loss = train_batched(Data)
        
        global loss_hist=hcat(loss_hist,current_loss)        
        println(current_loss)
        flush(stdout);
        
    end
    
end

function train_batched(Data,epochs::Int,save_every::Int)
    
    for j = 1:epochs
        
        current_loss = train_batched(Data)
        global loss_hist=hcat(loss_hist,current_loss)        
        println(current_loss)
        flush(stdout);
        
        if mod(j,save_every) == 0
            save_my_run()
        end
        
    end
    
end


function train_batched_live(Data,epochs::Int)
    
    for j = 1:epochs
        
        current_loss = train_batched(Data)

        global loss_hist=hcat(loss_hist,current_loss)
        plot(transpose(loss_hist), show=true, yaxis =:log, labels=["DEL" "regulariser"])
        
    end
    
end


function save_my_run()

	nowrun = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

	run_dict = Dict("time" => nowrun, "training_data" => training_dataU, "learned_parameters" => paramsVec, "training_losses" => loss_hist)

	open(nowrun*"run_data.json","w") do f
	    JSON.print(f, run_dict)
	end

	# loading 
	# run_dict=JSON.parsefile("2022-09-01_09-50-24run_data.json")

	try savefig(nowrun*"_training_loss.pdf"); catch nothing; end
	
end

println("Set up complete. Now training.")
println(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"))


## train the network (run in REPL)
# current_loss = train_batched_live(train_loader,5)

## alternative
current_loss = train_batched(train_loader,3000,1)
myplot=plot(transpose(loss_hist), yaxis = :log, labels=["DEL" "regulariser"])

println("Training complete.")
println(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"))

## save
save_my_run()






