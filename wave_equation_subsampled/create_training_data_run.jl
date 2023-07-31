# Create training data for this experiment

include("7ptStencilFun.jl")
include("TrainingData.jl")
using Dates	# for saving the run
using JSON	# for saving the run

time_now() = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS") 
println(time_now())

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

_,training_dataU = CreateTrainingData(Lagrangian; NoSamples=NoSamples,l = l,T = T, M = M, N = N);

# save training data
dictData = Dict([("M",M),("N",N),("dx",dx),("dt",dt),("training_dataU",training_dataU),("data_time",time_now())])
write("training_data.json",JSON.json(dictData))