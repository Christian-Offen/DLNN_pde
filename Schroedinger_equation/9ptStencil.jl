using ForwardDiff
using NLsolve
using LinearAlgebra

modInd(i::Int) = mod(i-1,M)+1


function DiscretiseLDensity(L)

	function Ldx(dx,U,Udot)
	    UMP    = ([U[:,2:M] U[:,1]] + U)/2
	    UdotMP = ([Udot[:,2:M] Udot[:,1]] + Udot)/2
	    UX = ([U[:,2:M] U[:,1]] - U)/dx
	    
	    return dx*sum([L(UMP[:,j],UdotMP[:,j],UX[:,j]) for j=1:M])
	end

	Ldxd(dt,dx,U0,U1) = dt*Ldx(dx,(U1+U0)/2,(U1-U0)/dt)

	function Ld(dx,dt,u,uup,uright,uupright)
	    centre = 1/4*(u+uup+uright+uupright)
	    ut = 1/(2*dt)*(uupright-uright+uup-u)
	    ux = 1/(2*dx)*(uupright+uright-uup-u)
	    return L(centre,ut,ux)
	end 

	momentumdx(dx,U) = dx/4*[0 1;-1 0]*([U[:,2:M] U[:,1]] +2*U + [U[:,end] U[:,1:M-1]])

	function firstStep(dx,dt,U0)
	    obj1(U1) = momentumdx(dx,U0) .+ ForwardDiff.gradient(U0->Ldxd(dt,dx,U0,U1),U0)
	    return nlsolve(obj1,U0,autodiff = :forward)
	end

return Ld,Ldx,Ldxd,firstStep

end

# (generic) variational midpoint rule for general discrete Lagrangians Ld

function DELSolve(Ld,q0,q1)
    q2guess = 2*q0-q1
    DELObjective(q2) = ForwardDiff.gradient(q1->Ld(q0,q1)+Ld(q1,q2),q1)
    return nlsolve(DELObjective,q2guess,autodiff = :forward)
end

function DELSolve(Ld,q0,q1,steps)

    trj = zeros((size(q0)...,steps+1))
    trj[:,:,1]=q0
    trj[:,:,2]=q1
    
    for j = 1:steps-1        
        trj[:,:,j+2] = DELSolve(Ld,trj[:,:,j],trj[:,:,j+1]).zero
    end
    
    return trj
end


# collect stencils from solution U over mesh

function CollectStencils(U::Matrix{Float64})
    
    N = size(U,2)-1 # time steps
    M = size(U,1)   # spacial steps
    
    # periodic spatial indices
    modInd(i::Int) = mod(i-1,M)+1
    
    Collection = zeros((N-1)*M,9)
    
    for i=2:N
            for j=1:M
                k = j + (i-2)*M
                # uupleft,uup,uupright, uleft,u,uright, udownleft, udown,udownright
                Collection[k,:] = [U[modInd(j-1),i+1] U[j,i+1] U[modInd(j+1),i+1] U[modInd(j-1),i] U[j,i] U[modInd(j+1),i] U[modInd(j-1),i-1] U[j,i-1] U[modInd(j+1),i-1]]
            end
        end

    return Collection
    
end

function CollectStencils(U::Array{Float64,3})
    N = size(U,3)-1 # time steps
    M = size(U,2)   # spacial steps
    K = size(U,1)   # dimensions
    
    Collection = zeros((N-1)*M,9,K)
    
    for k = 1:K
        Collection[:,:,k] = CollectStencils(U[k,:,:])
    end
    
    return Collection
end


# tools to verify solutions / check data consistency

function DiscreteEL(Ld, uupleft,uup,uupright, uleft,u,uright, udownleft,udown,udownright)
     
        # discrete Euler-Lagrange equation
        DELpre = u -> Ld(u,uup,uright,uupright)+Ld(udown,u,udownright,uright)+Ld(udownleft,uleft,udown,u)+Ld(uleft,uupleft,u,uup)
        return ForwardDiff.gradient(DELpre, u) 
end

function DiscreteEL(Ld,stencil)
    uupleft = stencil[:,1]
    uup  = stencil[:,2]
    uupright  = stencil[:,3]
    uleft  = stencil[:,4]
    u  = stencil[:,5]
    uright = stencil[:,6]
    udownleft  = stencil[:,7]
    udown  = stencil[:,8]
    udownright  = stencil[:,9]
    return DiscreteEL(Ld, uupleft,uup,uupright, uleft,u,uright, udownleft,udown,udownright)
end


function DataConsistency(Ld,U)
    # size(U) = (#dimensions=2, #spacial points, #temporal grid points)
    stencils = CollectStencils(U::Array{Float64,3})
    return sum([sum(DiscreteEL(Ld,transpose(stencils[j,:,:])).^2) for j=1:size(stencils,1)])
end
    
