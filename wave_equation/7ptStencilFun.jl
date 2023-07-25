using NLsolve
using Flux: gradient
using ForwardDiff


function PDEContinue(Ld, u0, u1; time_steps = 5)
    
    # u0,u1       initial values, not including the right boundary, periodic boundary conditions
    # u0dot    velocity profile 
    
    M = size(u0)[1]     # number of spatial_points (periodic boundary)
    N = time_steps      # time steps
    
        
    # 7 point stencil based on Lagrangian
    function Stencil7pt(u,udown,uleft,uupleft,uright,udownright)
     
        # discrete Euler-Lagrange equation
        function DEL(uup)
            DELpre = u -> Ld(u,udown,uleft)+Ld(uup,u,uupleft)+Ld(uright,udownright,u) 
            return gradient(DELpre, u)[1]
        end

        # pack into vector for nlsolve to understand; only necessary when u is scalar
        guess = [2. *u-udown]
        objective = uup -> DEL(uup[1])


        uupSolver= nlsolve(objective,guess,autodiff = :forward)
        return uupSolver.zero[1], uupSolver

    end
    
    
    # periodic spatial indices
    modInd(i::Int) = mod(i-1,M)+1
    
    
    # pre-allocate data
    U = zeros(N+1,M)
    U[1,:] = u0;
    U[2,:] = u1;
    
    
    # apply stencil
    for i=2:N
        for j=1:M
            U[i+1,j],solverInfo = Stencil7pt(U[i,j],U[i-1,j],U[i,modInd(j-1)],U[i+1,modInd(j-1)],U[i,modInd(j+1)],U[i-1,modInd(j+1)])
            #print(solverInfo)
        end
    end
    
    return U
    
end



function PDESolve(Lagrangian, u0, u0dot; interval_length = 1., time_steps = 5, t_final = 2.)
    
    # u0       initial values, not including the right boundary, periodic boundary conditions
    # u0dot    velocity profile 
    
    L = Lagrangian
    M = size(u0)[1]     # number of spatial_points (periodic boundary)
    N = time_steps      # time steps
    l = interval_length # length of spatial domain including (periodic) boundary
    T = t_final         # final time
    
    # discretisation parameters
    dx = l/M # periodic boundary conditions
    dt = T/N
    
    # Discretisation of Lagrangian
    FDdx(u) = [(u[1]-u[end]); (u[2:end]-u[1:end-1])]/dx
    L_delx(u,udot) = sum(dx*L.(u,udot,FDdx(u)))
    L_delxd(uold,u) = L_delx(u,(u-uold)/dt)

    Ld(u,udown,uleft) = L(u,(u-udown)/dt,(u-uleft)/dx)

    p_L_delx(u,udot)  = gradient(udot -> L_delx(u,udot),udot)[1]
    p0_L_delxd(u0,u1) = - gradient(u0->L_delxd(u0,u1),u0)[1]

    
    # compute first time step
    function u1Step(u0,u0dot)
        objective = u1 -> p_L_delx(u0,u0dot)-p0_L_delxd(u0,u1)
        guess = u0+dt*u0dot
        u1Solver = nlsolve(objective,guess,autodiff = :forward)
        return u1Solver.zero, u1Solver
    end
    
    
    # computation of 1st step
    u1,_ = u1Step(u0,u0dot)
    
    
    # compute remaining steps
    U = PDEContinue(Ld, u0, u1; time_steps = N)
    
    return U
    
end



# Verify solutions

function DiscreteEL(Ld,u,udown,uleft,uupleft,uup,uright,udownright)
     
        # discrete Euler-Lagrange equation
        DELpre = u -> real(Ld(u,udown,uleft)+Ld(uup,u,uupleft)+Ld(uright,udownright,u) )
        return gradient(DELpre, u)[1]
    
end


# check whether U fulfills DEL
function DELTest(Ld,U)

	N = size(U)[1]-1
	M = size(U)[2]

	# periodic spatial indices
	modInd(i::Int) = mod(i-1,M)+1

    UTest = zeros(N+1,M)
    for i=2:N
        for j=1:M
            UTest[i+1,j] = DiscreteEL(Ld,U[i,j],U[i-1,j],U[i,modInd(j-1)],U[i+1,modInd(j-1)],U[i+1,j],U[i,modInd(j+1)],U[i-1,modInd(j+1)])
        end
    end
    return UTest
end
