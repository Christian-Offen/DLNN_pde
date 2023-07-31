function InitLosses(weights_losses = [1.,0.1])

	# discrete Euler-Lagrange equation
	function DiscreteEL(Ld,stencil)
	    u,uup,uleft,udown,uright,uupleft,udownright = stencil
        function DELpre(u)
            return (Ld([u,udown,uleft])+Ld([uup,u,uupleft])+Ld([uright,udownright,u]))[1] 
        end
	    return ForwardDiff.derivative(DELpre, u) #[1]
	end

	function lossData(Ld,data)
	
		consistencyCheck(stencil) = DiscreteEL(Ld,stencil)
		ConsistencyMat = mapcols(consistencyCheck,data).^2
		
		return sum(ConsistencyMat)/(size(data,2))
	end


	function RegulariserSolvability(Ld,uupuuupleft)
	    
	    nabla_uuup = ForwardDiff.hessian(Ld,uupuuupleft)[1,2]^2
	    
	    return 1/nabla_uuup^2
	    
	end


	function lossRegulariserSolvabilityPre(Ld,stencil)
	    # u,uup,uleft,udown,uright,uupleft,udownright = stencil
	    return RegulariserSolvability(Ld,[stencil[2],stencil[1],stencil[6]])  # uup, u, uupleft
	end
	
	

	function lossRegulariserSolvability(Ld,data)
        function regInstance(stencil)
            return lossRegulariserSolvabilityPre(Ld,stencil)
        end
		regMatrix = mapcols(regInstance,data).^2 
		return sum(regMatrix)/(size(data)[2])
	end

	function lossesLd(Ld,data)
		return [lossData(Ld,data); lossRegulariserSolvability(Ld,data)]
	end

	function lossLd(Ld,data)
		return transpose(weights_losses)*lossesLd(Ld,data)
	end


return lossesLd, lossLd

end
