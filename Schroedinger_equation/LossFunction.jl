
function InstantiateLdLoss(rng)

	## regularisation

	# create matrix used in Newton iterations in forward problem
	function CondForwardProblem(Ld,U0,U1,U2)
	    
	    dim, M = size(U0)
	    modInd(i::Int) = mod(i-1,M)+1
	    modInd2(i::Int) = mod(i-1,2*M)+1
	    
	    D1(u,uup,uright,uupright) = ForwardDiff.gradient(u->Ld(u,uup,uright,uupright),u)
	    D3(u,uup,uright,uupright) = ForwardDiff.gradient(uright->Ld(u,uup,uright,uupright),uright)
	    
	    D12(u,uup,uright,uupright) = ForwardDiff.jacobian(uup->D1(u,uup,uright,uupright),uup)
	    D14(u,uup,uright,uupright) = ForwardDiff.jacobian(uupright->D1(u,uup,uright,uupright),uupright)
	    
	    D32(u,uup,uright,uupright) = ForwardDiff.jacobian(uup->D3(u,uup,uright,uupright),uup)
	    D34(u,uup,uright,uupright) = ForwardDiff.jacobian(uupright->D3(u,uup,uright,uupright),uupright)
	    
	    
	    function CreateRow(k::Int)
		
		uupleft = U2[:,modInd(k-1)]
		uup  = U2[:,k]
		uupright  = U2[:,modInd(k+1)]
		uleft  = U1[:,modInd(k-1)]
		u  = U1[:,k]
		uright = U1[:,modInd(k+1)]
		udownleft  = U0[:,modInd(k-1)]
		udown  = U0[:,k]
		udownright  = U0[:,modInd(k+1)]

		Mright  = D14(u,uup,uright,uupright)
		Mcentre = D12(u,uup,uright,uupright) + D34(uleft,uupleft,u,uup)
		Mleft   = D32(uleft,uupleft,u,uright)

		if k ==1
		    return [Mcentre Mright zeros(2,2*M-6) Mleft]
		
		elseif k== M
		    return [Mright zeros(2,2*M-6) Mleft Mcentre]
		    
		else
		    return [zeros(2,2*k-4) Mleft Mcentre Mright zeros(2,2*M-2k-2)]
		end
	    end
	    
	    # no pre-allocation possible due to ForwardDiff
	    CondMat = reduce(vcat,[CreateRow(k) for k=1:M]) # don't forget to multiply condMat by dx. This wouldn't change condskeel, though, and is, therefore, left out
	    
	    return CondMat #, abs(det(CondMat)) #use sparsity (todo) # Alternative: condskeel(CondMat)           
	end

	# for definition of invIterM2
	iterStart = randn(rng,2*M)
	iterStart = iterStart ./ norm(iterStart)

	function invIterM2(M,TOL)
	    # smallest eigenvalue of M^T*M
	    
	    MChol = cholesky(transpose(M)*M, check=false)
	    vNrm = iterStart
	    lambda0 = Inf
	    lambda1 = 0.
	    
	    while abs(lambda1-lambda0)>TOL
		lambda0 = lambda1
		v = vNrm
		v = MChol\v
		vNrm = v ./ norm(v)
		lambda1 = 1/(transpose(vNrm)*v)
	    end
	    
	    return lambda1
	    
	end

	function invIterM2Steps(M,steps)
	    # smallest eigenvalue of M^T*M
	    
	    MChol = cholesky(transpose(M)*M, check=false)
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


	function RegulariserSolvability(Ld::Function,DataBlock)
	    
	    # dataBlock = output of Flux.DataLoader

	    # Only when using SKEEL condition number
	    #CondNumberRelax(cond) = Flux.softplus(cond-40)  # map cond between 0 and Inf
	    
	    # using determinants
	    #CondNumberRelax(absdet) = Flux.sigmoid(-log(absdet))  # map cond between 0 and Inf
	    
	    function CondNumberRelax(CondMat)
		
		relax(s) = relu(-10*abs(s)+1) # punish smallest eigenvalue if its norm is smaller 0.1; max. punishment = 1
		steps = 3
		smallestEig = invIterM2Steps(CondMat,steps)
		return relax(smallestEig)                     # square of spectral norm of M
	    end
	    
	    condRelaxed = 0.

	    for U123 in DataBlock
		CondMat = CondForwardProblem(Ld,U123[:,:,1],U123[:,:,2],U123[:,:,3]);
		condRelaxed = condRelaxed + CondNumberRelax(CondMat)
	    end

	    condRelaxed = condRelaxed/length(DataBlock)

	end

	function lossesLd(Ld,DataBlock)
	    
	    loss_data = sum((U->DataConsistency(Ld,U) ).(DataBlock))
	    loss_reg = RegulariserSolvability(Ld,DataBlock)
	    
	    return [loss_data, loss_reg]

	end

	return lossesLd

end
