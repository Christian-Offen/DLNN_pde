using Flux
using ForwardDiff
using Dates	# for saving TW data
using JSON	# for saving TW data


# main functions are "find_travelling_wave" and "find_discrete_travelling_wave"

# find_travelling_wave
# works with continuous Lagrangian densities
# returns function train_tw! which updates cf and loss_hist
# cf = [guess for wave speed; guess for f(xi)=u(0,xi) l-periodic function]

# find_discrete_travelling_wave
# works with discrete Lagrangian densities
# returns function train_tw! which updates cf and loss_hist
# cf = [guess for wave speed; guess for f(xi)=u(0,xi) l-periodic function]





# check whether Euler-Lagrange equation is fulfilled (continuous Lagrangians)

function EL(L,uJet2)
    u,ut,ux,utt,utx,uxx = uJet2
    uJet1 = uJet2[1:3]
    
    HessL=ForwardDiff.hessian(L,uJet1)
    
    el_dt = HessL[2,1]*ut+HessL[2,2]*utt+HessL[2,3]*utx
    el_dx = HessL[3,1]*ux+HessL[3,2]*utx+HessL[3,3]*uxx
    el_p  = ForwardDiff.derivative(u->L([u,ut,ux]),u)
    
    return el_dt+el_dx-el_p
    
end


# objective for find_travelling_wave (continuous Lagrangians)

function tw_objectiveEL(L,cf)
    
        c = cf[1]
        f = cf[2:end]
        period= c*l    # period
        
        # spectral derivative
        # df,ddf = fft_Jet2(f,period)
        df,ddf = dft_Jet2(f,period) # no fft
    
        u = f
        ut = -c*df
        ux = df
        utt = c^2*ddf
        utx = -c*ddf
        uxx = ddf

        UJet2 = [u ut ux utt utx uxx]
        ELTest = ujet2 -> EL(L,ujet2)
        return mapslices(ELTest,UJet2; dims=2)
        
end


function find_travelling_wave(L,cf,loss_hist) # for continuous Lagrangians
    
    # returns function train_tw! which updates cf and loss_hist
    # cf = [guess for wave speed; guess for f(xi)=u(0,xi) l-periodic function]

    function loss0(cf)

        M = length(cf)-1
        dxi = l/M
        normsquare_f = (dxi*sum(cf[2:end].^2)-1)^2
        EL_consistency = sum(tw_objectiveEL(L,cf).^2) # MSE

        return [EL_consistency; normsquare_f]

    end

    loss(cf) = sum(loss0(cf))
    
    
    # select optimiser
    opt = ADAM()

    # define training functions

    function train_tw!()

        # compute gradient
        grads=ForwardDiff.gradient(loss,cf)

        # update parameters
        Flux.update!(opt, cf, grads)

    end

    function train_tw!(epochs::Int)

        for j = 1:epochs

            train_tw!();

            current_loss = loss0(cf)
            global loss_hist=hcat(loss_hist,current_loss)        
            println(current_loss)
            flush(stdout);

        end

    end
    
    return train_tw!

end




# discrete travelling wave 

function translate_fourier_coefficients(M)
# translation of fft and real coefficients needed in "find_discrete_travelling_wave"

    noRFreq = length(rfftfreq(M))  

    function f_to_frhatR(f)
        frhat = rfft(f);
        return frhatR = [real(frhat[1]); real(frhat[2:end]); imag(frhat[2:end])]
    end

    function frhatR_to_f(frhatR)
        frhat =[frhatR[1]; frhatR[2:noRFreq].+im*frhatR[1+noRFreq:end]]
        return irfft(frhat,M)
    end
    
    return f_to_frhatR, frhatR_to_f
    
end


function find_discrete_travelling_wave(L,cfrhatR,M) # for discrete Lagrangian L
    
    # returns function train_tw! which updates cfrhatR and global loss_hist_d_tw
    
    # M number of spatial grid points
    # cfrhatR = [guess for wave speed; f_to_frhatR(f)], where
    # f = guess for f(xi)=u(0,xi) l-periodic function
    
    # cfrhatR can be converted back to wave speed c and profile f as follows
    #c_tw_disc_opt = cfrhatR[1]
    #frhat_tw_disc = cfrhatR[2:end]
    #f_tw_disc_opt=frhatR_to_f(frhat_tw_disc);
    
    
    
    # loss function
    
    freq = fftfreq(M)*M
    noRFreq = length(rfftfreq(M))  

    # periodic spatial indices
    modInd(i::Int) = mod(i-1,M)+1


    function loss_cfhat(cfhat)

        c = cfhat[1]
        fhat = cfhat[2:end]

        u(t,x) = 1/M*transpose(fhat)*exp.(freq*2*pi*im/l*(x-c*t)) # spectral interpolation

        U = real(u.(TMesh,transpose(XMesh)))
        #DELloss  = sum(DELTest(L,U).^2)/(M*(N+1))

        DELErr = 0.
        for i=2:N
                for j=1:M
                    # u,uup,uleft,udown,uright,uupleft,udownright
                    DELErr += DiscreteEL(L,U[i,j],U[i-1,j],U[i,modInd(j-1)],U[i+1,modInd(j-1)],U[i+1,j],U[i,modInd(j+1)],U[i-1,modInd(j+1)])^2
                end
        end
        DELloss = DELErr/(M*(N+1))
        l2UU = sum(U.^2)/(M*(N+1))
        #normloss = (sum(U.^2)/(M*(N+1))-0.5)^2
        normloss = exp(-100*l2UU)      # punish trivial solution

        return [DELloss,normloss]

    end

    f_to_frhatR, frhatR_to_f = translate_fourier_coefficients(M)
    

    function lossR0(cfrhatR)

        c = cfrhatR[1]
        frhatR = cfrhatR[2:end]
        frhat = [frhatR[1]; frhatR[2:noRFreq].+im*frhatR[1+noRFreq:end]]
        fhat = [frhat;conj.(frhat[end-1:-1:2])]

        return loss_cfhat([c;fhat])
        
    end
    
    function lossR(cfrhatR)
        return sum(lossR0(cfrhatR))
    end
    
    
    # select optimiser
    opt = ADAM()

    # define training functions

    function train_tw!()

        # compute gradient
        grads=ForwardDiff.gradient(lossR,cfrhatR)

        # update parameters
        Flux.update!(opt,cfrhatR, grads)

    end

    function train_tw!(epochs::Int; print_every=1, save_every=NaN)

            for j = 1:epochs

                train_tw!();

                current_loss = lossR0(cfrhatR)
                global loss_hist_d_tw=hcat(loss_hist_d_tw,current_loss)        
                
                if mod(j,print_every)==0
	                println(current_loss)
        	        flush(stdout);
        	end
        	
        	if mod(j,save_every)==0
	                nowrun = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
			run_dict = Dict("time" => nowrun, "cfrhatR" => cfrhatR)
			open(nowrun*"run_TW.json","w") do f
			    JSON.print(f, run_dict)
	end

        	end

            end

    end 
    
    return train_tw!

end










