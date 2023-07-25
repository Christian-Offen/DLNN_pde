# returns NN model for Ld and random initialisation vector
# 2 hidden layers (same size)

function InitLd(rng; NNwidth=10, sigma = softplus)

    # Nwidth = size of hidden layer
    
    # initial parameters
    A1 = randn(rng,(NNwidth,8))
    A2 = randn(rng,(NNwidth,NNwidth))
    A3 = randn(rng,(NNwidth,NNwidth))
    A4 = randn(rng,(1,NNwidth))

    b1 = randn(rng,NNwidth)
    b2 = randn(rng,NNwidth)
    b3 = randn(rng,NNwidth)

    paramsVec=[A1[:];A2[:];A3[:];A4[:];b1[:];b2[:];b3[:]];


    function LdLearn(paramVec,x0::Vector)

            A1 = reshape(paramVec[1:NNwidth*8],(NNwidth,8))
            offset = NNwidth*8
            sz = NNwidth*NNwidth
            A2 = reshape(paramVec[offset+1:offset+sz],(NNwidth,NNwidth))
            offset = offset+sz
            A3 = reshape(paramVec[offset+1:offset+sz],(NNwidth,NNwidth))
            offset = offset+sz
            sz = 1*NNwidth
            A4 = reshape(paramVec[offset+1:offset+sz],(1,NNwidth))

	    offset = offset+sz
	    sz = NNwidth	    
            b1 = paramVec[offset+1:offset+sz]
            offset = offset+sz
	    sz = NNwidth	    
            b2 = paramVec[offset+1:offset+sz]
            offset = offset+sz
	    sz = NNwidth	    
            b3 = paramVec[offset+1:offset+sz]


            x1 = sigma.(A1*x0 + b1)
            x2 = sigma.(A2*x1 + b2)
            x3 = sigma.(A3*x2 + b3)
            x4 = A4*x3

            return x4[1]

        end

        function LdLearn(paramVec,UStencil::Matrix)
            if size(UStencil,2) !== 4; throw("Stencil of incorrect shape in LdLearn"); end
            return LdLearn(paramVec,reshape(UStencil,8));
        end

        function LdLearn(paramVec,u,uup,uright,uupright)
            return LdLearn(paramVec,[u uup uright uupright]);
        end

return LdLearn, paramsVec

end

