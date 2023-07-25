# returns NN model for Ld and random initialisation vector
# 2 hidden layers (same size)

function InitLd(rng; inputWidth=2, NNwidth=10, sigma = softplus)

    # Nwidth = size of hidden layer
    
    # initial parameters
    A1 = randn(rng,(NNwidth,inputWidth))
    A2 = randn(rng,(NNwidth,NNwidth))
    A3 = randn(rng,(1,NNwidth))

    b1 = randn(rng,NNwidth)
    b2 = randn(rng,NNwidth)

    paramsVec=[A1[:];A2[:];A3[:];b1[:];b2[:]];


    function LdLearn(paramVec,x0::Vector)

            A1 = reshape(paramVec[1:NNwidth*inputWidth],(NNwidth,inputWidth))
            offset = NNwidth*inputWidth
            sz = NNwidth*NNwidth
            A2 = reshape(paramVec[offset+1:offset+sz],(NNwidth,NNwidth))
            offset = offset+sz
            sz = 1*NNwidth
            A3 = reshape(paramVec[offset+1:offset+sz],(1,NNwidth))

	    offset = offset+sz
	    sz = NNwidth	    
            b1 = paramVec[offset+1:offset+sz]
            offset = offset+sz
	    sz = NNwidth	    
            b2 = paramVec[offset+1:offset+sz]
            offset = offset+sz

            x1 = sigma.(A1*x0 + b1)
            x2 = sigma.(A2*x1 + b2)
            x3 = A3*x2

            return x3[1]

        end

return LdLearn, paramsVec

end

