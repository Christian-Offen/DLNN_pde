# returns NN model for Ld and random initialisation vector
# 1 hidden layers 

function InitLd(rng; NNwidth=10, sigma = softplus)

    # Nwidth = size of hidden layer
    
    # initial parameters
    A1 = randn(rng,(NNwidth,3))
    A2 = randn(rng,(NNwidth,NNwidth))
    A3 = randn(rng,(1,NNwidth))

    b1 = randn(rng,NNwidth)
    b2 = randn(rng,NNwidth)

    paramsVec=[A1[:];A2[:];A3[:];b1[:];b2[:];];


    function LdLearn(paramVec,x0::Vector)

            A1 = reshape(paramVec[1:NNwidth*3],(NNwidth,3))
            offset = NNwidth*3

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


            x1 = sigma.(A1*x0 + b1)
            x2 = sigma.(A2*x1 + b2)
            x3 = A3*x2

            return x3[1]

        end

        function LdLearn(paramVec,u,uup,uright)
            return LdLearn(paramVec,[u uup uright]);
        end

return LdLearn, paramsVec

end

