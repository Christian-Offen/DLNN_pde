function TrainingFunctions(LdLearn,lossesLd,loss_gradient,opt)

	# define training functions

	function train_single_batch!(data,paramsVec)
	    
	    grads=loss_gradient(paramsVec,data)
	    Flux.update!(opt, paramsVec, grads)
	    
	end


	function train_batched!(Data,paramsVec)
	    
	    for data in Data
		train_single_batch!(data,paramsVec)
	    end
	    
	end

	function train_batched!(Data,paramsVec,epochs::Int; print_every=1, save_every=Inf)
	    
	    for k = 1:epochs
		
		train_batched!(Data,paramsVec)
		
		if mod(k,print_every)==0
			LdLearnInstance(u,uup,uright,uupright) = LdLearn(paramsVec,u,uup,uright,uupright)
            		println(lossesLd(LdLearnInstance,Data.data))
            		flush(stdout);
	        end
        	if mod(k,save_every)==0
	           save_params(paramsVec,k)
        	end
		
		
	    end
	    
	end



	function save_params(paramsVec,epoch)

		    nowrun = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

		    run_dict = Dict("time" => nowrun, "learned_parameters" => paramsVec)

		    open(string(epoch)*"_"*nowrun*"run_param_data.json","w") do f
			JSON.print(f, run_dict)
		    end
	end

	function save_TrainingData(training_dataU)

			nowrun = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
			run_dict = Dict("training_data" => training_dataU)

			open("Schroedinger_"*nowrun*"training_data.json","w") do f
			    JSON.print(f, run_dict)
			end

	end

	return train_batched!, save_params, save_TrainingData
	
end
