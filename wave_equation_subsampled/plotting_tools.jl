using Plots
using LaTeXStrings

# function for plotting

function InstantiatePlotFun(dt,dx)

	function plotU(U)
	    n = size(U)[1]-1
	    M = size(U,2)
	    TMesh = 0:dt:n*dt
	    XMeshbd = 0:dx:M*dx # repeat boundary
	    UPlot = [U U[:,1]] # add repeated boundary for plotting
	    return plot(XMeshbd,TMesh,UPlot,st=:surface,xlabel=L"x",ylabel=L"t",legend=:none)
	end

	function contourU(U)
	    n = size(U)[1]-1
    	    M = size(U,2)
	    TMesh = 0:dt:n*dt
	    XMeshbd = 0:dx:M*dx # repeat boundary
	    UPlot = [U U[:,1]] # add repeated boundary for plotting
	    return contour(XMeshbd,TMesh,UPlot,xlabel=L"x",ylabel=L"t",legend=:none)
	end
	
	
	
	function contourU!(U)
	    n = size(U)[1]-1
    	    M = size(U,2)
	    TMesh = 0:dt:n*dt
	    XMeshbd = 0:dx:M*dx # repeat boundary
	    UPlot = [U U[:,1]] # add repeated boundary for plotting
	    return contour!(XMeshbd,TMesh,UPlot,xlabel=L"x",ylabel=L"t",legend=:none)
	end

	return plotU, contourU,contourU!
	
end
