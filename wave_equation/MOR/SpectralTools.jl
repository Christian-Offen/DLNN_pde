using FFTW

# spectral derivatives

function fft_Jet2(f,period)
    # spectral derivative
    
        M = length(f)
        freq = fftfreq(M)*M
    
        fftf=fft(f)        
        df = real.(ifft( 2*pi*im/period * freq.* fftf ))
        ddf = real.(ifft( (2*pi*im/period)^2 * freq.^2 .* fftf ))
    return df,ddf
end

# version without fft for the convenience of AD tools

function dft_matrix(N::Int)
    
      k = 0:(N-1)
      n = transpose(0:(N-1))
      DFT_matrix = exp.(-2im * π * k * n / N)

    return DFT_matrix
end

function dft_Jet2(f,period)
    
    # spectral derivative (no fft)
    
        M = length(f)
        freq = fftfreq(M)*M
    
        dft_mat = dft_matrix(length(f))
        fftf = dft_mat*f
        df = real.(dft_mat\( 2*pi*im/period * freq.* fftf ))
        ddf = real.(dft_mat\( (2*pi*im/period)^2 * freq.^2 .* fftf ))
    
     return df,ddf
end

# spectral interpolation

function SpectralInterpolation_dft(coeff,x,period)
    M = length(coeff)
    freq = fftfreq(M)*M
    return 1/M*transpose(coeff)*exp.(freq*2*pi*im/period*x)
end

function SpectralInterpolation(f,x,period)
    return SpectralInterpolation_dft(fft(f),x,period)
end
