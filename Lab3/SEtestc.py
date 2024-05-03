import numpy, scipy
import sounddevice
import scipy.io.wavfile as wav
from scipy.io import wavfile
from scipy.io.wavfile import write
from matplotlib import pyplot as plot
import sys
import cmath

def SEtest(inputFile = "", cleanspeechFile = "", outfile = ""):
#         inputFile - noisy speech file in .wav format
#         cleanspeechFile - original clean speech file in .wav format
#         outFile - enhanced output file in .wav format
#
#  To Enhance the noisy speech :
#  Example call:  SEtest('noisy speech_babble.wav','speech.wav','out_log_babble.wav');
#  To listen the speech :
#  Example call: x = wavread('noisy speech_babble.wav');
#  Example call: sound(x,16000)
#  Example call: s = wavread('speech.wav');
#  Example call: sound(s,16000)
#  Example call: sh = wavread('out_babble.wav');
#  Example call: sound(sh,16000)
# ========== Check the input parameters of function ================
# nargin: number of input arguments in function
#####Start
    if (inputFile == "" or cleanspeechFile == "" or outfile == ""):
        #if number of parameters are not enough (empty) display error message
        print("Error: the function should call as SEtest(noisyfile.wav,outFile.wav) \n\n")
    else:
        #enough parameters are given, start the job
        Srate, x = wav.read(inputFile)
        Srate, s = wav.read(cleanspeechFile)

        x = numpy.divide(x,32767) #in Python, you need to x by 32767 to make the magnitude as same as in MATLAB
        s = numpy.divide(s,32767)
        # Returning the sampled data in x and s, Time domain
        # Sample rate (Srate) in Hertz
        # The number of bits per sample (bits) used to encode the data in the file.
        length = numpy.floor(32 * Srate / 1000) # Frame size in samples
        length = int(length) #convert back to int
        if (numpy.remainder(length,2) == 1): #adjust length to a even number
            length += 1
        PERC = 75 # window overlap in percent of frame size
        len1 = numpy.floor(length * PERC / 100)
        len1 = int(len1)
        len2 = length - len1
        win = numpy.hamming(length)
        nFFT = length * 2
        #Draw the spectrogram for clean speech
        plot.figure(3)
        pxx,freq, t, cax = plot.specgram(s, pad_to= length, noverlap= len1, NFFT= nFFT, Fs= Srate)
        plot.colorbar(cax)
        plot.title("Original clean speech")
        #Draw the spectrogram for noisy speech
        plot.figure(4)
        pxx, freq, t, cax = plot.specgram(x, pad_to=length, noverlap=len1, NFFT=nFFT, Fs=Srate)
        plot.colorbar(cax)
        plot.title("Noisy speech")

        #Noise power estimations: using the first 12 frames to estimate
        noise_power = numpy.zeros(nFFT)
        j = length * 5

        for m in range(12): #first 12 frames
            na = numpy.multiply(win,x[j:j + length])

            na = numpy.fft.fft(na,nFFT)
            noise_amplitude = numpy.abs(na)
            #noise_power = numpy.add(noise_power,numpy.square(noise_amplitude))
            noise_power = noise_power + numpy.square(noise_amplitude)
            j += length

        noise_power = numpy.divide(noise_power, 12)

        Nframes = numpy.subtract(numpy.floor(numpy.divide(numpy.size(x), len2)), numpy.floor(numpy.divide(length,len2)))

        Nframes = int(Nframes)

        xfinal = numpy.zeros(numpy.size(x))

################ Algorithm of speech enhancement ##########################
        k = 0
        img = complex(0,1) # 0 + 1j
        for n in range(0,Nframes):
            insigc = numpy.multiply(win, s[k:k + length]) #input signal, clean
            #print(insigc)
            specc = numpy.fft.fft(insigc,nFFT)
            sigc = numpy.abs(specc) #compute the magnitude

            insign = numpy.multiply(win, x[k:k + length]) #input signal, noisy
            spec = numpy.fft.fft(insign, nFFT)
            sig = numpy.abs(spec) #compute the magnitude

#===========================================================================
# Your enhancement algorithm to each frame should start here
#
# noise_power => The power spectral density of noise
# sig => magnitude of noisy speech frame
# sigc => magnitude of clean speech frame

# find pxx, note that Puu = noise power
            pss = sigc**2
            Pss_Puu = pss+noise_power
            
            sig = (pss/Pss_Puu)*sig
            

#===============================================#

            xi_w = numpy.fft.ifft(numpy.multiply(sig,numpy.exp(numpy.multiply(img,numpy.angle(spec)))))
            xi_w = numpy.real(xi_w)
            xfinal[k:k+length] = xfinal[k:k+length] + xi_w[0:length]
            k = numpy.add(k,len2)
#####End
        xfinal = numpy.multiply(xfinal,((100-PERC)/50))
        wav.write(outfile,Srate,xfinal)

        plot.figure(7)
        pxx, freq, t, cax = plot.specgram(xfinal, pad_to=length, noverlap=len1, NFFT=nFFT, Fs=Srate)
        plot.colorbar(cax)
        plot.title("Enhanced speech")
        plot.show()