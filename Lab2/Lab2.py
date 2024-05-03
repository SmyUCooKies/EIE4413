import numpy as np # import numpy library to use FFT and arrays
from numpy import random  # import random from numpy to use random functions
from matplotlib import pyplot as plt

def part1a():
    w = np.hamming(73)
    plt.plot(w)
    return w
    
def part1b():
    wT1 = 2*(2665/16000)
    wT2 = 2*(5335/16000)
    
    x = np.arange(-36,37)
    sinc1 = np.multiply(np.sinc(x*wT1), wT1)
    plt.plot(sinc1)
    plt.title('Sinc of 2665Hz')
    plt.show()
    
    sinc2 = np.multiply(np.sinc(x*wT2), wT2)
    plt.plot(sinc2)
    plt.title('Sinc of 5335Hz')
    plt.show()
    
    return sinc1, sinc2

def part1c(a, b, window):
    finalFilter = abs(np.fft.fft(b*window - a*window, 16000))
    
    plt.plot(finalFilter)
    plt.title('Magnitude response of the bandpass filter')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()


def part2a(n):
    #%% <Bandpass filter>
# Part1a
    window = np.hamming(n)
# Part1b
    deltaF = 750/2
    wT1 = 2*( (3040-deltaF)/16000 )
    wT2 = 2*( (4960+deltaF)/16000 )
    
    x = np.arange(-np.floor(n/2), np.ceil(n/2))
    sinc1 = np.multiply(np.sinc(x*wT1), wT1)
    sinc2 = np.multiply(np.sinc(x*wT2), wT2)
# Part1c
    band_sinc = sinc2*window - sinc1*window
    bandpass = abs(np.fft.fft(band_sinc, 16000))
    
    plt.plot(bandpass)
    plt.title('Magnitude response of all 3 filters')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    #%% <Lowpass filter>
    low_wT = 2*( (2290+deltaF)/16000 )
    low_sinc = np.multiply(np.sinc(x*low_wT), low_wT) * window
    lowpass = abs(np.fft.fft(low_sinc, 16000))

    plt.plot(lowpass)
    #%% <Highpass filter>
    high_wT = 2*( (5710-deltaF)/16000 )
    high_sinc = np.multiply(np.sinc(x*high_wT), high_wT)
    highpass = 1 - abs(np.fft.fft(high_sinc*window, 16000))

    plt.plot(highpass)
#%%
def part2b():
    import scipy.io 
    import sounddevice
    from scipy.io import wavfile
    from scipy.io.wavfile import write

    fs, data = wavfile.read('./music16k.wav')
    sounddevice.play(data,fs, blocking=False)
    #%% <Bandpass filter>
def part2c(n):
# Part1a
    window = np.hamming(n)
# Part1b
    deltaF = 750/2
    wT1 = 2*( (3040-deltaF)/16000 )
    wT2 = 2*( (4960+deltaF)/16000 )
    
    x = np.arange(-np.floor(n/2), np.ceil(n/2))
    sinc1 = np.multiply(np.sinc(x*wT1), wT1)
    sinc2 = np.multiply(np.sinc(x*wT2), wT2)
# Part1c
    bpf = sinc2*window - sinc1*window
    #%% <Lowpass filter>
    low_wT = 2*( (2290+deltaF)/16000 )
    lpf = np.multiply(np.sinc(x*low_wT), low_wT) * window
    #%% <Highpass filter>
    k = int(np.floor(n/2))
    high_wT = 2*( (5710-deltaF)/16000 )
    hpf = (np.multiply(np.sinc(x*high_wT), -high_wT))
    hpf[k] = 1 - high_wT
    hpf = hpf * window
    #%% Filtering
    import scipy.io
    import sounddevice
    from scipy.io import wavfile
    from scipy.io.wavfile import write   
    
    fs, data = wavfile.read('./music16k.wav')	
    # fs is the sampling frequency
    data = np.divide(data, 32767)	       
    # data read by wavefile.read need to be divided by 32767
    # Source: https://stackoverflow.com/questions/39316087/how-to-read-a-audio-file-in-python-similar-to-matlab-audioread
    
    # Assume the designed low pass, band pass, and high pass filters are
    #     lpf, bpf, and hpf, respectively. Change them with your names
    # Note that data has two channels, since the music is stereo
    low = 0.05
    band = 0.5
    high = 1.5
    
    musicL1 = np.convolve(lpf, data[:, 0])
    musicL2 = np.convolve(lpf, data[:, 1])
    musicM1 = np.convolve(bpf, data[:, 0])
    musicM2 = np.convolve(bpf, data[:, 1])
    musicH1 = np.convolve(hpf, data[:, 0])
    musicH2 = np.convolve(hpf, data[:, 1])
    music1 = np.add(np.add(musicL1*low, musicM1*band), musicH1*high)
    music2 = np.add(np.add(musicL2*low, musicM2*band), musicH2*high)
    
    # music1 = np.add(musicL1*low, musicM1*band, musicH1*high)
    # music2 = np.add(musicL2*low, musicM2*band, musicH2*high)
    
    music = np.zeros((np.size(music1),2))
    music[:, 0] = music1
    music[:, 1] = music2
    
    sounddevice.play(music, fs, blocking=True)
#%%
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   # window = part1a()
    # sinc1, sinc2 = part1b()
    # part1c(sinc1, sinc2, window)
    # part2a(73)
    part2b()
    # part2c(73)