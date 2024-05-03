def part1a():
    import sounddevice
    import scipy.io.wavfile as wav
    from scipy.io.wavfile import write
    
    fs, white =     wav.read("white.wav")
    fs, pink = wav.read("pink.wav")
    fs, f16 = wav.read("f16.wav")
    fs, babble = wav.read("babble.wav")
    
    print("\nPlaying white noise ...")
    sounddevice.play(white, fs)
    command = "n"
    while command == "n":
        command = input("Play next (pink noise)? y/n: ")
        if command == "y":
            sounddevice.stop()
            
    print("\nPlaying pink noise ...")
    sounddevice.play(pink, fs)
    command = "n"
    while command == "n":
        command = input("Play next (f16 noise)? y/n: ")
        if command == "y":
            sounddevice.stop()
            
    print("\nPlaying f16 noise ...")
    sounddevice.play(f16, fs)
    command = "n"
    while command == "n":
        command = input("Play next (babble noise)? y/n: ")
        if command == "y":
            sounddevice.stop()
            
    print("\nPlaying babble noise ...")
    sounddevice.play(babble, fs)
    command = "n"
    while command == "n":
        command = input("Stop? y/n: ")
        if command == "y":
            sounddevice.stop()
    sounddevice.stop()

def part1b():
    import numpy
    import scipy.io.wavfile as wav
    import matplotlib.pyplot as plot
    
    fs, white = wav.read("white.wav")
    fs, pink = wav.read("pink.wav")
    
    # Extract the first 1024 data and normalize it
    w = white[0:1024]/32767
    p = pink[0:1024]/32767
    
    wCorr = numpy.correlate(w, w, mode='full')/len(w)
    pCorr = numpy.correlate(p, p, mode='full')/len(p)
    
    plot.plot(wCorr)
    plot.plot(pCorr)
    plot.legend(["white","pink"])
    plot.show()

def part1c():
    import numpy as np
    import matplotlib.pyplot as plot
    import scipy.io.wavfile as wav

    fw, musik = wav.read("music16k.wav")
    fp, babble = wav.read("babble.wav")

    m1 = musik[:, 0]
    m1 = m1[10001:11025]/32767
    m2 = musik[:, 1]
    m2 = m2[10001:11025]/32767
    babble = babble[10001:11025]/32767

    # cross correlation coefficients should be calculated manually from basic cross-correlation
    def cross_coef(x,y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)
   
        correlation = np.correlate(x, y, mode="full")/(len(x)) # Cross-Correlation \phi
        coef = (correlation - mean_x*mean_y) / (std_x*std_y) # Normalized - Correlation Coefficient (covariance)
        return coef
    x1 = cross_coef(m1, m2)         # x1 is the covariance between the two channels of the music
    x2 = cross_coef(m1, babble)     # x2 is the covariance between channel 1 and the babble noise
    lags = np.arange(-len(m1)+1, len(m1)) # Shifting the cross-correlation index to center
    
    plot.figure(figsize=(10, 5))
    plot.plot(lags, x1)	
    plot.plot(lags, x2)	
    plot.legend(["x1","x2"])
    plot.xlabel('Lag')
    plot.ylabel('Correlation Coefficient')
    plot.grid(True)
    plot.show()

def part1d():
    import numpy as np
    import matplotlib.pyplot as plot
    import scipy.io.wavfile as wav
    
    fw, musik = wav.read("music16k.wav")
    fs, white = wav.read('white.wav')	
    white = white[10001:11025]/14129.972
    
    m1 = musik[:, 0]
    m1 = m1[10001:11025]/32767
    m2 = musik[:, 1]
    m2 = m2[10001:11025]/32767
    m2_noise = m2 + white
    print(np.std(white)/np.std(m2))
    
    # cross correlation coefficients should be calculated manually from basic cross-correlation
    def cross_coef(x,y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)
   
        correlation = np.correlate(x, y, mode="full")/(len(x)) # Cross-Correlation \phi
        coef = (correlation - mean_x*mean_y) / (std_x*std_y) # Normalized - Correlation Coefficient (covariance)
        return coef
    x1 = cross_coef(m1, m2)         # x1 is the covariance between the two channels of the music
    x2 = cross_coef(m1, m2_noise)   # x2 is the covariance between channel 1 and channel 2 with added babble noise
    lags = np.arange(-len(m1)+1, len(m1)) # Shifting the cross-correlation index to center
    
    plot.plot(lags, x1)	
    plot.plot(lags, x2)	
    plot.legend(["x1","x2"])
    plot.xlabel('Lag')
    plot.ylabel('Correlation Coefficient')
    plot.grid(True)
    plot.show()

def part2a():
    import numpy
    import sounddevice
    import scipy.io.wavfile as wav
    import SEtest
    
    Srate, n = wav.read("white.wav") #Load the noise file
    Srate, s = wav.read("TIMITTrainF_02_01r.WAV") #Load the speech file
    
    x = numpy.add(s,n[0:numpy.size(s)]) #Add noise to sppech
    
    #Store the noisy speech to a file
    wav.write("noisy speech_white.wav",16000,x)
    
    SEtest.SEtest("noisy speech_white.wav","TIMITTrainF_02_01r.WAV","out.wav")
    
    fe, es = wav.read("out.wav")
    sounddevice.play(es,fe,blocking="true")
    
def part2b():
    import numpy
    import sounddevice
    import scipy.io.wavfile as wav
    import SEtestb
    
    Srate, n = wav.read("white.wav") #Load the noise file
    Srate, s = wav.read("TIMITTrainF_02_01r.WAV") #Load the speech file
    
    x = numpy.add(s,n[0:numpy.size(s)]) #Add noise to sppech
    
    #Store the noisy speech to a file
    wav.write("noisy speech_white.wav",16000,x)
    
    SEtestb.SEtest("noisy speech_white.wav","TIMITTrainF_02_01r.WAV","out.wav")
    
    fe, es = wav.read("out.wav")
    sounddevice.play(es,fe,blocking="true")
          
def part2c():
    import numpy
    import sounddevice
    import scipy.io.wavfile as wav
    import SEtestc
    
    Srate, n = wav.read("white.wav") #Load the noise file
    Srate, s = wav.read("TIMITTrainF_02_01r.WAV") #Load the speech file
    
    x = numpy.add(s,n[0:numpy.size(s)]) #Add noise to sppech
    
    #Store the noisy speech to a file
    wav.write("noisy speech_white.wav",16000,x)
    
    SEtestc.SEtest("noisy speech_white.wav","TIMITTrainF_02_01r.WAV","out.wav")
    
    fe, es = wav.read("out.wav")
    sounddevice.play(es,fe,blocking="true")
    
    
if __name__ == "__main__":
    # part1a()
    # part1b()
    # part1c()
    # part1d()
    # part2a()
    # part2b()
    # part2c()
    True
    