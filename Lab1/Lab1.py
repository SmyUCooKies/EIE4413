import numpy                # To use FFT and arrays
from numpy import random
import time                 # To use time() function for time computation.
from matplotlib import pyplot as plot
from sklearn.metrics import mean_squared_error

def part1a(n):
    x = random.rand(65536)  # Define a random vector x with 65536 elements
    y = random.rand(65213)  # Define a random vector y with 65213 elements
    
    xt = 0
    for i in range(n):
        tic = time.time()  # Count start time
        x1 = numpy.fft.fft(x)
        toc = time.time()  # Count finish time
        xt += toc - tic

    print("N is a power of 2:", xt/n) # Output the average FFT computation time

    yt = 0
    for i in range(n):
        tic = time.time()  # Count start time
        y1 = numpy.fft.fft(y)
        toc = time.time()  # Count finish time
        yt += toc - tic

    print("N is not a power of 2:", yt/n) # Output the average FFT computation time
    
def part1b():
    x = [1,1,1,1]
    N_values = [4, 8, 32, 128, 1024]
    
    # Plot the magnitude of X for N = 4, 8, 32, 128, 1024
    plot.subplots(5, 1, figsize=(8, 20))
    for i, N in enumerate(N_values):
        plot.subplot(5, 1, i+1)
        X = numpy.fft.fft(x, N)
        plot.stem(abs(X))
        plot.xlim([0, N-1])
        plot.ylim([0, 4.5])
        plot.title(f'FFT of x with N = {N}')
        plot.ylabel('Magnitude')
        plot.subplots_adjust(hspace=0.5)  # Add spacing between plots
    plot.xlabel('Frequency (k)')

    # Find the frequency response at 900Hz with sampling frequency of 12800Hz
    X = numpy.fft.fft(x, 128)
    print("Using sampling frequency of 12800Hz, the frequency response at 900Hz is when k = 9.")
    print("The frequency response X[9] =", X[9], "\n")
    
    plot.show()

def part2a():
    x = [1, 2, 3, 4, 5, 6, 5, 4, 3]
    h = [1, 4, 6, 4, 1]
    y = numpy.convolve(x, h)    # Direct convolution result
    Y = numpy.fft.fft(y,16)     # Compute 16-point FFT of y
    
    X = numpy.fft.fft(x,16)	    # Compute 16-point FFT of x
    H = numpy.fft.fft(h,16)     # Compute 16-point FFT of h
    Y2 = numpy.multiply(X, H)   # Point-by-point multiplication
    
    diff = mean_squared_error(abs(Y), abs(Y2))
    print("Mean squared error = ", diff)
    
    print("Direct convolution: ", y)
    print("FFT: ", numpy.real(numpy.fft.ifft(Y2)))

def part2b(n):
    x = random.rand(102500)
    h = random.rand(1024)
    
    x1 = 0
    for i in range(n):
        tic = time.time()  # Count start time
        y = numpy.convolve(x,h)
        toc = time.time()  # Count finish time
    
    x2 = 0
    for i in range(n):
        t1 = time.time()
        X = numpy.fft.fft(x,131072)
        H = numpy.fft.fft(h,131072)
        Y2 = numpy.multiply(X,H)
        y2 = numpy.fft.ifft(Y2)
        t2 = time.time()
        x2 += t2 - t1
        x1 += toc - tic
    print("\nPart2b\nAverage computation time for Direct Convolution: ", x1/n)
        
    diff = mean_squared_error(abs(y), abs(numpy.real(y2[0:numpy.size(y)])))
    print("Average computation time for FFT approach: ", x2/n)
    print("Mean squared error: ", diff)    

def part2c(n):
    x = random.rand(102500)
    h = random.rand(1024)
    
    # Direct Conv
    time_conv = 0
    for i in range(n):
        t1 = time.time()
        y = numpy.convolve(x, h)	# Direct convolution result
        t2 = time.time()
        time_conv += t2 - t1
    print("\nPart2c\nAverage computation time for the Direct convolution approach: ", time_conv/n, "\n")

    # Overlap-and-add with FFT
    time_fft = 0
    for i in range(n):
        tic = time.time()
        y2 = numpy.zeros(102500 + 1024 - 1)
        H = numpy.fft.fft(h, 2048)
    
        for i in range(100):
            si = i * 1025
            segment = x[si:si+1025]
            segment_fft = numpy.fft.fft(segment,2048) # FFT of segment_x (freq)
            segment_conv = numpy.multiply(segment_fft,H) # p-by-p multiply (freq)
            y2[si:si+2048] += numpy.real(numpy.fft.ifft(segment_conv))
    
        toc = time.time()
        time_fft += toc - tic

    print("Average computation time for the overlap-and-add with FFT approach: ", time_fft/n, "\n" )
    diff = mean_squared_error(abs(y), abs(y2))
    print("Mean squared error: ",diff)


if __name__ == '__main__':
    # part1a(30)
    # part1b()
    # part2a()
    # part2b(30)
    # part2c(30)
    1