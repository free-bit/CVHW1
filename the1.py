# -*- coding: utf-8 -*-
import numpy as np               #Matrices
from scipy import signal, misc   #Signal, Test Data
import matplotlib.pyplot as plt  #Plots

def normalize(data):
    return data/np.linalg.norm(data, ord=1)

def getHistogram(data, number_of_bins=10, low=0, high=255):
    freqs, bins=np.histogram(data, bins=number_of_bins, 
                                       range=(low,high))
    return (normalize(freqs), bins)

def getCoordinates(bins):
    x=[]
    for i in range(len(bins)-1):
        x.append((bins[i]+bins[i+1])/2)
    return x
    
def plotHistogram(x_vals, weights, keys):
    n, bins, patches = plt.hist(x=x_vals, weights=weights, bins=keys,
                                rwidth=0.75, alpha=0.7, color='#0504aa')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Intensity Histogram')
    plt.xlim(0, 255)
    plt.xticks(keys)
    plt.show()
    return (n, bins, patches)

def main():
    gray_face = misc.face(gray=True)
    color_face = misc.face()
    r=color_face[:,:,0].ravel()
    g=color_face[:,:,1].ravel()
    b=color_face[:,:,2].ravel()
    freqs, bins=getHistogram(gray_face)
    x_vals=getCoordinates(bins)
    plotHistogram(x_vals, freqs, bins)
    
    
if __name__ == "__main__":
    main()
    