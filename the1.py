# -*- coding: utf-8 -*-
import numpy as np                        #Matrices
from scipy import signal,misc   #Signal, Test Data
import matplotlib.pyplot as plt          #Plots

def normalize(data):
    return data/np.linalg.norm(data, ord=1)

def getHistogram(data, bin_size=10, low=0, high=255):
    freqs, bins=np.histogram(data, bins=bin_size, 
                                       range=(low,high))
    return (normalize(freqs), bins)

def getCoordinates(bins):
    x=[]
    for i in range(len(bins)-1):
        x.append((bins[i]+bins[i+1])/2)
    return x
    
def plotHistogram(x_vals, weights, keys, **kwargs):
    color="0504aa"
    plot="bar"
    if "color" in kwargs:
        color=kwargs["color"] 
    if "plot" in kwargs:
        plot=kwargs["plot"]
    if(plot is "bar"):
        plt.hist(x=x_vals, weights=weights, bins=keys,
                     rwidth=0.75, alpha=0.7, color=color)
    elif(plot is "line"):
        plt.plot(x_vals, weights, color)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Intensity Histogram')
    plt.xlim(0, 255)
    plt.xticks(keys)
    #plt.show()

def colorHistogram(r_vec, g_vec, b_vec, bin_size):
    step=255/bin_size
    iterations=range(len(r_vec))
    color_hist=np.zeros((bin_size, bin_size, bin_size))
    print(color_hist.shape)
    for i in iterations:
      r=int(r_vec[i]/step)
      if(r==bin_size):
          r-=1
      g=int(g_vec[i]/step)
      if(g==bin_size):
          g-=1
      b=int(b_vec[i]/step)
      if(b==bin_size):
          b-=1
      color_hist[r,g,b]+=1
    return color_hist

def partitionImage(image, level):
    if(level==1):
        return [image]
    factor=2**(level-1)
    sub_height, sub_width,_=image.shape
    sub_height=int(sub_height/factor)
    sub_width=int(sub_width/factor)
    iterations=range(factor)
    partitions=[]
    for i in iterations:
        for j in iterations:
            partitions.append(
                    image[i*sub_height:(i+1)*sub_height, 
                          j*sub_width:(j+1)*sub_width])  
    return partitions

def test1():#gray hist
    gray_face = misc.face(gray=True)
    color_face = misc.face()
    r=color_face[:,:,0].ravel()
    g=color_face[:,:,1].ravel()
    b=color_face[:,:,2].ravel()
    freqs, bins=getHistogram(gray_face)
    x_vals=getCoordinates(bins)
    plotHistogram(x_vals, freqs, bins)
    
def test2():#plot hist
    image=plt.imread("test.jpg")
    r=image[:,:,0].ravel()
    g=image[:,:,1].ravel()
    b=image[:,:,2].ravel()
    freqs1, bins1=getHistogram(r, 4)
    freqs2, bins2=getHistogram(g, 4)
    freqs3, bins3=getHistogram(b, 4)
    x_vals1=getCoordinates(bins1)
    x_vals2=getCoordinates(bins2)
    x_vals3=getCoordinates(bins3)
    plotHistogram(x_vals1, freqs1, bins1, color="red", plot="line")
    plotHistogram(x_vals2, freqs2, bins2, color="green", plot="line")
    plotHistogram(x_vals3, freqs3, bins3, color="blue", plot="line")
    plt.xticks([])
    plt.show()
    
def test3():#color hist
    image=plt.imread("test.jpg")
    r=image[:,:,0].ravel()
    g=image[:,:,1].ravel()
    b=image[:,:,2].ravel()
    color_hist=colorHistogram(r, g, b, 4)
    
def test4():#partition
    image=plt.imread("test.jpg")
    partitions=partitionImage(image, 1)
    for part in partitions:
        plt.imshow(part)
        plt.show()
        
def main():
    test4()
    
if __name__ == "__main__":
    main()
    