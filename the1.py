# -*- coding: utf-8 -*-
import numpy as np                        #Matrices
from scipy import signal,misc   #Signal, Test Data
import matplotlib.pyplot as plt          #Plots

def normalize(data):
    return data/np.linalg.norm(data, ord=1)

def grayscaleHistogram(data, bin_size=10, low=0, high=255):
    freqs, bins=np.histogram(data, bins=bin_size, range=(low,high))
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
    color_hist=np.zeros((bin_size, bin_size, bin_size))
    shape=r_vec.shape
    row_size=range(shape[0])
    col_size=range(shape[1])
    for i in row_size:
        for j in col_size:
          r=int(r_vec[i][j]/step)
          if(r==bin_size):
              r-=1
          g=int(g_vec[i][j]/step)
          if(g==bin_size):
              g-=1
          b=int(b_vec[i][j]/step)
          if(b==bin_size):
              b-=1
          color_hist[r,g,b]+=1
    return color_hist

def partitionImage(image, level):
    if(level==1):
        return [image]
    factor=2**(level-1)
    sub_height, sub_width,_=image.shape
    
    remainder_height=sub_height%factor
    sub_height=int(sub_height/factor)
    
    remainder_width=sub_width%factor
    sub_width=int(sub_width/factor)
    
    iterations=range(factor)
    partitions=[]
    for i in iterations:
        row_start=i*sub_height
        row_end=row_start+sub_height
        if(remainder_height>0):
            row_end+=1
            remainder_height-=1
        rows=image[row_start:row_end]
        for j in iterations:                
            col_start=j*sub_width
            col_end=col_start+sub_width
            if(remainder_width>0):
                col_end+=1
                remainder_width-=1
            cols=rows[:,col_start:col_end]
            partitions.append(cols)  
    return partitions

def applyGradient(image, **kwargs):
    filters={"centered": {"horizontal": np.array([1,0,-1]), 
                          "vertical": np.array([1,0,-1]).reshape(3,1)},
             "backward": {"horizontal": np.array([0,1,-1]), 
                          "vertical": np.array([0,1,-1]).reshape(3,1)},
             "forward": {"horizontal": np.array([-1,1,0]), 
                         "vertical": np.array([-1,1,0]).reshape(3,1)}}
    #Default values
    gradType="centered"
    orient="horizontal"
    #Apply preferences
    if("orient" in kwargs):
        orient=kwargs["orient"]
    if("gradType" in kwargs):
        gradType=kwargs["gradType"]
    #Construct filter    
    d_filter=filters[gradType][orient]   
    d_filter=np.broadcast_to(d_filter, (3, 3))
    #Apply convolution
    return signal.convolve2d(image, d_filter, mode='same')

def findMagnitudesAndAngles(h_filtered, v_filtered):
    shape=h_filtered.shape
    magnitudes=np.empty(shape)
    angles=np.empty(shape)
    row_size=range(shape[0])
    col_size=range(shape[1])
    for i in row_size:
        for j in col_size:
            magnitudes[i][j]=np.sqrt(v_filtered[i][j]**2+h_filtered[i][j]**2)
            #TODO: Check angle interval
            angles[i][j]=np.rad2deg(np.arctan2(v_filtered[i][j], h_filtered[i][j]))
    return (magnitudes, angles)

def gradientHistogram(magnitudes, angles, bin_size=9):
    step=180/bin_size
    shape=magnitudes.shape
    row_size=range(shape[0])
    col_size=range(shape[1])
    grad_hist=np.zeros(bin_size)
    for i in row_size:
        for j in col_size:
          magnitude=magnitudes[i][j]
          index=int(angles[i][j]/step)
          if(index==bin_size):
              index-=1
          grad_hist[index]+=magnitude
    return grad_hist
    


def test1():#gray hist
    gray_face = misc.face(gray=True)
    color_face = misc.face()
    r=color_face[:,:,0].ravel()
    g=color_face[:,:,1].ravel()
    b=color_face[:,:,2].ravel()
    freqs, bins=grayscaleHistogram(gray_face)
    x_vals=getCoordinates(bins)
    plotHistogram(x_vals, freqs, bins)
    
def test2():#plot hist
    image=plt.imread("test.jpg")
    r=image[:,:,0]
    g=image[:,:,1]
    b=image[:,:,2]
    freqs1, bins1=grayscaleHistogram(r.ravel(), 4)
    freqs2, bins2=grayscaleHistogram(g.ravel(), 4)
    freqs3, bins3=grayscaleHistogram(b.ravel(), 4)
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
    r=image[:,:,0]
    g=image[:,:,1]
    b=image[:,:,2]
    color_hist=colorHistogram(r, g, b, 4)
    print(color_hist)
    
def test4():#partition
    image=plt.imread("test.jpg")
    partitions=partitionImage(image, 2)
    for part in partitions:
        plt.imshow(part)
        plt.show()
        
def test5():#gradient
    image = misc.face(gray=True)
    h_filtered=applyGradient(image, gradType="centered", orient="horizontal")
    v_filtered=applyGradient(image, gradType="centered", orient="vertical")
    #plt.imshow(h_filtered)
    #plt.imshow(v_filtered)
    magnitudes, angles=findMagnitudesAndAngles(h_filtered, v_filtered)
    plt.imshow(magnitudes)
        
def main():
    test3()
    
if __name__ == "__main__":
    main()
    