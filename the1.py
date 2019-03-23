# -*- coding: utf-8 -*-
import os
import numpy as np              #Matrices
import matplotlib.pyplot as plt #Plots
from sys import argv
from scipy import signal,misc   #Signal, Test Data
from threading import Thread
from PIL import Image
from operator import itemgetter

#Works fine
def normalize(data):
    return data/np.linalg.norm(data, ord=1)

def numpyImplementation(image2D, bin_size):
    gray_hist, bins=np.histogram(image2D, bin_size, range=(0,255))
    return (normalize(gray_hist), bins)

#Works fine
def grayscaleHistogram(image2D, bin_size):
    step=255/bin_size
    bins=[step*i for i in range(bin_size+1)]
    image2D=np.sort(image2D, axis=None)
    indices = np.searchsorted(bins, image2D, side='right')
    gray_hist=np.zeros(bin_size)
    for j in indices:
        gray_hist[j-1]+=1
    return (normalize(gray_hist), bins)
    
'''
#Older implementation
#Works fine
def grayscaleHistogram(image2D, bin_size):
    step=255/bin_size
    bins=[step*i for i in range(bin_size+1)]
    shape=image2D.shape
    row_size=range(shape[0])
    col_size=range(shape[1])
    gray_hist=np.zeros(bin_size)
    for i in row_size:
        row=image2D[i]
        for j in col_size:
          index=int(row[j]/step)
          if(index==bin_size):
              index-=1
          gray_hist[index]+=1
    return (normalize(gray_hist), bins)
'''

def getCoordinates(bins):
    x=[]
    for i in range(len(bins)-1):
        x.append((bins[i]+bins[i+1])/2)
    return x
    
def plotHistogram(x_vals, weights, keys, **kwargs):
    color="black"
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

def newColorHistogram(image2D, bin_size):
    step=255/bin_size
    bins=[step*i for i in range(bin_size+1)]

    r_vec2D=image2D[:,:,0].ravel()
    r_indices = np.digitize(r_vec2D, bins)
    
    g_vec2D=image2D[:,:,1].ravel()
    g_indices = np.digitize(g_vec2D, bins)

    b_vec2D=image2D[:,:,2].ravel()
    b_indices = np.digitize(b_vec2D, bins)

    color_hist=np.zeros((bin_size, bin_size, bin_size))

    for i in range(len(r_vec2D)):
        color_hist[r_indices[i]-1, g_indices[i]-1, b_indices[i]-1]+=1
    return color_hist
#TODO: Needs check and performance improvement
def colorHistogram(image2D, bin_size):
    r_vec2D=image2D[:,:,0]
    g_vec2D=image2D[:,:,1]
    b_vec2D=image2D[:,:,2]
    step=255/bin_size
    color_hist=np.zeros((bin_size, bin_size, bin_size))
    shape=r_vec2D.shape
    row_size=range(shape[0])
    col_size=range(shape[1])
    for i in row_size:
        r_row=r_vec2D[i]
        g_row=g_vec2D[i]
        b_row=b_vec2D[i]
        for j in col_size:
          r=int(r_row[j]/step)
          if(r==bin_size):
              r-=1
          g=int(g_row[j]/step)
          if(g==bin_size):
              g-=1
          b=int(b_row[j]/step)
          if(b==bin_size):
              b-=1
          color_hist[r,g,b]+=1
    return color_hist

#Works fine in both gray and color images
def partitionImage(image2D, level):
    if(level==1):
        return [image2D]
    factor=2**(level-1)
    sub_height, sub_width,_=image2D.shape
    
    remainder_height=sub_height%factor
    sub_height=int(sub_height/factor)
    
    remainder_width=sub_width%factor
    sub_width=int(sub_width/factor)
    
    iterations=range(factor)
    partitions=[]
    row_start=0
    row_end=sub_height
    row_remainder=remainder_height
    for i in iterations:
        if(row_remainder>0):
            row_end+=1
            row_remainder-=1
        #print("Rows from:",row_start,"to:",row_end)
        rows=image2D[row_start:row_end]
        col_start=0
        col_end=sub_width
        col_remainder=remainder_width
        for j in iterations:
            if(col_remainder>0):
                col_end+=1
                col_remainder-=1
            #print("Cols from:",col_start,"to:",col_end)
            cols=rows[:,col_start:col_end]
            partitions.append(cols)
            col_start=col_end
            col_end+=sub_width
        row_start=row_end
        row_end+=sub_height
        #print("\n")
    return partitions
#TODO: check
def applyGradient(image2D, **kwargs):
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
    return signal.convolve2d(image2D, d_filter, mode='same')
#TODO: check
def findMagnitudesAndAngles(h_filtered, v_filtered):
    shape=h_filtered.shape
    magnitudes=np.empty(shape)
    angles=np.empty(shape)
    row_size=range(shape[0])
    col_size=range(shape[1])
    for i in row_size:
        for j in col_size:
            magnitudes[i][j]=np.sqrt(v_filtered[i][j]**2+h_filtered[i][j]**2)
            angle=np.rad2deg(np.arctan2(v_filtered[i][j], h_filtered[i][j]))
            #angles[i][j]=angle if angle>=0 else angle+180
            angles[i][j]=(angle+180)%180
    return (magnitudes, angles)

#TODO: Normalization???
def gradientHistogram(magnitudes, angles, bin_size=9):
    step=180/bin_size
    bins=[step*i for i in range(bin_size+1)]
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
    return (normalize(grad_hist), bins)
#Works fine
def euclideanDistance(hist1, hist2):
    size1=hist1.size
    size2=hist2.size
    if(size1!=size2):
        return
    distance=np.sqrt(np.sum((hist1-hist2)**2))
    return distance

#TODO: Write a file I/O handler to save features right after extraction
#TODO: File read to read color and grayscale images properly

#TODO: Complete
def extractFeatures(image2D, method, levels, bins):
    features=np.array([])
    #Partition the image first
    partitions=partitionImage(image2D, levels)
    #Pick a method
    if(method=="grayscale"):
        for partition in partitions:
            #TODO: replace numpy with grayscaleHistogram later
            gray_hist,_=numpyImplementation(partition, bins)#1xlen(bin)
            features=np.append(features, gray_hist)
        
    elif(method=="color3D"):
        for partition in partitions:
            color_hist=colorHistogram(partition, bins)
            features=np.append(features, color_hist)
            
    elif(method=="gradient"):
        for partition in partitions:
            grad_hist,_=gradientHistogram(partition, bins)#1xlen(bin)
            features=np.append(features, grad_hist)
    return features

#Read an image and return numpy array for color and grayscale
def readImage(path):
    image = Image.open(path)
    color=np.array(image.convert(mode='RGB'))#TODO: This might be changed
    grayscale=np.array(image.convert(mode='L'))
    return (color, grayscale)

def readFeaturesFromFile(folder, file):
    return np.loadtxt(folder+file)

def readDistancesFromFile(folder, file):
    return np.loadtxt(folder+file)

def readLinesFromFile(file):
    with open(file,"r") as file:
        lines=[]
        for line in file:
            if(line[-1]=='\n'):
                line=line[:-1]
            lines.append(line)
        return lines

def getFeatureFileName(filename, feature):
    index=filename.rfind(".")
    name=filename[:index]
    ext=".txt"
    return (name+str(feature)+ext)

def getDistanceFileName(query):
    index1=query.rfind(".")
    name="distances_from_"+query[:index1]+".txt"
    return name

def saveFeaturesToFile(folder, file, features):
    count=0
    for feature in features:
        name=getFeatureFileName(file, count)
        np.savetxt(folder+name, feature)
        count+=1
        
def saveResultsToFile(folder, results):
    name="results.txt"
    with open(folder+name,"+w") as file:
        for query in results:
            items=results[query].items()
            items=sorted(items, key=itemgetter(1))
            line=query+":"
            for name, distance in items:
                line+=("{} {} ").format(distance, name)
            line=line[:-1]+"\n"
            file.write(line)

def getMatches(others, distances, threshold):
    indices=np.where(distances < threshold)[0]
    matches=[others[i] for i in indices]
    print(matches)
        
#Thread routine for CBIR
def CBIRPipeline(files, params):
    fflag=False
    qflag=False
    feature_folder=params["feature_folder"]
    query_file=params["query_db"]
    if(params["mode"] is "full"):
        fflag=True
        qflag=True
    elif(params["mode"] is "fext"):
        fflag=True
    elif(params["mode"] is "query" and query_file):
        qflag=True
    if(fflag):
        print("Extracting features...")
        dataset_folder=params["dataset_folder"]
        for file in files:
            color, gray=readImage(dataset_folder+file)
            #Extract features of all files in the list
            features=[]
            features.append(extractFeatures(gray, "grayscale", params["level"], params["bins"]))
            #features.append(extractFeatures(color, "color3D", params["level"], params["bins"]))
            #features.append(extractFeatures(gray, "gradient", params["level"], params["bins"]))
            #Save extracted features
            saveFeaturesToFile(feature_folder, file, features)
    if(qflag):
        queries=readLinesFromFile(query_file)[:1]#TODO: change later
        results={}
        for query in queries:
            print("Querying {} in the database...".format(query))
            results[query]={}
            distance_folder=params["distance_folder"]
            query_feature_name=getFeatureFileName(query, 0)#TODO: 0 is tmp
            q_feature=readFeaturesFromFile(feature_folder, query_feature_name)
            for file in files:
                feature_name=getFeatureFileName(file, 0)#TODO: 0 is tmp
                o_feature=readFeaturesFromFile(feature_folder, feature_name)
                #Similarity test of query file with all other files
                results[query][file]=euclideanDistance(q_feature, o_feature)
            #matches=getMatches(files, distances, 0.40)
            #print(matches)
        #Save distance
        saveResultsToFile(distance_folder, results)
    
def parseArgvSetParams(argv, params):
    #Get thread count (if provided)
    try:
        index=argv.index("--threads")        
        params["thread_count"]=int(argv[index+1])
    except ValueError:
        pass
    #Get dataset location (if provided)
    try:
        index=argv.index("--folder")
        tmp=argv[index+1]
        params["dataset_folder"]=tmp if ('/' in tmp) else (tmp+'/')
    except ValueError:
        pass
    #Get image partitioning level (if provided)
    try:
        index=argv.index("--level")
        params["level"]=int(argv[index+1])
    except ValueError:
        pass
    #Get histogram bins to be used (if provided)
    try:
        index=argv.index("--bins")
        params["bins"]=int(argv[index+1])
    except ValueError:
        pass
    #Get mode for pipeline execution (if provided)
    try:
        index=argv.index("--mode")
        tmp=argv[index+1]
        params["mode"]=tmp if tmp in ("fext", "query", "full") else params["mode"]
    except ValueError:
        pass
    #Get file to be searched in dataset (if provided)
    try:
        index=argv.index("--query")
        params["query_db"]=argv[index+1]
    except ValueError:
        pass
    print(
"Parameter configuration\n\
-------------------------\n\
thread_count: {}\n\
dataset_folder: {}\n\
feature_folder: {}\n\
distance_folder: {}\n\
image_db: {}\n\
query_db: {}\n\
threshold: {}\n\
mode: {}\n\
level: {}\n\
bins: {}\n\
-------------------------"
.format(params["thread_count"],
        params["dataset_folder"],
        params["feature_folder"],
        params["distance_folder"],
        params["image_db"],
        params["query_db"],
        params["threshold"],
        params["mode"],
        params["level"],
        params["bins"]))
'''
Possible flags:
   --threads int
   --folder string
   --query string
   --level int
   --bins int
   --mode string
'''
def main(argv):
    params={"thread_count": 1,
            "dataset_folder": "subdataset/",
            "feature_folder": "features/",
            "distance_folder": "distances/",
            "image_db": "images.dat",
            "query_db": "validation_queries.dat",
            "threshold": 0.4,
            "mode": "query",
            "level": 1,
            "bins": 10}
    parseArgvSetParams(argv, params)
    #Read files under folder
    files=readLinesFromFile(params["image_db"])
    if(params["mode"] is "fext"):
        try:
            os.mkdir(params["feature_folder"])
        except FileExistsError:
            if(len(os.listdir(params["feature_folder"]))):
                print("WARNING: Aborting script not to overwrite existing data under '{}'!".format(params["feature_folder"]))
                return
    if(params["mode"] is "query"):
        try:
            os.mkdir(params["distance_folder"])
        except FileExistsError:
            if(len(os.listdir(params["distance_folder"]))):
                print("WARNING: Aborting script not to overwrite existing data under '{}'!".format(params["distance_folder"]))
                return
    #Per thread file pool logic
    number_of_files=len(files)
    if(number_of_files<params["thread_count"]):
        params["thread_count"]=number_of_files
    step=int(number_of_files/params["thread_count"])
    remainder=int(number_of_files%params["thread_count"])
    pool=[]
    start=0
    for i in range(params["thread_count"]):
        end=start+step
        if(remainder):
            end+=1
            pool.append(files[start:end])
            remainder-=1
        else:
            pool.append(files[start:end])
        start=end
    print("Under directory:{}, total number of images: {}".format(params["dataset_folder"], 
                                                                  len(files)))
    threads=[]
    for i in range(params["thread_count"]):
        threads.append(Thread(target=CBIRPipeline, args=(pool[i], params,)))
    for th in threads:
        th.start()
    print("Running {} threads...".format(params["thread_count"]))
    for th in threads:
        th.join()
    print("All threads terminated")
    
if __name__ == "__main__":
    main(argv[1:])
    