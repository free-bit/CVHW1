# -*- coding: utf-8 -*-
import os
import numpy as np              #Matrices
import matplotlib.pyplot as plt #Plots
from sys import argv
from scipy import signal,misc   #Signal, Test Data
from threading import Thread, Lock
from PIL import Image
from operator import itemgetter

'''
def numpyImplementation(image2D, bin_size):
    gray_hist, bins=np.histogram(image2D, bin_size, range=(0,255))
    return (normalize(gray_hist), bins)
'''
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
'''
#Works fine
#Older implementation
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
    return (color_hist, bins)
'''

#CONSTANTS
CH=np.broadcast_to(np.array([1,0,-1]), (3,3))
CV=np.transpose(CH)
BH=np.broadcast_to(np.array([0,1,-1]), (3,3))
BV=np.transpose(BH)
FH=np.broadcast_to(np.array([-1,1,0]), (3,3))
FV=np.transpose(FH)
FILTERS={"centered": {"horizontal": CH, 
                      "vertical": CV},
         "backward": {"horizontal": BH, 
                      "vertical": BV},
         "forward": {"horizontal": FH, 
                     "vertical": FV}}

#Works fine
def normalize(data):
    return data/np.sum(data)

#Works fine
def grayscaleHistogram(image2D, bin_size):
    step=255/bin_size
    bins=[step*i for i in range(1, bin_size+1)]
    bins[-1]+=1
    image2D=np.sort(image2D, axis=None)
    indices = np.searchsorted(bins, image2D, side='right')
    gray_hist=np.zeros(bin_size)
    for j in indices:
        gray_hist[j]+=1
    return (normalize(gray_hist), bins)
    
def plotHistogram(x_vals, weights, keys, **kwargs):
    color="black"
    plot="bar"
    if "color" in kwargs:
        color=kwargs["color"] 
    if "plot" in kwargs:
        plot=kwargs["plot"]
    if(plot == "bar"):
        plt.hist(x=x_vals, weights=weights, bins=keys,
                 rwidth=0.75, alpha=0.7, color=color)
    elif(plot == "line"):
        plt.plot(x_vals, weights, color)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Intensity Histogram')
    plt.xlim(0, 255)
    plt.xticks(keys)
    #plt.show()

def colorHistogram(image2D, bin_size):
    step=255/bin_size
    bins=[step*i for i in range(1, bin_size+1)]
    bins[-1]+=1

    r_vec2D=image2D[:,:,0].ravel()
    r_indices = np.digitize(r_vec2D, bins)
    
    g_vec2D=image2D[:,:,1].ravel()
    g_indices = np.digitize(g_vec2D, bins)

    b_vec2D=image2D[:,:,2].ravel()
    b_indices = np.digitize(b_vec2D, bins)

    color_hist=np.zeros((bin_size, bin_size, bin_size))

    for i in range(len(r_vec2D)):
        color_hist[r_indices[i], g_indices[i], b_indices[i]]+=1
    return (normalize(color_hist), bins)

#Works fine in both gray and color images
def partitionImage(image2D, level):
    if(level==1):
        return [image2D]
    factor=2**(level-1)
    sub_height=image2D.shape[0]
    sub_width=image2D.shape[1]
    
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
        # print("Rows from:",row_start,"to:",row_end)
        rows=image2D[row_start:row_end]
        col_start=0
        col_end=sub_width
        col_remainder=remainder_width
        for j in iterations:
            if(col_remainder>0):
                col_end+=1
                col_remainder-=1
            # print("Cols from:",col_start,"to:",col_end)
            cols=rows[:,col_start:col_end]
            partitions.append(cols)
            col_start=col_end
            col_end+=sub_width
        row_start=row_end
        row_end+=sub_height
        # print("\n")
    return partitions

#TODO: check
def applyGradient(image2D, **kwargs):
    #Default values
    gradType="centered"
    orient="horizontal"
    #Apply preferences
    if("orient" in kwargs):
        orient=kwargs["orient"]
    if("gradType" in kwargs):
        gradType=kwargs["gradType"]
    #Construct filter    
    d_filter=FILTERS[gradType][orient]
    #Apply convolution
    return signal.convolve2d(image2D, d_filter, mode='same')

def findMagnitudesAndAngles(h_filtered, v_filtered):
    if h_filtered.ndim!=1:
        h_filtered=h_filtered.ravel()
    if v_filtered.ndim!=1:
        v_filtered=v_filtered.ravel()
    size=h_filtered.size
    magnitudes=np.empty(size)
    angles=np.empty(size)
    for i in range(size):
        magnitudes[i]=np.sqrt(v_filtered[i]**2+h_filtered[i]**2)
        angle=np.rad2deg(np.arctan2(v_filtered[i], h_filtered[i]))
        #angles[i][j]=angle if angle>=0 else angle+180
        angles[i]=(angle+180)%180
    return (magnitudes, angles)

def gradientHistogram(image2D, bin_size):
    h_filtered=applyGradient(image2D, gradType="centered", orient="horizontal")
    v_filtered=applyGradient(image2D, gradType="centered", orient="vertical")
    magnitudes, angles=findMagnitudesAndAngles(h_filtered, v_filtered)
    
    step=180/bin_size
    bins=[step*i for i in range(1, bin_size+1)]
    bins[-1]+=1

    indices = np.digitize(angles, bins)
    gray_hist=np.zeros(bin_size)
    for i in range(len(indices)):
        gray_hist[indices[i]]+=magnitudes[i]
    return (normalize(gray_hist), bins)
'''
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

#Calculate gradient histogram
def gradientHistogram(image2D, bin_size):#Use 9 bins
    h_filtered=applyGradient(image2D, gradType="centered", orient="horizontal")
    v_filtered=applyGradient(image2D, gradType="centered", orient="vertical")
    magnitudes, angles=findMagnitudesAndAngles(h_filtered, v_filtered)
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
'''
#Get euclidean distance between two feature vectors
def euclideanDistance(hist1, hist2):
    size1=hist1.size
    size2=hist2.size
    if(size1!=size2):
        return
    distance=np.sqrt(np.sum((hist1-hist2)**2))
    return distance

#Extract features using provided extraction function
def extractFeatures(image2D, ext_method, levels, bins):
    features=np.array([])
    #Partition the image first
    partitions=partitionImage(image2D, levels)
    #For each partition get features and append to the list
    for partition in partitions:
        hist,_=ext_method(partition, bins)
        features=np.append(features, hist)
    return features

#Read an image and return numpy array for color and grayscale
def readImage(path):
    image = Image.open(path)
    color=np.array(image.convert(mode='RGB'))#TODO: This might be changed
    gray=np.array(image.convert(mode='L'))
    return (color, gray)

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
    return (name+"_"+feature+ext)

def getDistanceFileName(query):
    index1=query.rfind(".")
    name="distances_from_"+query[:index1]+".txt"
    return name

def saveFeatureToFile(folder, file, feature, ext_mode):
    if(feature.ndim!=1):
        feature=feature.ravel()
    name=getFeatureFileName(file, ext_mode)
    np.savetxt(folder+name, feature)
        
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
'''
def insertKey(results, key):
    with Lock():
        results[key]={}

def insertResult(results, key1, key2, value):
    with Lock():
        results[key1][key2]=value
'''
#Thread routine for CBIR
def CBIRPipeline(files, queries, params):
    fflag=False
    qflag=False
    pipe_mode=params["pipe_mode"]
    ext_mode=params["ext_mode"]
    if(pipe_mode == "full"):
        fflag=True
        qflag=True
    elif(pipe_mode == "ext"):
        fflag=True
    elif(pipe_mode == "query" and queries):
        qflag=True

    #Extract features save results to specified folder
    if(fflag):
        print("Extracting features...")
        feature_folder=params["feature_wfolder"]
        dataset_folder=params["dataset_rfolder"]
        ext_func, image_type=EXT_FUNCS[ext_mode]
        processed=0
        one_fourth=int(len(files)/4)
        half=one_fourth*2
        three_fourth=one_fourth*3
        for file in files:
            #Select gray or color image (based on the method of choice)
            image2D=readImage(dataset_folder+file)[image_type]
            #Extract features of all files (using the method of choice)
            feature=extractFeatures(image2D, ext_func, params["level"], params["bins"])
            #Save extracted features
            saveFeatureToFile(feature_folder, file, feature, ext_mode)
            processed+=1
            if(processed==one_fourth):
                print("25% of features extracted. Total files processed: {}".format(processed))
            elif(processed==half):
                print("50% of features extracted. Total files processed: {}".format(processed))
            elif(processed==three_fourth):
                print("75% of features extracted. Total files processed: {}".format(processed))
        print("Extraction is completed.")
    #Query given files to get distances and save results to specified folder
    if(qflag):
        results={}
        feature_folder=params["feature_rfolder"]
        distance_folder=params["distance_wfolder"]
        processed=0
        one_fourth=int(len(queries)/4)
        half=one_fourth*2
        three_fourth=one_fourth*3
        for query in queries:
            print("Querying {} in the database...".format(query))
            #insertKey(results, query)
            results[query]={}
            query_feature_name=getFeatureFileName(query, ext_mode)#TODO: 0 is tmp
            q_feature=readFeaturesFromFile(feature_folder, query_feature_name)
            for file in files:
                feature_name=getFeatureFileName(file, ext_mode)#TODO: 0 is tmp
                o_feature=readFeaturesFromFile(feature_folder, feature_name)
                #Similarity test of query file with all other files
                distance=euclideanDistance(q_feature, o_feature)
                #insertResult(results, query, file, distance)
                results[query][file]=distance
            processed+=1
            if(processed==one_fourth):
                print("25% of queries processed. Total queries processed: {}".format(processed))
            elif(processed==half):
                print("50% of queries processed. Total queries processed: {}".format(processed))
            elif(processed==three_fourth):
                print("75% of queries processed. Total queries processed: {}".format(processed))
            #matches=getMatches(files, distances, 0.40)
            #print(matches)
        print("Querying is completed. Saving results to a file...")
        saveResultsToFile(distance_folder, results)

def printConfig(params):
#- thread_count: {}\n\
    print(
"Parameter configuration\n\
-------------------------\n\
- dataset_rfolder: {}\n\
- feature_rfolder: {}\n\
- feature_wfolder: {}\n\
- distance_wfolder: {}\n\
- image_db: {}\n\
- query_db: {}\n\
- pipe_mode: {}\n\
- ext_mode: {}\n\
- threshold: {}\n\
- level: {}\n\
- bins: {}\n\
-------------------------"
.format(#params["thread_count"],
        params["dataset_rfolder"],
        params["feature_rfolder"],
        params["feature_wfolder"],
        params["distance_wfolder"],
        params["image_db"],
        params["query_db"],
        params["pipe_mode"],
        params["ext_mode"],
        params["threshold"],
        params["level"],
        params["bins"]))

def help():
#--threads : thread count (1,2,3...)
    print('''
Possible flags:
--drfolder: dataset read location
--frfolder: feature read location
--fwfolder: feature write location
--rwfolder: distrance write location
--imagedb : file name of full image list
--querydb : file name of query image list
--pipemode: mode for pipeline execution (full, ext, query)
--extmode : mode for feature extraction (gray, color, grad)
--level   : image partitioning level (1,2,3...)
--bins    : histogram bins to be used (1,2,3...)
''')

def parseArgvSetParams(argv, params):
    '''
    #Get thread count (if provided)
    try:
        index=argv.index("--threads")        
        params["thread_count"]=int(argv[index+1])
    except ValueError:
        pass
    '''
    #Get dataset read location (if provided)
    try:
        index=argv.index("--drfolder")
        tmp=argv[index+1]
        params["dataset_rfolder"]=tmp if ('/' in tmp) else (tmp+'/')
    except ValueError:
        pass
    #Get feature read location (if provided)
    try:
        index=argv.index("--frfolder")
        tmp=argv[index+1]
        params["feature_rfolder"]=tmp if ('/' in tmp) else (tmp+'/')
    except ValueError:
        pass
    #Get full image list (if provided)
    try:
        index=argv.index("--imagedb")
        params["image_db"]=argv[index+1]
    except ValueError:
        pass
    #Get image list to be searched in dataset (if provided)
    try:
        index=argv.index("--querydb")
        params["query_db"]=argv[index+1]
    except ValueError:
        pass
    #Get mode for pipeline execution (if provided)
    try:
        index=argv.index("--pipemode")
        tmp=argv[index+1]
        params["pipe_mode"]=tmp if tmp in ("ext", "query", "full") else params["pipe_mode"]
    except ValueError:
        pass
    #Get mode for feature extraction (if provided)
    try:
        index=argv.index("--extmode")
        tmp=argv[index+1]
        params["ext_mode"]=tmp if tmp in EXT_FUNCS.keys() else params["ext_mode"]
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
    #Get feature write location (if provided)
    try:
        index=argv.index("--fwfolder")
        tmp=argv[index+1]
        params["feature_wfolder"]=tmp if ('/' in tmp) else (tmp+'/')
    except ValueError:
        params["feature_wfolder"]="features_{}_bin{}_level{}/".format(params["ext_mode"],
                                                                      params["bins"],
                                                                      params["level"])
    #Get distance write location (if provided)
    try:
        index=argv.index("--rwfolder")
        tmp=argv[index+1]
        params["distance_wfolder"]=tmp if ('/' in tmp) else (tmp+'/')
    except ValueError:
        params["distance_wfolder"]="distances_{}_bin{}_level{}/".format(params["ext_mode"],
                                                                        params["bins"],
                                                                        params["level"])
    printConfig(params)

#CONSTANTS
#Methods that can be used for extraction and image index (0 for color, 1 for gray) 
EXT_FUNCS={
    "gray": (eval("grayscaleHistogram"), 1),
    "color": (eval("colorHistogram"), 0),
    "grad": (eval("gradientHistogram"), 1)
}

def main(argv):
    help()
    #Example configuration, this config can be changes with flags above
    params={#"thread_count": 2,
            "dataset_rfolder": "dataset/",
            "feature_rfolder": "features/",
            "feature_wfolder": None,
            "distance_wfolder": "distances/",
            "image_db": "images.dat",
            "query_db": "validation_queries.dat",
            "pipe_mode": "query",
            "ext_mode": "color",
            "threshold": 0.4,
            "level": 1,
            "bins": 1}
    #Parse cmd args
    if(len(argv)):
        parseArgvSetParams(argv, params)
    else:
        print("WARNING: No flag is provided using the following preset configuration:\n")
        params["feature_wfolder"]="features_{}_bin{}_level{}/".format(params["ext_mode"],
                                                                      params["bins"],
                                                                      params["level"])
        params["distance_wfolder"]="distances_{}_bin{}_level{}/".format(params["ext_mode"],
                                                                        params["bins"],
                                                                        params["level"])
        printConfig(params)
    ans=input("Do you confirm these settings? [y/N]\n")
    if(ans!="y"):
        print("Script aborted.")
        return
    #Check write directories abort if they are not empty
    if(params["pipe_mode"] in ("ext", "full")):
        try:
            os.mkdir(params["feature_wfolder"])
        except FileExistsError:
            if(len(os.listdir(params["feature_wfolder"]))):
                print("WARNING: Aborting script not to overwrite existing data under '{}'!".format(params["feature_wfolder"]))
                return
    if(params["pipe_mode"] in ("query", "full")):
        try:
            os.mkdir(params["distance_wfolder"])
        except FileExistsError:
            if(len(os.listdir(params["distance_wfolder"]))):
                print("WARNING: Aborting script not to overwrite existing data under '{}'!".format(params["distance_wfolder"]))
                return
    #Read files in db from the dataset_rfolder
    files=readLinesFromFile(params["image_db"])
    queries=readLinesFromFile(params["query_db"])

    #TODO: For testing purposes remove later
    # files=["ahshLeADzK.jpg", "ALeTHlSSQi.jpg", "AocBBVgmHL.jpg", "FkFRyMVOPM.jpg", "SHllYUopJZ.jpg", "TQYtyQwlvQ.jpg"]
    # queries=["TQYtyQwlvQ.jpg", "ahshLeADzK.jpg"]
    #ENDTODO
    CBIRPipeline(files, queries, params)
    '''
    #Per thread file pool logic
    number_of_files=len(files)
    if(number_of_files<params["thread_count"]):
        params["thread_count"]=number_of_files
    step=int(number_of_files/params["thread_count"])
    remainder=int(number_of_files%params["thread_count"])
    file_pool=[]
    start=0
    for i in range(params["thread_count"]):
        end=start+step
        if(remainder):
            end+=1
            file_pool.append(files[start:end])
            remainder-=1
        else:
            file_pool.append(files[start:end])
        start=end
    #Per thread query pool logic
    number_of_queries=len(queries)
    step=int(number_of_queries/params["thread_count"])
    remainder=int(number_of_queries%params["thread_count"])
    query_pool=[]
    start=0
    for i in range(params["thread_count"]):
        end=start+step
        if(remainder):
            end+=1
            query_pool.append(queries[start:end])
            remainder-=1
        else:
            query_pool.append(queries[start:end])
        start=end
    print(query_pool)
    print("Under directory:{}, total number of images: {}".format(params["dataset_rfolder"], 
                                                                  len(files)))
    #Creating threads
    threads=[]
    results={}
    for i in range(params["thread_count"]):
        threads.append(Thread(target=CBIRPipeline, args=(file_pool[i], query_pool[i], params, results)))

    #Starting threads
    for th in threads:
        th.start()
    print("Running {} threads...".format(params["thread_count"]))

    #Joining threads
    for th in threads:
        th.join()
    print("All threads terminated")
    print("Saving results...")
    #Save distance
    if(params["pipe_mode"] in ("query", "full")):
        if(results):
            saveResultsToFile(params["distance_wfolder"], results)
        else:
            print("WARNING: No result to save!")
    '''
    
if __name__ == "__main__":
    main(argv[1:])
    