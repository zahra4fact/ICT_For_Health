import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, sobel
import pandas as pd

#read the image
import glob
from imageio import imwrite

path = "F:\ICT\ICT for health\lab\ICT for health_LAB\Mole\image"
files = glob.glob(path+'/*.jpg')

for f in files:
    im = mpl.image.imread(f)
    
    #determin the best number of cluster
#    for k in np. arange (1 ,5):
#        model = KMeans ( n_clusters =k, random_state =0)
#        labels = model . fit_predict ( gaussian_filter (im , sigma= (10 , 10, 0)). reshape (( im. shape [0]* im. shape [1] ,3)))
#        ineria . append ( model . inertia_ )
#    
#    plt . plot (np. arange (1 ,5) ,ineria ,'ro -')
#    plt . xlabel (" Number Of Cluster ")
#    plt . ylabel ('Inertia ')
#    plt . show()

    # enhance the image and do the clustering, then reshape
    model = KMeans(n_clusters=4, random_state=0)
    labels = model.fit_predict(gaussian_filter(im, sigma = (10, 10, 0)).reshape((im.shape[0]*im.shape[1], 3)))
    #find the label that corresponds to the Mole, find the center with the darkest color
    centers = model.cluster_centers_
    darkest_label = centers.sum(axis = 1).argmin()
    
    #segment the image
    segmented = labels.reshape((im.shape[0], im.shape[1]))
    segmented[segmented != darkest_label] = 0
    
    #find the edges
    edge = sobel(segmented)
    edge = (edge<0)+(edge>0)
    
   #calculate the premiter of circle and ratio toward to premiter of area
    premiter = edge.sum()
    area = segmented.sum()
    r2 = area/np.pi
    r = np.sqrt(r2)
    prem_circle = 2*r*np.pi
    ratio=premiter/prem_circle
    print(ratio)
    #draw the borders
    im.setflags(write=1)
    im[edge,0] = 255
    
    fname = path+'\\segmented\\'+f.split('\\')[-1]
    print(fname)
    imwrite(fname, im)
