# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:34:51 2017

@author: Dindin Meryll
"""

from imports import *

# Defines a way to load the scans relative to a patient
def loadScans(patient) :
    
    slices = [dicom.read_file('./Input/{}/{}'.format(patient,s)) for s in os.listdir('./Input/{}'.format(patient))]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

# Get the images relative to a patient, along the z axis
def getImages(patient):
    
    scans = loadScans(patient)
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    image[image == -2000] = 0
    
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

# Display function for each patient
def displayScan(patient, slices, colormap) :
    
    image = getImages(patient)
    
    plt.figure(figsize=(18,len(slices)*4))
    
    for i,nSlice in enumerate(slices) :
        plt.subplot(len(slices), 2, 2*i+1)
        plt.imshow(cv2.resize(image[nSlice], (256,256)), cmap=colormap)
        plt.subplot(len(slices), 2, 2*i+2)
        plt.hist(image[nSlice].flatten(), bins=100, color='orange')
        plt.xlim([-1000, 1000])
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
    
    plt.tight_layout
    plt.show

# Thresholder tool
def tresh(x, down, up) :

    if x < down or x > up : return 0
    else : return x

# Binarizer tool 
def binarize(x) :
    
    if x == True : return 1
    else : return 0

# Defines the binarized sobol image out of a thresholded image
def getSobolImage(img) :
    
    thresh = np.vectorize(tresh)
    binar = np.vectorize(binarize)
    
    dx = ndimage.sobel(img - thresh(img, 0, 1200), 1) 
    dy = ndimage.sobel(img - thresh(img, 0, 1200), 0)
    sImage = np.hypot(dx, dy)  
    sImage *= 255.0 / np.max(sImage) 
    g1 = threshold_otsu(sImage)
    sImage = sImage > g1
    sImage = binar(sImage)
    
    return sImage  

# Defines the binarized laplacian image out of a thresholded image
def getLaplacianImage(img) :
    
    binar = np.vectorize(binarize)
    thresh = np.vectorize(tresh)
    
    lImage = thresh(img, -1200, -800)
    g2 = threshold_otsu(lImage)
    lImage = lImage > g2
    lImage = binar(lImage)
    
    return lImage

# Observe both the obtained laplacian and sobol images
def cutHist(img, nSlice, nColum) :
    
    thresh = np.vectorize(tresh)
    
    sImage = getSobolImage(img)
    lImage = getLaplacianImage(img)
    
    plt.figure(figsize=(18,8))
    
    plt.subplot(2,3,1)
    plt.xlim([-10,522])
    plt.bar(range(len(sImage[nSlice])), list(sImage[nSlice]), width=0.1)
    
    plt.subplot(2,3,3)
    plt.xlim([-10,522])
    plt.bar(range(len(sImage[:,nColum])), list(sImage[:,nColum]), width=0.1)
    
    plt.subplot(2,3,2)
    sFake = copy.copy(sImage)
    sFake[nSlice] = 1
    sFake[:,nColum] = 1
    plt.imshow(sFake, cmap="Greys")
    
    plt.subplot(2,3,4)
    plt.xlim([-10,522])
    plt.bar(range(len(lImage[nSlice])), list(lImage[nSlice]), width=0.1)
    
    plt.subplot(2,3,6)
    plt.xlim([-10,522])
    plt.bar(range(len(lImage[:,nColum])), list(lImage[:,nColum]), width=0.1)
    
    plt.subplot(2,3,5)
    lFake = copy.copy(lImage)
    lFake[nSlice] = 1
    lFake[:,nColum] = 1
    plt.imshow(lFake, cmap="Greys")
    
    plt.tight_layout
    plt.show()

# Extracts the nZones largest zones out of a list
def getLargest(l, nZones) :
        
    if len(l) <= nZones : return l
    else :
        v = [len(e) for e in l]
        w = list(np.argsort(v))
    
    return list(np.asarray(l)[w[-nZones:]])

# Extracts the zones according a given direction   
def extractZones(l, direction) :
    
    zones, u = [], []
    u = []
    
    for i, v in enumerate(l[::direction]) :
        if v == 1 and direction == 1 : u.append(i)
        elif v == 1 and direction == -1 : u.append(len(l)-(i+1))
        else :
            if len(u) == 0 : pass
            else :
                zones.append(u)
                u = []

    # Memory efficiency
    del u

    return zones

# Extracts the largest zone according to a direction
def getZones(img, ind, typ, direction) :
    
    if typ == "row" : return extractZones(img[ind], direction)
    elif typ == "col" : return extractZones(img[:, ind], direction)
    else : print("|-> Wrong type !")  

# Defines the appropriate mask of pixel to remove
def getRmMatrix(img) :
    
    mat = np.matrix([[1 for k in range(len(img))] for k in range(len(img))])
    
    img = getLaplacianImage(img)
    
    for nCol in range(len(img)) :
        zonesForw = getZones(img, nCol, 'col', 1)
        zonesBack = getZones(img, nCol, 'col', -1)
        for zones in [zonesForw, zonesBack] :
            if len(zones) == 0 : pass
            elif len(zones) == 1 : 
                for ind in zones[0] : mat[ind, nCol] = 0
            elif len(zones) == 2 : 
                for zone in zones : 
                    for ind in zone : mat[ind, nCol] = 0                  
            else :
                zones = getLargest(zones, 2)
                for zone in zones : 
                    for ind in zone : mat[ind, nCol] = 0
                    
    for nRow in range(len(img)) :
        zonesForw = getZones(img, nRow, 'row', 1)
        zonesBack = getZones(img, nRow, 'row', -1)
        for zones in [zonesForw, zonesBack] :
            if len(zones) == 0 : pass
            elif len(zones) == 1 : 
                for ind in zones[0] : mat[nRow, ind] = 0
            elif len(zones) == 2 : 
                for zone in zones : 
                    for ind in zone : mat[nRow, ind] = 0
            else :
                zones = getLargest(zones, 3)
                for zone in zones : 
                    for ind in zone : mat[nRow, ind] = 0
    
    binar = np.vectorize(binarize)
    
    return np.asarray(binar(mat))*img

# Display that same extraction
def observeExtraction(img) :
    
    plt.figure(figsize=(18,10))
    
    plt.subplot(2,3,1)
    plt.imshow(img,cmap=plt.cm.bone)
    plt.subplot(2,3,2)
    plt.imshow(getLaplacianImage(img), cmap='Greys')
    plt.subplot(2,3,3)
    plt.imshow(getSobolImage(img), cmap='Greys')
    plt.subplot(2,3,4)
    plt.imshow(getRmMatrix(img), cmap='Greys')
    plt.subplot(2,3,5)
    plt.imshow(getRmMatrix(img)*img, cmap=plt.cm.afmhot)
    plt.subplot(2,3,6)
    plt.imshow(clearOutsiders(img, False)*img, cmap=plt.cm.afmhot)
    
    plt.tight_layout
    plt.show()

# Extract zones according to average parameters
def extractAverageLungZone(l):
    
    zone, zones = [], []

    for i, e in enumerate(l) :
        if e == 0 : 
            if len(zone) == 0 : pass
            else : 
                zones.append(zone)
                zone = []
        else : zone.append(i)
            
    return getLargest(zones, 1)

# Clear the remaining outsiders
def clearOutsiders(img, graph=False):
    
    im = getRmMatrix(img)
    hi = np.zeros(len(im))
    
    for col in range(len(img)) :
        hi += im[:, col]
        
    def movingAverage(interval, size):
        wi = np.ones(int(size))/float(size)
        return np.convolve(interval, wi, 'same')
    
    hi = movingAverage(hi, 30)
    mx = 0.1*max(hi)
    
    def thresh(x):
        if x < mx : return 0
        else : return x
        
    th = np.vectorize(thresh)
    hi = th(hi)
    
    lungs = extractAverageLungZone(hi)
    if len(lungs) == 0 :
        return np.asarray([[0 for k in range(len(im))] for k in range(len(im))])
    else :
        lungs = lungs[0]
        clear = copy.copy(im)

        for col in range(len(img)) :
            for i in range(len(img)) :
                if i not in lungs :
                    clear[:, col][i] = 0       

        if graph :
            plt.figure(figsize=(18,6))
            plt.subplot(1, 3, 1)
            plt.imshow(im*img, cmap='Greys')
            plt.subplot(1, 3, 2)
            plt.plot(hi, color='orange')
            plt.subplot(1, 3, 3)
            plt.imshow(clear*img, cmap='Greys')

        return clear
