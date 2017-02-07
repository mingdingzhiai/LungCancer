# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:34:51 2017

@author: Deus ExMachina
"""

import dicom
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import skimage.morphology as morph

from skimage.filters import threshold_otsu
from skimage.restoration import inpaint
from scipy import ndimage

def loadScans(patient) :
    
    slices = [dicom.read_file('./input/sample_images/{}/{}'.format(patient,s)) for s in os.listdir('./input/sample_images/{}'.format(patient))]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

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

def displayScan(patient,slices,colormap) :
    
    image = getImages(patient)
    
    plt.figure(figsize=(12,len(slices)*4))
    
    for i,nSlice in enumerate(slices) :
        plt.subplot(len(slices),2,2*i+1)
        plt.imshow(cv2.resize(image[nSlice],(256,256)),cmap=colormap)
        plt.subplot(len(slices),2,2*i+2)
        plt.hist(image[nSlice].flatten(),bins=100,color='orange')
        plt.xlim([-1000,1000])
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
    
    plt.tight_layout
    plt.show
    
def tresh(x,down,up) :
    if x < down or x > up :
        return 0
    else :
        return x
    
def binarize(x) :
    if x == True : return 1
    else : return 0
    
def getSobolImage(img) :
    
    thresh = np.vectorize(tresh)
    binar = np.vectorize(binarize)
    
    dx = ndimage.sobel(img-thresh(img,0,1200), 1) 
    dy = ndimage.sobel(img-thresh(img,0,1200), 0)
    sImage = np.hypot(dx,dy)  
    sImage *= 255.0 / np.max(sImage) 
    g1 = threshold_otsu(sImage)
    sImage = sImage > g1
    sImage = binar(sImage)
    
    return sImage  

def getLaplacianImage(img) :
    
    binar = np.vectorize(binarize)
    thresh = np.vectorize(tresh)
    
    lImage = thresh(img,-1200,-800)
    g2 = threshold_otsu(lImage)
    lImage = lImage > g2
    lImage = binar(lImage)
    
    return lImage

def cutHist(img,nSlice,nColum) :
    
    thresh = np.vectorize(tresh)
    
    sImage = getSobolImage(img)
    lImage = getLaplacianImage(img)
    
    plt.figure(figsize=(12,6))
    
    plt.subplot(2,3,1)
    plt.xlim([-10,522])
    plt.bar(range(len(sImage[nSlice])),list(sImage[nSlice]),width=0.1)
    
    plt.subplot(2,3,3)
    plt.xlim([-10,522])
    plt.bar(range(len(sImage[:,nColum])),list(sImage[:,nColum]),width=0.1)
    
    plt.subplot(2,3,2)
    sFake = sImage
    sFake[nSlice] = 1
    sFake[:,nColum] = 1
    plt.imshow(sFake,cmap="Greys")
    
    plt.subplot(2,3,4)
    plt.xlim([-10,522])
    plt.bar(range(len(lImage[nSlice])),list(lImage[nSlice]),width=0.1)
    
    plt.subplot(2,3,6)
    plt.xlim([-10,522])
    plt.bar(range(len(lImage[:,nColum])),list(lImage[:,nColum]),width=0.1)
    
    plt.subplot(2,3,5)
    lFake = lImage
    lFake[nSlice] = 1
    lFake[:,nColum] = 1
    plt.imshow(lFake,cmap="Greys")
    
    plt.tight_layout
    plt.show()

def getLargest(l,nZones) :
        
    if len(l) <= nZones :
        return l
    else :
        v = [len(e) for e in l]
        w = list(np.argsort(v))
    
    return list(np.asarray(l)[w[-nZones:]])

def getZones(img,ind,typ) :
    
    if typ == "row" :
        return extractZones(img[ind])
    elif typ == "col" :
        return extractZones(img[:,ind])
    else :
        print("|-> Wrong type !")
        
def extractZones(l) :
    
    zones = []
    u = []
    
    for i,v in enumerate(l) :
        if v == 1 :
            u.append(i)
        else :
            if len(u) == 0 : pass
            else : 
                zones.append(u)
                u = []

    return zones

def getRmMatrix(img) :
    
    mat = np.matrix([[1 for k in range(len(img))] for k in range(len(img))])
    
    img = getLaplacianImage(img)
    
    for nCol in range(len(img)) :
        zones = getZones(img,nCol,"col")
        if len(zones) == 0 : pass
        elif len(zones) == 1 : 
            for ind in zones[0] : mat[ind,nCol] = 0
        elif len(zones) == 2 : 
            for zone in zones : 
                for ind in zone : mat[ind,nCol] = 0                  
        else :
            zones = getLargest(zones,2)
            for zone in zones : 
                for ind in zone : mat[ind,nCol] = 0
                    
    for nRow in range(len(img)) :
        zones = getZones(img,nRow,"row")
        if len(zones) == 0 : pass
        elif len(zones) == 1 : 
            for ind in zones[0] : mat[nRow,ind] = 0
        elif len(zones) == 2 : 
            for zone in zones : 
                for ind in zone : mat[nRow,ind] = 0
        else :
            zones = getLargest(zones,3)
            for zone in zones : 
                for ind in zone : mat[nRow,ind] = 0
    
    binar = np.vectorize(binarize)
    
    for nRow in range(len(img)-int(len(img)/4),len(img)) :
        mat[nRow,:] = 0
    
    return np.asarray(binar(mat))*img

def observeExtraction(img) :
    
    plt.figure(figsize=(12,8))
    
    plt.subplot(2,3,1)
    plt.imshow(img,cmap=plt.cm.magma)
    plt.subplot(2,3,2)
    plt.imshow(getLaplacianImage(img),cmap="Greys")
    plt.subplot(2,3,3)
    plt.imshow(getRmMatrix(img),cmap="Greys")
    plt.subplot(2,3,4)
    plt.imshow(img,cmap=plt.cm.bone)
    plt.subplot(2,3,5)
    plt.imshow(getSobolImage(img),cmap="Greys")
    plt.subplot(2,3,6)
    plt.imshow(getRmMatrix(img)*img,cmap=plt.cm.magma)
    
    plt.tight_layout
    plt.show()