# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:34:51 2017

@author: Deus ExMachina
"""
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from scipy import ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from visual import *

def resample(image, scan, new_spacing=[1,1,1]) :
    
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image

def plot3D(patient,threshold=400) :
    
    image = resample(getImages(patient),loadScans(patient))
    
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
 
def binaryPlot3D(patient) :
    
    image = resample(getImages(patient),loadScans(patient))
    
    def thresh(x,down,up) :
        
        if x < down or x > up :
            return 0
        else :
            return x
        
    threshold = np.vectorize(thresh)
    
    def binar(image) :
        
        nw = image-threshold(image,0,1200)
        
        dx = ndimage.sobel(nw, 1) 
        dy = ndimage.sobel(nw, 0)
        ed = np.hypot(dx,dy)  
        ed *= 255.0 / np.max(ed) 
        
        gT = threshold_otsu(mag)
        bn = ed > gT
        
        return bn
    
    binarize = np.vectorize(binar)
    
    for img in image :
        img = binarize(img)
    
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()