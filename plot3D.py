# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:34:51 2017

@author: Dindin Meryll
"""

from visual import *

# Readapt the pixel space for uniformed format
def resample(image, scan, new_spacing=[1,1,1]) :
    
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image

# Display the 3d recreated image
def plot3D(patient, threshold=400) :
    
    image = resample(getImages(patient), loadScans(patient))
    
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces, _, _ = measure.marching_cubes_lewiner(p, level=threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.05)
    mesh.set_facecolor((0.2, 0.2, 0.2, 0.05))
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
  
# Binarize that same 3d object
def getBinary(patient) :
    
    image = resample(getImages(patient), loadScans(patient))
    
    for nSlice in range(len(image)) :
        try : image[nSlice] = clearOutsiders(image[nSlice], False)*image[nSlice]
        except :
            print("|-> Problematic slice : {}".format(nSlice))
            image[nSlice] = np.asarray([[0 for k in range(len(image[nSlice]))] for k in range(len(image[nSlice]))])
        
    return image

# Binary plot of 3d object
def binaryPlot3D(img) :
    
    p = img.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces, _, _ = measure.marching_cubes_lewiner(p, level=0)
    print('|-> Ended marching cubes algorithm of patient {}'.format(patient))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    alpha = 0.005
    # Alpha on the mesh will put edges transparent
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    # Alpha on the facecolor will turn the faces transparent
    mesh.set_facecolor((0.2, 0.2, 0.2, alpha))
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
