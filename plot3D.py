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
    
    image = resample(getImages(patient),loadScans(patient))
    
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

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

# Display as comparison between two patients
def comparePatients3D(patient1, patient2) :
    
    im1 = np.load('./preprocessed/{}.npy'.format(patient1))
    p1 = im1.transpose(2,1,0)[:,:,::-1]
    verts1, faces1 = measure.marching_cubes(p1, 0)
    print('|-> Ended marching cubes algorithm of patient {}'.format(patient1))
    
    im2 = np.load('./preprocessed/{}.npy'.format(patient2))
    p2 = im2.transpose(2,1,0)[:,:,::-1]
    verts2, faces2 = measure.marching_cubes(p2, 0)
    print('|-> Ended marching cubes algorithm of patient {}'.format(patient2))
    
    fig = plt.figure(figsize=(18, 6))
    alpha = 0.005
    
    ax1 = fig.add_subplot(121, projection='3d')
    mesh1 = Poly3DCollection(verts1[faces1], alpha=alpha)
    mesh1.set_facecolor((0.1, 0.1, 0.1, alpha))
    ax1.add_collection3d(mesh1)
    ax1.set_xlim(0, p1.shape[0])
    ax1.set_ylim(0, p1.shape[1])
    ax1.set_zlim(0, p1.shape[2])
    
    ax2 = fig.add_subplot(122, projection='3d')
    mesh2 = Poly3DCollection(verts2[faces2], alpha=alpha)
    mesh2.set_facecolor((0.1, 0.1, 0.1, alpha))
    ax2.add_collection3d(mesh2)
    ax2.set_xlim(0, p2.shape[0])
    ax2.set_ylim(0, p2.shape[1])
    ax2.set_zlim(0, p2.shape[2])
    
    plt.show()

# Binary plot of 3d object
def binaryPlot3D(patient) :
    
    image = np.load('./Preprocessed/{}.npy'.format(patient))
    
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, 0)
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