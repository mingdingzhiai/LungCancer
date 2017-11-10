import dicom
import copy
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import skimage.morphology as morph

from skimage.filters import threshold_otsu
from scipy import ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection