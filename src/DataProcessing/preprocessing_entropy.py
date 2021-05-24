#DATA AUGMENTATION: AXIAL SLICES ENTROPY

from nilearn.image.resampling import resample_img
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from deepbrain import Extractor
import SimpleITK as sitk
from nilearn import plotting
from nilearn.image import load_img
import cv2 as cv
import os
from skimage.measure import shannon_entropy

def norm(data,dim):
    
    dim_keep = tuple(np.delete(np.array([0,1,2]),dim))
    data_dev = np.std(data,dim_keep)
    data_mean = np.mean(data,dim_keep)
    
    if min(data_dev) > 0:
        ret = (data - data_mean) / data_dev
    else:
        ret = data * 0.
    return ret

def skull_strip(image):
    img = image.get_fdata()
    ext = Extractor()

    prob = ext.run(img)  # 3d numpy array with prob that the pixel comtains brain tissue
    mask = prob > 0.5  # mask of the mri-brain tissue
    img = np.array(img)
    mask = np.array(mask)

    result = img * mask
    
    return result

def cutting(img):
    
    #cut = img[:, :, 70] # 88
    # RESIZING JUST FOR RAW IMAGES NEEDED
    #resized = cv.resize(cut, (160, 256), interpolation=cv.INTER_NEAREST) 
    #print(cut.shape)    
    
    resized = np.pad(img, ((16, 16), (0,0), (0,0)), mode='constant')
    
    return(resized)

def entropy(img,dim):
    
    all_entropy = np.zeros(img.shape[dim])

    if dim == 0:
        for i in range(img.shape[dim]):
            all_entropy[i] = shannon_entropy(img[i,:,:])
        
        threshold = np.sort(all_entropy)[-33]
        reduced_img = img[all_entropy > threshold,:,:] 
        
    elif dim == 1:
         for i in range(img.shape[dim]):
            all_entropy[i] = shannon_entropy(img[:,i,:])
            
         threshold = np.sort(all_entropy)[-33]
         reduced_img = img[:,all_entropy > threshold,:] 
            
    elif dim == 2:
         for i in range(img.shape[dim]):
             all_entropy[i] = shannon_entropy(img[:,:,i])
         
         threshold = np.sort(all_entropy)[-33]
         reduced_img = img[:,:,all_entropy > threshold] 
      
    return(reduced_img)

####### MAIN #######

def main(data):
    
    # LOAD IMAGE
    data_raw = load_img(data) #load_img(normfilename)

    # SKULL STRIPPING
    brain_tissue = skull_strip(data_raw) 

    # CUTTING
    data_reduced = entropy(brain_tissue[:,:,:,0],2)
    data_cut = cutting(data_reduced)
    
    # DATA NORMALIZATION
    data_normal = norm(data_cut,2)
    
    return data_normal
