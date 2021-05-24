# SAGGITAL CUT

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

def norm(data):
    data_dev = np.std(data)
    data_mean = np.mean(data)
    if data_dev > 0:
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
    cut = result[92, :, :]#result[:, :, 92]
    #resized = cv.resize(cut, (160, 256), interpolation=cv.INTER_NEAREST)  
    resized = np.pad(cut, ((0, 0), (16,16), (0,0)), mode='constant')
    
    return resized


def main(data):
    
    # DATA NORMALIZATION

    data_norm = load_img(data) #load_img(normfilename)

    # SKULL STRIPPING
    brain_tissue = skull_strip(data_norm) 
    normalized = norm(brain_tissue)

    # image=nib.Nifti1Image(brain_tissue, affine=np.eye(4))
    # skullstrip_filename = "skullstrippedimg.nii"
    # nib.save(image, skullstrip_filename)

    return normalized
