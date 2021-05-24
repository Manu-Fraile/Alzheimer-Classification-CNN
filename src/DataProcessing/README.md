# Data Processing

Run *start.py* for any of the three cuts of the 3D MRI scan. Also for data augmentation using rotation on the images between -5° and 5° randomly for three times, as done in [1]. Change the dataLoop function for the desired dataset.

Run *start_entropy.py* for data augmentation using an entropy-based sorting mechanism taking 32 cuts of the axial images in order to get the most relevant cuts out of the MRI scan [2].

## Requirements
* pip install nilearn
* pip install dltk
* pip install SimpleITK
* pip install nipype
* pip install opencv-python
* pip install intensipy
* pip install deepbrain
* pip install tensorflow==1.15

## References

[1] A. M. Taqi, A. Awad, F. Al-Azzo and M. Milanova, "The Impact of Multi-Optimizers and Data Augmentation on TensorFlow Convolutional Neural Network Performance," 2018 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR), 2018, pp. 140-145, doi: 10.1109/MIPR.2018.00032.

[2] M. Hon and N. M. Khan, "Towards Alzheimer's disease classification through transfer learning," 2017 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2017, pp. 1166-1169, doi: 10.1109/BIBM.2017.8217822.
