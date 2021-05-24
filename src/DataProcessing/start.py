
import numpy as np
import shutil
import nibabel as nb
from nilearn.image import load_img
from nilearn import plotting
from nilearn.image import resample_img
import pandas as pd
import os
import preprocessing_axial
import preprocessing_cor
import preprocessing_sag
import preprocessing_rotate
import matplotlib.pyplot as plt
import nibabel.processing
import createFiles

def readData(path,name,name_short):

    fname = name + '.img'  
    img = nb.load(fname)
    fnamenii = name_short + '.nii'
    nb.save(img, fnamenii)

    return(fnamenii)

def removeData(path, name, data, folder):
    #delete SUBJ_111, FSL_SEG and previuous nii file
    if os.path.exists(path+'/'+name+'/FSL_SEG/'):
        shutil.rmtree(path+'/'+name+'/FSL_SEG/')
    if os.path.exists(path+'/'+name+'/PROCESSED/MPRAGE/SUBJ_111/'):
        shutil.rmtree(path+'/'+name+'/PROCESSED/MPRAGE/SUBJ_111/')
    os.remove(data)

def dataLoop(path,allNames,folder):
    
    tAll = allNames.shape[0]
    t = 1
    for name in allNames:
        
        print(folder+':'+str(t)+'/'+str(tAll))
        t = t+1
        
        ## Raw
        #path_long = path+'/'+name+'/RAW/'+name+'_mpr-1_anon'
        
        ## Prepocessed
        count = sum((len(n) for _, _, n in os.walk(path+'/'+name+'/RAW')))/3       
        path_long = path+'/'+name+'/PROCESSED/MPRAGE/T88_111/'+name+'_mpr_n'+str(int(count))+'_anon_111_t88_gfc' 

        name_short = folder+'/'+name  
         
        # (1) Save Nifty    
        data = readData(path,path_long,name_short)
        
        # (2) PREPROCESSING --> CHANGE FOR THE PRE-PROCESSING 
        preprocessed_data = preprocessing_axial.main(data) #axial cut
        #preprocessed_data = preprocessing_cor.main(data) #coronal cut
        #preprocessed_data = preprocessing_sag.main(data) #saggital cut
        #preprocessed_data = preprocessing_rotate.main(data) #data augmentation, rotation
        preprocessed_data = preprocessed_data[:, :, 0]

        #plt.imshow(preprocessed_data)
        #plt.show()

        #print(np.min(preprocessed_data))
        #print(np.max(preprocessed_data))
        #print(np.mean(preprocessed_data)) #around 0
        #print(np.std(preprocessed_data))  #around 1
        
        # (3) Save Slice
        nameCSV = 'DataSliced/'+name_short+'.csv'
        np.savetxt(nameCSV, preprocessed_data, delimiter=',')
        removeData(path,name,data,folder)


def divideData(N,labels,test,val):

    permute = list(range(0,N))
    np.random.shuffle(permute)
    
    N_train = round(N*test) # 0 < test < 1
    N_val = round(N*val)
    test_range = permute[:N_train]
    val_range = permute[N_train:N_train+N_val]
    train_range = permute[N_train+N_val:]
    
    test_data = np.array(labels.iloc[test_range])
    val_data = np.array(labels.iloc[val_range])
    train_data = np.array(labels.iloc[train_range])
    
    return(test_data,val_data,train_data)
            
def cutImage(data,size):
        
    #data._data_cache = data._data_cache[size[0]:size[1],size[2]:size[3],size[4]:size[5]]
    #reduced_data = np.squeeze(data._data_cache, axis = 3)
    data = data[size[0]:size[1],size[2]:size[3],size[4]:size[5]]
    
    return(data)

###### MAIN ######

labels = pd.read_csv("data.csv",sep=';',names=['name','label'])
#labels = labels.iloc[19:20,:] 
print(labels)
N = np.size(labels,0)
[test_labels, val_labels, train_labels] = divideData(N,labels,0.2,0.08)

np.save('DataSliced/test_labels.npy',test_labels)
np.save('DataSliced/val_labels.npy', val_labels)
np.save('DataSliced/train_labels.npy', train_labels)

dataLoop('Data_1',test_labels[:,0],'Test')
dataLoop('Data_1',val_labels[:,0],'Val')
dataLoop('Data_1',train_labels[:,0],'Train')

createFiles.main('DataSliced')
