
import numpy as np
import shutil
import nibabel as nb
from nilearn.image import load_img
from nilearn import plotting
from nilearn.image import resample_img
import pandas as pd
import os
import preprocessing_entropy
import matplotlib.pyplot as plt
import nibabel.processing

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


def dataLoop(path,labels,folder):
    
    allNames = labels[:,0]
    labels_out = np.zeros((labels.shape[0]*32,labels.shape[1]),dtype='object')
    t = 0
    for name in allNames:
        
        ## Raw
        #path_long = path+'/'+name+'/RAW/'+name+'_mpr-1_anon'
        
        ## Prepocessed
        count = sum((len(n) for _, _, n in os.walk(path+'/'+name+'/RAW')))/3       
        path_long = path+'/'+name+'/PROCESSED/MPRAGE/T88_111/'+name+'_mpr_n'+str(int(count))+'_anon_111_t88_gfc' 

        name_short = folder+'/'+name  
         
        # (1) Save Nifty    
        data = readData(path,path_long,name_short)
        
        # (2) PREPROCESSING
        preprocessed_data = preprocessing_entropy.main(data)

        #print(np.min(preprocessed_data))
        #print(np.max(preprocessed_data))
        #print(np.mean(preprocessed_data)) #around 0
        #print(np.std(preprocessed_data))  #around 1

        print(name)
        
        # (3) Save Slices
        for i in range(32):
            nameCSV = 'DataSliced/'+name_short+'_'+str(i+1)+'.csv'
            np.savetxt(nameCSV, preprocessed_data[:,:,i], delimiter=',')
            labels_out[t*32+i,:] = ['DataSliced/'+name_short+'_'+str(i+1),labels[t,1]]  
            #plt.imshow(preprocessed_data[:,:,i])
            #plt.show()
        t = t+1   

        removeData(path,name,data,folder)
    
    return(labels_out)

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
            
def cutImageDELETE(data,size):
        
    #data._data_cache = data._data_cache[size[0]:size[1],size[2]:size[3],size[4]:size[5]]
    #reduced_data = np.squeeze(data._data_cache, axis = 3)
    data = data[size[0]:size[1],size[2]:size[3],size[4]:size[5]]
    
    return(data)

###### MAIN ######

labels = pd.read_csv("data.csv",sep=';',names=['name','label'])
#labels = labels.iloc[26:27,:]
N = np.size(labels,0)
[test_labels, val_labels, train_labels] = divideData(N,labels,0.2,0.08)

test_labels_all = dataLoop('Data_1',test_labels,'Test')
val_labels_all = dataLoop('Data_1',val_labels,'Val')
train_labels_all = dataLoop('Data_1',train_labels,'Train')

np.save('DataSliced/test_labels.npy',test_labels_all)
np.save('DataSliced/val_labels.npy', val_labels_all)
np.save('DataSliced/train_labels.npy', train_labels_all)