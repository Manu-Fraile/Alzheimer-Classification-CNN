import numpy as np


def ReadLabels(path):
    
    labels = np.load(path,allow_pickle=True)
    labels = labels[np.argsort(labels[:, 0])]
    labelsName = labels[:,0]
    
    labelsNum = np.float64(labels[:,1])    
    labelsNum[labelsNum > 0] = 1
    labelsNum = -(labelsNum-1)
    labelsNum = np.expand_dims(labelsNum,1)

    labelsNum=np.float64(np.int16(np.append(labelsNum,-(labelsNum-1),axis=1)))
    
    return(labelsName,labelsNum)

def ReadData(path,names):
    
    first = True
    for name in names:
        
        namePath = path+'/'+name+'.csv'
        dataSlice = np.loadtxt(namePath,delimiter=',')
        
        if first:            
            data = np.expand_dims(dataSlice,0) 
            first = False
        else:
            data = np.append(data,np.expand_dims(dataSlice,0),axis=0)
    
    return(data)

####### MAIN ########

def main(folder):

    [test_name,y_test] = ReadLabels(folder+'/test_labels.npy')
    x_test = ReadData(folder+'/Test',test_name)

    [train_name,y_train] = ReadLabels(folder+'/train_labels.npy')
    x_train = ReadData(folder+'/Train',train_name)
    
    [valid_name,y_valid] = ReadLabels(folder+'/val_labels.npy')
    x_valid = ReadData(folder+'/Val',valid_name)
    
    ## Saving ##
    np.save(folder+'/y_test.npy',y_test)
    np.save(folder+'/x_test.npy',x_test)
    np.save(folder+'/y_train.npy',y_train)
    np.save(folder+'/x_train.npy',x_train)
    np.save(folder+'/y_valid.npy',y_valid)
    np.save(folder+'/x_valid.npy',x_valid)
    


