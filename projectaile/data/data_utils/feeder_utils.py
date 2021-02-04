import numpy as np


'''
    utility functions for the feeder
'''

'''
    split_dset : Split the given data in train and validation sets
    
    indices : The indices of the data
    split_size : The ratio of the validation set
    shuffle : Whether to randomly shuffle the data before splitting
    
    returns : 
    train_indices : Indices of the training set
    valid_indices : Indices of the validation set
'''
def split_dset(indices, split_size, shuffle=True):
    if shuffle:
        np.random.shuffle(indices)
        
    split_index = len(indices)-int(len(indices)*split_size)
    train_indices = indices[:split_index]
    valid_indices = indices[split_index:]
    
    return train_indices, valid_indices    


'''
    k_fold : Perform K-Fold Data Splitting
    
    k : number of folds
    shuffle : Whether to shuffle data before splitting
    repeat : How many times to repeat k_fold operation
'''
def k_fold(indices, k=5, shuffle=True, repeat=1):
    iterations = {}
    samples_per_fold = len(indices)//k
    
    for i in range(repeat):
        if shuffle:
            np.random.shuffle(indices)
                
        for j in range(k):
            print('TODO')    
        
        
    return