import numpy as np


'''
    utility functions for the feeder
'''

'''
    resolve_path : Resolves the loading path based on the data path and passed path
    
'''
def resolve_path(dset_conf, file_path):
    file_path = file_path.replace('\\', '/')
    file_path_comps = file.split('/')
        
    if dset_conf.DATASET.DATA_PATH:
        root_path = dset_conf.DATASET.DATA_PATH
        sub_dirs = parse_dirs(dset_conf.DATASET)
    else:
        root_path = [
            dset_conf.TRAIN_DATA.DATA_PATH,
            dset_conf.VALID_DATA.DATA_PATH
        ]


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
    stratified : Whether to keep class ratios in each fold same
'''
class KFOLD:
    def __init__(self, x, y, k=5, shuffle=True, repeat=1, stratified=True):
        self.x = x
        self.y = y
        self.indices = []
        self.num_folds = k
        self.shuffle = shuffle
        self.repeat = repeat
        self.iteration = 1
        self.stratified = stratified
        
    def get_folds(self):
        if self.iteration > self.repeat:
            return
        else:
            if self.shuffle:
                np.random.shuffle(self.indices)
                
            elems_per_fold = len(self.x)//self.num_folds

            