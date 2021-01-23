import os
import numpy as np
import pandas as pd

'''
    Utility Functions For Data Feeder And Loaders
'''

def get_features_labels_from_dirs(root, dirs, features, targets):
    if dirs == '*':
        dirs = next(os.walk(root_path))[1]
    elif dirs == '':
        dirs = []
            
    if targets == '__dirname__':
        targets = dirs
            
    # Targets can also be file_name_regex like : *_mask.jpg|*_mask.png or path to directory like /masks or a hybrid like /masks/*_mask.jpg|*_mask.png, every file in that dir will be mapped to the feature file using feature file name
    elif '/' in targets:
        # target is a directory and filenames need to be mapped.
        # Iter the train files and check for the equivalent file in the target_dir
        pass
    # Targets can also be elements like the bounding box arrays or the mask encodings like in the HuBMAP competition.
    else:
        pass
            
    return


def get_dset_info_for_dirs():
    if self.split_data:
        root_path = self.config.DATA.DATASET.DATA_PATH
        dirs = self.config.DATA.DATASET.DIRECTORIES
                        
        features, targets = get_features_labels_from_dirs(root_path, dirs, self.feature_names, self.target_names)
        indices = np.arange(0, len(features))
        self.split_dset(indices)
                        
    else:
        # Train Data
        train_root_path = self.config.DATA.DATASET.TRAIN_DATA.DATA_PATH
        train_dirs = self.config.DATA.DATASET.TRAIN_DATA.DIRECTORIES

        # Valid Data
        valid_root_path = self.config.DATA.DATASET.VALID_DATA.DATA_PATH
        valid_dirs = self.config.DATA.DATASET.VALID_DATA.DIRECTORIES
                            
        train_features, train_targets = get_features_labels_from_dirs(train_root_path, train_dirs, self.feature_names, self.target_names)
        valid_features, valid_targets = get_features_labels_from_dirs(valid_root_path, valid_dirs, self.feature_names, self.target_names)
                        
        self.train_indices = np.arange(0, len(self.train_features))
        self.valid_indices = np.arange(0, len(self.valid_features))


def split_dset(indices, split_size):
    np.random.shuffle(indices)
    split_index = len(indices)-int(len(indices)*split_size)
    train_indices = indices[:split_index]
    valid_indices = indices[split_index:]
    
    return train_indices, valid_indices
    


'''
    Extract Dataset Information, i.e., features, targets and load_paths from csv files.
'''
def get_features_labels_from_csv(config):
    if self.split_data:
        fl = self.config.DATA.DATASET.INTERFACE_FILE
        root_path = self.config.DATA.DATASET.DATA_PATH
        df = pd.read_csv(root_path+fl)
        indices = np.arange(0, len(df))
        split_dset(indices)
    # If train and valid datasets are provided separately.
    else:
        train_fl = self.config.DATA.DATASET.TRAIN_DATA.INTERFACE_FILE
        train_root_path = self.config.DATA.DATASET.TRAIN_DATA.DATA_PATH
        valid_fl = self.config.DATA.DATASET.VALID_DATA.INTERFACE_FILE
        valid_root_path = self.config.DATA.DATASET.VALID_DATA.DATA_PATH
                        
        self.train_indices = np.arange(0, len(pd.read_csv(train_root_path+train_fl)))
        self.valid_indices = np.arange(0, len(pd.read_csv(valid_root_path+valid_fl)))