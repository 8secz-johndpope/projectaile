import os
import numpy as np
import pandas as pd
from .feeder_utils import *

'''
    Different parsers for extracting features, targets, load_paths etc. from
    different interface types.
'''

'''
    get_features_labels_from_dirs : Get feature and target lists for a directory interface
    
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


def dir_parser(config):
    split_data = config.DATA.SPLIT_DATA
    feature_names = config.DATA.DATASET.FEATURES
    target_names = config.DATA.DATASET.TARGETS
    
    if split_data:
        root_path = config.DATA.DATASET.DATA_PATH
        dirs = config.DATA.DATASET.DIRECTORIES
                        
        features, targets = get_features_labels_from_dirs(root_path, dirs, feature_names, target_names)
        indices = np.arange(0, len(features))
        train_indices, valid_indices = split_dset(indices, config.DATA.VALID_SPLIT_RATIO)
                        
    else:
        # Train Data
        train_root_path = config.DATA.DATASET.TRAIN_DATA.DATA_PATH
        train_dirs = config.DATA.DATASET.TRAIN_DATA.DIRECTORIES

        # Valid Data
        valid_root_path = config.DATA.DATASET.VALID_DATA.DATA_PATH
        valid_dirs = config.DATA.DATASET.VALID_DATA.DIRECTORIES
                            
        train_features, train_targets = get_features_labels_from_dirs(train_root_path, train_dirs, self.feature_names, self.target_names)
        valid_features, valid_targets = get_features_labels_from_dirs(valid_root_path, valid_dirs, self.feature_names, self.target_names)
                        
        self.train_indices = np.arange(0, len(self.train_features))
        self.valid_indices = np.arange(0, len(self.valid_features))


'''
    Extract Dataset Information, i.e., features, targets and load_paths from csv files.
'''
def csv_parser(config):
    split_data = config.DATA.SPLIT_DATA
    feature_names = config.DATA.DATASET.FEATURES
    target_names = config.DATA.DATASET.TARGETS
    
    train_x, train_y, valid_x, valid_y = [], [], [], []
    
    if split_data:
        split_size = config.DATA.VALID_SPLIT_SIZE
        fl = config.DATA.DATASET.INTERFACE_FILE
        root_path = config.DATA.DATASET.DATA_PATH
        df = pd.read_csv(root_path+fl)
        indices = np.arange(0, len(df))
        train_indices, valid_indices = split_dset(indices, split_size)
        features = df[[feature_names]].to_numpy()
        targets = df[[target_names]].to_numpy()
        
        train_x = features[train_indices]
        train_y = targets[train_indices]
        valid_x = features[valid_indices]
        valid_y = targets[valid_indices]
    
    # If train and valid datasets are provided separately.
    else:
        train_fl = config.DATA.DATASET.TRAIN_DATA.INTERFACE_FILE
        train_root_path = config.DATA.DATASET.TRAIN_DATA.DATA_PATH
        valid_fl = config.DATA.DATASET.VALID_DATA.INTERFACE_FILE
        valid_root_path = config.DATA.DATASET.VALID_DATA.DATA_PATH
                        
        train_df = pd.read_csv(train_root_path+train_fl)
        valid_df = pd.read_csv(valid_root_path+valid_fl)
        
        train_x = train_df[[feature_names]].to_numpy()
        train_y = train_df[[target_names]].to_numpy()
        valid_x = valid_df[[feature_names]].to_numpy()
        valid_y = valid_df[[target_names]].to_numpy()
        
    return np.array(train_x), np.array(valid_x), np.array(train_y), np.array(valid_y)