import numpy as np

'''
    LOADER : LOADER class for loading batches according to the indices passed
            from the feeder and apply preprocessing and augmentations
'''

class LOADER:
    '''
        constructor : make sense of the config and extract and assign important
                    properties
                    
        config : configuration object
    '''
    def __init__(self, config):
        self.config = config
        dataset = self.config.DATA
        
        if dataset.SPLIT:
            if dataset.DATA_TYPE != 'structured':
                self.train_data_path = dataset.DATASET.TRAIN_DATA.DATA_PATH
                self.valid_data_path = dataset.DATASET.VALID_DATA.DATA_PATH
            else:
                self.train_data_path = dataset.DATASET.TRAIN_DATA.INTERFACE_FILE
                self.valid_data_path = dataset.DATASET.VALID_DATA.INTERFACE_FILE
        else:
            if dataset.DATA_TYPE != 'structured':
                self.train_data_path = self.valid_data_path = dataset.DATASET.DATA_PATH
            else:
                pass
            
        self.interface_type = dataset.DATASET.INTERFACE_TYPE
                
    '''
        get_indices : extract data paths or features and targets and indices for shuffling
                    and batch generation
    '''
    def get_indices(self):
        self.features = []
        self.targets = []
        self.indices = np.arange(len(self.features))
        
    def load_batch(self):
        pass