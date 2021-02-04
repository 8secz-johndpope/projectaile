import numpy as np


'''
    to_numeric : convert to numeric ids
'''
def to_numeric(labels):
    targets = np.zeros(len(labels))
    unique_classes = np.unique(labels)
    ids = np.arange(0, len(labels))
    id_map = dict(zip(unique_classes, ids))
    
    for i in range(len(labels)):
        targets[i] = id_map[labels[i]]
    
    return targets
    
'''
    onehot_encode : returns one-hot encoded labels
'''
def onehot_encode(labels, id_map={}):
    if not id_map:
        unique_classes = np.unique(labels)
        ids = np.arange(0, len(unique_classes))
        id_map = dict(zip(unique_classes, ids))
    
    num_classes = len(id_map.keys())
    
    targets = np.zeros((len(labels), num_classes))    
    
    for i in range(len(labels)):
        targets[i, id_map[labels[i]]] = 1
        
    return targets