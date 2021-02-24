import numpy as np


'''
    An abstract class for callable that will be used in a pipeline
'''
class CALLABLE:
    def __init__(self, apply_on_targets=False, **kwargs):
        self.apply_on_targets = apply_on_targets
        self.__dict__.update(kwargs)
        self.props = self.init()
        
    def init(self):
        print('Please overwrite this method with the initialization steps.')
        return {}
        
    def _call(self, x, **kwargs):
        print('Please overwrite this method with the actual function to be executed on one example.')
        return _, _
            
    def __call__(self, x, y):
        features, targets = [], []
            
        for ftr in x:
            updates, out = self._call(ftr, **self.props)
            features.append(out)
            self.props.update(updates)
            
        if self.apply_on_targets:
            for lbl in y:
                updates, out = self._call(lbl, **self.props)
                targets.append(out)
                self.props.update(updates)
        else:
            targets = y
            
        return np.array(features), np.array(targets)