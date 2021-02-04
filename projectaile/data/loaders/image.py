from .loader import LOADER
import numpy as np

class IMAGE_LOADER(LOADER):
    def __init__(self, config):
        super(IMAGE_LOADER, self).__init__(config)
        
    def get_batch(self, x, y):
        train_x = []
        train_y = []
            
        for i in range(len(x)):
            img = cv2.imread(x[i])
            if self.config.MODEL.PROBLEM_TYPE == 'segmentation':
                target = cv2.imread(y[i])
            elif self.config.MODEL.PROBLEM_TYPE == 'coco':
                target = coco_loader(y[i])
            elif self.config.MODEL.PROBLEM_TYPE == 'pascal':
                target = voc_loader(y[i])
            else:
                target = y[i]
                
            train_x.append(img)
            train_y.append(target)
                
        return np.array(train_x), np.array(train_y)