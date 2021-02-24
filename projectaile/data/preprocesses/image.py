# Different Preprocessing Steps.

# Images

from .preprocesses import PREPROCESS

import cv2
import numpy as np


'''
    RESIZE : PREPROCESS : resize images to given width and height or calculate one
    to preserve the aspect ratio.
'''
class RESIZE(PREPROCESS):
    def __init__(self, apply_on_targets=False, width=None, height=None, keep_ar=False, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.keep_ar = keep_ar
        self.interpolation = interpolation
        super(RESIZE, self).__init__(apply_on_targets)
        
    def init(self):
        if not self.width and not self.height:
            print('Both target width and height cant be None!')
            exit(0)
        else:
            if not self.width or not self.height:
                non_nan = self.width if self.width else self.height
                if self.keep_ar:
                    # None Dimension will be calculated according to the first image and will
                    # be used as target dimension for the rest of the images
                    target_size = (self.width, self.height)
                else:
                    target_size = (non_nan, non_nan)
            else:
                target_size = (self.width, self.height)
                    
        props = {
            'target_size': target_size, 
            'keep_ar' : self.keep_ar, 
            'interpolation' : self.interpolation
        }
            
        return props
        
    def _call(self, inp, target_size, interpolation, keep_ar):
        h, w = inp.shape[:2]
        update_params = False
        if keep_ar:
            width, height = target_size
            if width is None:
                r = height / float(h)
                dim = (int(w * r), height)
            elif height is None:
                r = width / float(w)
                dim = (width, int(h * r))
            target_size = dim
            keep_ar = False
            update_params = True
                
        out = cv2.resize(inp, target_size, interpolation=interpolation)
            
        if update_params:
            return {
                'target_size' : target_size,
                'keep_ar' : keep_ar
            }, out
        else:
            return {}, out


def gray_scale():
    return

def color_normalize():
    return

def mask():
    return