from projectaile.utils import PIPELINE, CALLABLE


class AUGMENTATION(CALLABLE):
    def __init__(self, params):
        super(AUGMENTATION, self).__init__(params)
        
    def __call__(self, x, y):
        return

class AUGMENTOR(PIPELINE):
    def __init__(self, config):
        super(AUGMENTOR, self).__init__(config)